"""Visualization for EntityGraph — color-coded, renderable, zero-dependency.

Five output formats, no required external dependencies:

  - to_cytoscape(): Cytoscape.js JSON, embeddable in any web UI
  - to_d3_json():    D3 force-graph JSON
  - to_mermaid():    Mermaid flowchart string for docs/notebooks
  - to_graphml():    GraphML XML for Gephi/yEd
  - render_html():   Self-contained interactive HTML (Cytoscape.js via CDN)

Color coding:
  - by="community" (default if computed): each community a distinct color
  - by="type":      entity=blue, concept=green, list-derived=orange,
                    categorical=purple, range=gray
  - by=<column>:    color entity nodes by their value of the given column

Edge coloring:
  - SIMILAR_TO edges shown dashed
  - Entity-to-entity edges (e.g. COMPETES_WITH) in red (real relationships)
  - Feature edges in muted gray
  - Width ∝ edge weight

Quick start::

    from pydantic_ai_cloudflare import EntityGraph

    kg = EntityGraph()
    await kg.build_from_records(data, id_column="account_id")
    kg.compute_features()  # populate communities for color-coding

    # Interactive HTML in the browser
    kg.render_html("graph.html", color_by="community", max_nodes=200)

    # Cytoscape.js JSON for embedding in your own app
    spec = kg.to_cytoscape(color_by="community")

    # Mermaid for Markdown / Confluence / GitLab
    print(kg.to_mermaid(max_nodes=30))

    # GraphML for Gephi or yEd
    kg.to_graphml("graph.graphml")
"""

from __future__ import annotations

import colorsys
import html as _html
import json as _json
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph import EntityGraph

logger = logging.getLogger(__name__)


# ============================================================
# Color palettes
# ============================================================

# Distinct, accessibility-friendly community palette (Tableau 20 + extras).
_COMMUNITY_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

# Node-type palette (used when color_by="type")
_TYPE_PALETTE: dict[str, str] = {
    "entity": "#3B82F6",  # blue — the main thing
    "concept": "#10B981",  # green — LLM extracted
}

_TYPE_FALLBACK_RANGE = "#9CA3AF"  # gray for *_range nodes
_TYPE_FALLBACK_LIST = "#F59E0B"  # orange for list-derived
_TYPE_FALLBACK_CATEGORICAL = "#8B5CF6"  # purple

# Edge styling defaults
_EDGE_FEATURE = "#CBD5E1"  # muted gray
_EDGE_ENTITY = "#DC2626"  # red — direct entity relationships
_EDGE_SIMILAR = "#6B7280"  # gray, will be drawn dashed


# ============================================================
# Color helpers
# ============================================================


def _hash_to_color(text: str) -> str:
    """Deterministic but distinct color for an arbitrary string."""
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.65)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def _community_color(cid: int) -> str:
    """Pick a community color from a palette, looping if many communities."""
    if cid < 0:
        return "#D1D5DB"  # light gray for unassigned
    return _COMMUNITY_PALETTE[cid % len(_COMMUNITY_PALETTE)]


def _color_for_node(
    node: dict[str, Any],
    *,
    mode: str,
    communities: dict[str, int] | None,
    column_value_colors: dict[str, str] | None,
    column: str | None,
) -> str:
    """Resolve a node's display color.

    mode ∈ {"community", "type", "column"}.
    """
    ntype = node.get("type", "")
    nid = node.get("id", "")

    if mode == "community" and communities and ntype == "entity":
        return _community_color(communities.get(nid, -1))

    if mode == "column" and ntype == "entity" and column and column_value_colors is not None:
        data = node.get("data", {}) or {}
        value = str(data.get(column, "")).strip().lower()
        if value:
            if value not in column_value_colors:
                column_value_colors[value] = _hash_to_color(value)
            return column_value_colors[value]

    # Fallback: type-based coloring
    if ntype in _TYPE_PALETTE:
        return _TYPE_PALETTE[ntype]
    if ntype.endswith("_range"):
        return _TYPE_FALLBACK_RANGE
    # List-derived nodes have a known list_columns mapping; we can't
    # tell here without metadata, so default to "categorical" style.
    return _TYPE_FALLBACK_CATEGORICAL


def _color_for_edge(etype: str) -> tuple[str, str]:
    """Return (color, style) for an edge type. style ∈ {'solid', 'dashed'}."""
    if etype == "SIMILAR_TO":
        return _EDGE_SIMILAR, "dashed"
    # Direct entity-to-entity edges (extracted relationships, COMPETES_WITH, etc.)
    if etype in {
        "COMPETES_WITH",
        "PARTNERS_WITH",
        "ACQUIRED",
        "DISPLACED",
        "MIGRATED_TO",
        "USES",
        "SUPPLIES",
        "REFERRED_BY",
        "RELATED_TO",
    }:
        return _EDGE_ENTITY, "solid"
    return _EDGE_FEATURE, "solid"


# ============================================================
# Subgraph selection
# ============================================================


def _truncate_raw_data(
    data: dict[str, Any],
    *,
    max_chars: int | None,
) -> dict[str, Any]:
    """Return a shallow copy of ``data`` with long string values truncated.

    Used by :meth:`GraphVisualizer.to_cytoscape` to keep visualization
    payloads small. Only ``str`` values longer than ``max_chars`` are
    truncated; other types pass through unchanged. ``max_chars=None``
    means no truncation.
    """
    if max_chars is None:
        return dict(data)
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > max_chars:
            out[k] = v[:max_chars] + "…"
        else:
            out[k] = v
    return out


def _select_subgraph(
    kg: EntityGraph,
    *,
    max_nodes: int | None,
    focus: str | None,
    hops: int,
    include_node_types: list[str] | None,
    exclude_node_types: list[str] | None,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Pick a subset of nodes/edges suitable for rendering.

    Strategy:
      - If `focus` is given, take that entity's k-hop neighborhood.
      - Else, rank entities by degree and take the top
        max_nodes/2 entities, plus their feature neighbors up to max_nodes.
      - If max_nodes is None, return the full graph (use sparingly).
    """
    nodes = kg._nodes
    typed_adj = kg._typed_adj
    adj = kg._adj
    entity_ids = kg._entity_ids

    keep_types = set(include_node_types) if include_node_types else None
    skip_types = set(exclude_node_types or [])

    selected: set[str] = set()
    use_default_selection = not focus

    if focus:
        resolved = kg._resolve_entity(focus)
        if resolved:
            selected.add(resolved)
            frontier = {resolved}
            for _ in range(max(1, hops)):
                nxt: set[str] = set()
                for n in frontier:
                    for e in typed_adj.get(n, []):
                        t = e.get("target")
                        if t and t not in selected:
                            nxt.add(t)
                selected |= nxt
                frontier = nxt
            # Isolated focus node — warn but keep the single node so the
            # caller still sees the entity rather than an empty graph.
            if len(selected) == 1:
                logger.warning(
                    f"focus={focus!r} resolved but has no neighbors within "
                    f"{hops} hop(s). Returning the single isolated node."
                )
        else:
            # Unresolved focus is the silent-empty bug from CF1 testing.
            # Fall back to the default selection so users get *something*
            # and emit a clear warning so they can fix the focus arg.
            logger.warning(
                f"focus={focus!r} did not resolve to any entity in the graph. "
                f"Falling back to default selection (top entities by degree). "
                f"Pass an entity ID or label that exists in the graph."
            )
            use_default_selection = True

    if use_default_selection:
        # Rank entities by degree and take the top
        ranked_entities = sorted(
            entity_ids,
            key=lambda n: -len(adj.get(n, set())),
        )
        if max_nodes is None:
            selected.update(nodes.keys())
        else:
            entity_budget = max(1, max_nodes // 2)
            selected.update(ranked_entities[:entity_budget])

            # Add their feature neighbors, capped at max_nodes
            for ent in list(selected):
                for nbr in adj.get(ent, set()):
                    if len(selected) >= max_nodes:
                        break
                    selected.add(nbr)
                if len(selected) >= max_nodes:
                    break

    # Apply type filters
    if keep_types or skip_types:
        before = len(selected)
        filtered = set()
        for nid in selected:
            ntype = nodes.get(nid, {}).get("type", "")
            if keep_types and ntype not in keep_types:
                continue
            if ntype in skip_types:
                continue
            filtered.add(nid)
        selected = filtered
        # Friendlier feedback when filters wipe out the whole subgraph
        # (real CF1 case: passing exclude_node_types covering every type
        # in the graph silently produced an empty result).
        if before > 0 and not selected:
            logger.warning(
                f"Type filters removed all {before} nodes from the selection. "
                f"include_node_types={list(keep_types) if keep_types else None}, "
                f"exclude_node_types={list(skip_types) if skip_types else None}. "
                "Returning empty graph."
            )

    if max_nodes is not None and len(selected) > max_nodes:
        # Truncate deterministically (priority: entities first, then by degree)
        ordered = sorted(
            selected,
            key=lambda n: (
                0 if nodes.get(n, {}).get("type") == "entity" else 1,
                -len(adj.get(n, set())),
            ),
        )
        selected = set(ordered[:max_nodes])

    # Collect edges within the selected set
    selected_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for nid in selected:
        for e in typed_adj.get(nid, []):
            tgt = e.get("target")
            if tgt not in selected:
                continue
            # Dedup (we store each edge twice — once from each direction)
            src = e.get("source", nid)
            key = (
                min(src, tgt),
                max(src, tgt),
                e.get("type", ""),
            )
            if key in seen_edges:
                continue
            seen_edges.add(key)
            selected_edges.append(e)

    return selected, selected_edges


# ============================================================
# GraphVisualizer
# ============================================================


class GraphVisualizer:
    """Render an EntityGraph as JSON, Mermaid, GraphML, or interactive HTML.

    Most users won't instantiate this directly — use kg.to_cytoscape(),
    kg.to_mermaid(), kg.render_html() etc. instead, which delegate here.
    """

    def __init__(self, kg: EntityGraph) -> None:
        self.kg = kg

    # -- Color resolution --

    def _resolve_color_mode(self, color_by: str | None) -> tuple[str, str | None]:
        """Choose a color mode based on user input + graph state.

        If the user requests community coloring but communities haven't
        been computed yet, auto-compute them so the legend isn't empty.
        """
        if color_by is None:
            # Prefer community if computed, else type
            if self.kg._communities:
                return "community", None
            return "type", None
        if color_by == "community":
            if not self.kg._communities and self.kg._entity_ids:
                logger.info(
                    "color_by='community' requested but communities not computed; "
                    "running compute_features() automatically."
                )
                self.kg.compute_features()
            return "community", None
        if color_by == "type":
            return "type", None
        # Treat as column name for entity-value coloring
        return "column", color_by

    # -- Cytoscape.js JSON --

    def to_cytoscape(
        self,
        *,
        color_by: str | None = None,
        max_nodes: int | None = 300,
        focus: str | None = None,
        hops: int = 2,
        include_node_types: list[str] | None = None,
        exclude_node_types: list[str] | None = None,
        include_raw_data: bool = True,
        raw_data_max_chars: int | None = 200,
    ) -> dict[str, Any]:
        """Return a Cytoscape.js-compliant graph spec.

        See https://js.cytoscape.org/ for rendering. The returned dict
        has shape::

            {
              "nodes": [{"data": {"id", "label", "type", "color", ...}}],
              "edges": [{"data": {"id", "source", "target", "label",
                                  "color", "weight", "style", ...}}],
              "metadata": {...}
            }

        Args:
            include_raw_data: If True (default), entity nodes include the
                full source record under ``data.raw_data`` so the info
                panel can show full context. Set False to drop it
                entirely — recommended when serving thousands of nodes
                where bytes matter.
            raw_data_max_chars: Truncate long string values inside
                ``raw_data`` to this many characters with an ``"…"``
                suffix. Default 200. Set to None to disable truncation
                (legacy v0.2.0 behavior — can produce 100KB+ payloads
                when records contain long ``ae_notes`` / description
                fields).
        """
        selected, edges = _select_subgraph(
            self.kg,
            max_nodes=max_nodes,
            focus=focus,
            hops=hops,
            include_node_types=include_node_types,
            exclude_node_types=exclude_node_types,
        )
        mode, column = self._resolve_color_mode(color_by)
        column_value_colors: dict[str, str] = {}

        nodes_out: list[dict[str, Any]] = []
        for nid in selected:
            node = self.kg._nodes.get(nid)
            if not node:
                continue
            color = _color_for_node(
                node,
                mode=mode,
                communities=self.kg._communities,
                column_value_colors=column_value_colors,
                column=column,
            )
            ntype = node.get("type", "")
            label = self.kg._format_node_label(nid)
            data: dict[str, Any] = {
                "id": nid,
                "label": label,
                "type": ntype,
                "color": color,
            }
            if mode == "community" and self.kg._communities:
                data["community"] = self.kg._communities.get(nid, -1)
            if ntype == "entity":
                if include_raw_data:
                    data["raw_data"] = _truncate_raw_data(
                        node.get("data", {}),
                        max_chars=raw_data_max_chars,
                    )
                data["size"] = 30  # entities bigger
            else:
                data["size"] = 16
            nodes_out.append({"data": data})

        edges_out: list[dict[str, Any]] = []
        for i, e in enumerate(edges):
            etype = e.get("type", "")
            ec, estyle = _color_for_edge(etype)
            weight = e.get("weight", 1.0)
            edges_out.append(
                {
                    "data": {
                        "id": f"e{i}",
                        "source": e.get("source"),
                        "target": e.get("target"),
                        "label": etype,
                        "weight": weight,
                        "color": ec,
                        "style": estyle,
                        "confidence": e.get("confidence", 1.0),
                    }
                }
            )

        # Build a legend from the colors used
        legend: list[dict[str, str]] = []
        if mode == "community" and self.kg._communities:
            community_counts = Counter(
                self.kg._communities[nid]
                for nid in selected
                if self.kg._nodes.get(nid, {}).get("type") == "entity"
                and nid in self.kg._communities
            )
            for cid, _ in community_counts.most_common(20):
                legend.append(
                    {
                        "label": f"Community {cid}",
                        "color": _community_color(cid),
                    }
                )
        elif mode == "type":
            seen_types = {self.kg._nodes[nid]["type"] for nid in selected if nid in self.kg._nodes}
            for t in sorted(seen_types):
                # Pick a representative color
                fake_node = {"id": "x", "type": t, "data": {}}
                c = _color_for_node(
                    fake_node,
                    mode="type",
                    communities=None,
                    column_value_colors=None,
                    column=None,
                )
                legend.append({"label": t, "color": c})
        elif mode == "column" and column:
            for value, c in column_value_colors.items():
                legend.append({"label": f"{column}={value}", "color": c})

        return {
            "nodes": nodes_out,
            "edges": edges_out,
            "metadata": {
                "color_by": color_by or mode,
                "total_nodes": len(self.kg._nodes),
                "total_edges": len(self.kg._edges),
                "displayed_nodes": len(nodes_out),
                "displayed_edges": len(edges_out),
                "snapshot_date": self.kg._snapshot_date,
                "frozen": self.kg._frozen,
                "legend": legend,
            },
        }

    # -- D3 force-graph JSON --

    def to_d3_json(
        self,
        *,
        color_by: str | None = None,
        max_nodes: int | None = 300,
        focus: str | None = None,
        hops: int = 2,
        include_raw_data: bool = True,
        raw_data_max_chars: int | None = 200,
    ) -> dict[str, Any]:
        """Return a D3.js force-graph spec ({nodes:[...], links:[...]}).

        See :meth:`to_cytoscape` for ``include_raw_data`` /
        ``raw_data_max_chars`` semantics — both are forwarded as-is.
        """
        cy = self.to_cytoscape(
            color_by=color_by,
            max_nodes=max_nodes,
            focus=focus,
            hops=hops,
            include_raw_data=include_raw_data,
            raw_data_max_chars=raw_data_max_chars,
        )
        return {
            "nodes": [n["data"] for n in cy["nodes"]],
            "links": [
                {**e["data"], "source": e["data"]["source"], "target": e["data"]["target"]}
                for e in cy["edges"]
            ],
            "metadata": cy["metadata"],
        }

    # -- Mermaid --

    def to_mermaid(
        self,
        *,
        max_nodes: int | None = 50,
        focus: str | None = None,
        hops: int = 2,
        direction: str = "LR",
        color_by: str | None = None,
    ) -> str:
        """Return a Mermaid flowchart for documentation/notebooks.

        Args:
            max_nodes: Cap on rendered nodes (default 50 — Mermaid gets
                cluttered above this).
            direction: "LR" (left-right), "TD" (top-down), "RL", "BT".
        """
        selected, edges = _select_subgraph(
            self.kg,
            max_nodes=max_nodes,
            focus=focus,
            hops=hops,
            include_node_types=None,
            exclude_node_types=None,
        )
        mode, column = self._resolve_color_mode(color_by)
        column_value_colors: dict[str, str] = {}

        # Mermaid IDs must match /^[A-Za-z][A-Za-z0-9_]*$/. We hash node
        # IDs into safe identifiers.
        def _safe(nid: str) -> str:
            h = 0
            for ch in nid:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            return f"n{h:x}"

        lines: list[str] = [f"flowchart {direction}"]
        # Node declarations
        for nid in selected:
            node = self.kg._nodes.get(nid, {})
            label = self.kg._format_node_label(nid)
            label_safe = label.replace('"', "&quot;").replace("[", "(").replace("]", ")")
            ntype = node.get("type", "")
            shape = '("__LBL__")' if ntype == "entity" else '["__LBL__"]'
            lines.append(f"  {_safe(nid)}{shape.replace('__LBL__', label_safe)}")

        # Edges
        edge_styles: list[str] = []
        for i, e in enumerate(edges):
            src = _safe(e.get("source", ""))
            tgt = _safe(e.get("target", ""))
            etype = e.get("type", "")
            if etype == "SIMILAR_TO":
                arrow = "-.->|" + etype + "|"
            else:
                arrow = "-->|" + etype + "|"
            lines.append(f"  {src} {arrow} {tgt}")

        # Color via classDef + class
        class_for: dict[str, str] = {}
        class_defs: dict[str, str] = {}
        for nid in selected:
            node = self.kg._nodes.get(nid, {})
            color = _color_for_node(
                node,
                mode=mode,
                communities=self.kg._communities,
                column_value_colors=column_value_colors,
                column=column,
            )
            cls_name = "c" + color.lstrip("#")
            class_for[nid] = cls_name
            if cls_name not in class_defs:
                class_defs[cls_name] = f"classDef {cls_name} fill:{color},stroke:#222,color:#fff"

        for cls_def in class_defs.values():
            lines.append(f"  {cls_def}")
        for nid, cls_name in class_for.items():
            lines.append(f"  class {_safe(nid)} {cls_name}")
        for s in edge_styles:
            lines.append(s)

        return "\n".join(lines)

    # -- GraphML (Gephi / yEd) --

    def to_graphml(
        self,
        path: str | None = None,
        *,
        color_by: str | None = None,
        max_nodes: int | None = None,
    ) -> str:
        """Write GraphML XML to `path` (or return as string if path is None).

        GraphML is supported by Gephi, yEd, and most graph tools.
        """
        selected, edges = _select_subgraph(
            self.kg,
            max_nodes=max_nodes,
            focus=None,
            hops=2,
            include_node_types=None,
            exclude_node_types=None,
        )
        mode, column = self._resolve_color_mode(color_by)
        column_value_colors: dict[str, str] = {}

        parts: list[str] = []
        parts.append('<?xml version="1.0" encoding="UTF-8"?>')
        parts.append(
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
            'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">'
        )
        # Attribute keys
        parts.append('<key id="label" for="node" attr.name="label" attr.type="string"/>')
        parts.append('<key id="type" for="node" attr.name="type" attr.type="string"/>')
        parts.append('<key id="color" for="node" attr.name="color" attr.type="string"/>')
        parts.append('<key id="community" for="node" attr.name="community" attr.type="int"/>')
        parts.append('<key id="etype" for="edge" attr.name="etype" attr.type="string"/>')
        parts.append('<key id="weight" for="edge" attr.name="weight" attr.type="double"/>')
        parts.append('<key id="confidence" for="edge" attr.name="confidence" attr.type="double"/>')
        parts.append('<graph id="G" edgedefault="undirected">')

        for nid in selected:
            node = self.kg._nodes.get(nid, {})
            color = _color_for_node(
                node,
                mode=mode,
                communities=self.kg._communities,
                column_value_colors=column_value_colors,
                column=column,
            )
            label = self.kg._format_node_label(nid)
            parts.append(f'<node id="{_html.escape(nid)}">')
            parts.append(f'  <data key="label">{_html.escape(label)}</data>')
            parts.append(f'  <data key="type">{_html.escape(node.get("type", ""))}</data>')
            parts.append(f'  <data key="color">{color}</data>')
            if self.kg._communities and nid in self.kg._communities:
                parts.append(f'  <data key="community">{self.kg._communities[nid]}</data>')
            parts.append("</node>")

        for i, e in enumerate(edges):
            src = e.get("source", "")
            tgt = e.get("target", "")
            parts.append(
                f'<edge id="e{i}" source="{_html.escape(src)}" target="{_html.escape(tgt)}">'
            )
            parts.append(f'  <data key="etype">{_html.escape(e.get("type", ""))}</data>')
            parts.append(f'  <data key="weight">{e.get("weight", 1.0)}</data>')
            parts.append(f'  <data key="confidence">{e.get("confidence", 1.0)}</data>')
            parts.append("</edge>")

        parts.append("</graph>")
        parts.append("</graphml>")
        xml = "\n".join(parts)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(xml)
            logger.info(f"GraphML written to {path}")
        return xml

    # -- Interactive HTML --

    def render_html(
        self,
        path: str | None = None,
        *,
        title: str = "EntityGraph",
        color_by: str | None = None,
        max_nodes: int | None = 300,
        focus: str | None = None,
        hops: int = 2,
        include_node_types: list[str] | None = None,
        exclude_node_types: list[str] | None = None,
        layout: str = "cose",
        include_raw_data: bool = True,
        raw_data_max_chars: int | None = 200,
    ) -> str:
        """Render the graph as a self-contained interactive HTML file.

        The HTML loads Cytoscape.js from a CDN. Open it in any browser
        to pan, zoom, drag nodes, and inspect details.

        Args:
            path: Output file. If None, returns HTML as a string.
            title: Page title.
            color_by: "community" (default), "type", or column name.
            max_nodes: Cap on rendered nodes.
            focus: Center the view on this entity (k-hop neighborhood).
            hops: Hops for focus mode.
            layout: Cytoscape layout. "cose" (force-directed, default),
                "concentric" (community-grouped), "breadthfirst", "grid",
                "circle", or "klay".
            include_raw_data: Include source-record data in the click-to-
                inspect panel. Default True. Set False when serving
                thousands of nodes to avoid bloating the HTML payload.
            raw_data_max_chars: Truncate long string values in raw_data
                to this many characters. Default 200. Set None for the
                legacy v0.2.0 behavior of dumping full records.
        """
        spec = self.to_cytoscape(
            color_by=color_by,
            max_nodes=max_nodes,
            focus=focus,
            hops=hops,
            include_node_types=include_node_types,
            exclude_node_types=exclude_node_types,
            include_raw_data=include_raw_data,
            raw_data_max_chars=raw_data_max_chars,
        )
        html_str = _render_html_template(spec, title=title, layout=layout)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html_str)
            logger.info(f"Graph rendered to {path}")
        return html_str


# ============================================================
# HTML template
# ============================================================


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.1/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape-cose-bilkent/4.1.0/cytoscape-cose-bilkent.min.js"></script>
<style>
  :root {
    --bg: #0a0f1e;
    --panel: #161e2e;
    --panel-2: #1e293b;
    --panel-border: #334155;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --muted-2: #64748b;
    --accent: #3b82f6;
    --accent-2: #60a5fa;
    --highlight: #fbbf24;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg); color: var(--text);
    font: 13px/1.45 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
  }
  #app { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  #sidebar {
    background: var(--panel); border-right: 1px solid var(--panel-border);
    padding: 14px 16px 24px; overflow-y: auto;
  }
  #sidebar h1 {
    font-size: 15px; margin: 0 0 4px; letter-spacing: 0.02em;
    display: flex; align-items: center; justify-content: space-between;
  }
  #sidebar h1 .badge {
    background: var(--panel-2); color: var(--muted);
    font-size: 11px; font-weight: 500; padding: 2px 8px;
    border-radius: 10px; letter-spacing: 0;
  }
  #sidebar .help {
    color: var(--muted-2); font-size: 11px; margin: 0 0 14px;
  }
  details { margin: 14px 0 0; border-top: 1px solid var(--panel-border); }
  details > summary {
    cursor: pointer; padding: 10px 0 8px; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
    list-style: none; display: flex; justify-content: space-between;
    align-items: center; user-select: none;
  }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::after { content: "\\002B"; color: var(--muted); font-size: 14px; }
  details[open] > summary::after { content: "\\2212"; }
  details > summary:hover { color: var(--text); }
  details > .section-body { padding: 4px 0 8px; }
  #search {
    width: 100%; padding: 8px 10px; background: var(--bg);
    border: 1px solid var(--panel-border); border-radius: 6px;
    color: var(--text); font-size: 13px; margin-bottom: 8px;
  }
  #search:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15);
  }
  #stats div {
    font-size: 12px; color: var(--muted); display: flex;
    justify-content: space-between; margin-bottom: 5px;
  }
  #stats div span:last-child { color: var(--text); font-weight: 600; tabular-nums: 1; }
  #stats div.live span:last-child { color: var(--accent-2); }
  .filter-controls {
    display: flex; gap: 4px; margin-bottom: 10px;
  }
  .filter-controls button {
    flex: 1; background: var(--bg); border: 1px solid var(--panel-border);
    color: var(--muted); padding: 5px 6px; border-radius: 4px;
    cursor: pointer; font-size: 11px;
  }
  .filter-controls button:hover { border-color: var(--accent); color: var(--text); }
  .filter-list { display: flex; flex-direction: column; gap: 3px; }
  .filter-item {
    display: flex; align-items: center; gap: 8px; cursor: pointer;
    padding: 4px 6px; border-radius: 4px; font-size: 12px;
    user-select: none;
  }
  .filter-item:hover { background: var(--panel-2); }
  .filter-item input { margin: 0; cursor: pointer; }
  .filter-item .swatch {
    width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
  }
  .filter-item .swatch.dashed {
    background-image: linear-gradient(90deg, var(--swatch-color) 50%, transparent 50%);
    background-size: 4px 100%;
  }
  .filter-item .name {
    flex: 1; color: var(--text);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .filter-item .count { color: var(--muted); font-size: 11px; tabular-nums: 1; }
  .filter-item.disabled .name,
  .filter-item.disabled .count {
    color: var(--muted-2); text-decoration: line-through;
  }
  .layout-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
  .layout-buttons button, .display-toggles button {
    background: var(--bg); border: 1px solid var(--panel-border);
    color: var(--text); padding: 6px 8px; border-radius: 4px;
    cursor: pointer; font-size: 11px; transition: border-color 0.1s;
  }
  .layout-buttons button:hover, .display-toggles button:hover { border-color: var(--accent); }
  .layout-buttons button.active, .display-toggles button.active {
    background: var(--accent); border-color: var(--accent); color: #fff;
  }
  .display-toggles { display: flex; flex-direction: column; gap: 4px; }
  .display-toggles button { text-align: left; }
  #selection-section { display: none; }
  #selection-section.visible { display: block; }
  #selection-section .selection-actions {
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px;
  }
  #selection-section .selection-actions button {
    background: var(--bg); border: 1px solid var(--panel-border);
    color: var(--text); padding: 6px; border-radius: 4px;
    cursor: pointer; font-size: 11px;
  }
  #selection-section .selection-actions button.primary {
    background: var(--accent); border-color: var(--accent); color: #fff;
  }
  #selection-section .selection-actions button:hover { border-color: var(--accent); }
  #cy { width: 100%; height: 100%; background: #020617; }
  #info {
    position: absolute; right: 16px; top: 16px; max-width: 320px;
    background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 8px; padding: 14px; display: none;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  }
  #info h3 { margin: 0 0 10px; font-size: 14px; }
  #info .row {
    font-size: 12px; margin-bottom: 5px; display: flex;
    justify-content: space-between; gap: 12px;
  }
  #info .row span:first-child { color: var(--muted); }
  #info pre {
    font-size: 11px; background: var(--bg); padding: 8px;
    border-radius: 4px; overflow: auto; max-height: 200px;
    margin: 8px 0 0; line-height: 1.4;
  }
  #info .tag {
    display: inline-block; background: var(--bg); padding: 2px 8px;
    border-radius: 10px; font-size: 11px; color: var(--muted);
    margin-right: 4px;
  }
  #info .close {
    float: right; background: none; border: none; color: var(--muted);
    cursor: pointer; padding: 0; font-size: 16px; line-height: 1;
  }
  #info .close:hover { color: var(--text); }
  #toolbar {
    position: absolute; bottom: 16px; right: 16px; display: flex; gap: 6px;
    background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 8px; padding: 4px;
  }
  #toolbar button {
    background: transparent; border: none; color: var(--muted);
    padding: 6px 10px; border-radius: 4px; cursor: pointer; font-size: 12px;
  }
  #toolbar button:hover { background: var(--panel-2); color: var(--text); }
  #status-bar {
    position: absolute; bottom: 16px; left: 336px;
    background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 8px; padding: 6px 12px; font-size: 11px;
    color: var(--muted); display: none; max-width: 480px;
  }
  #status-bar.visible { display: block; }
  #status-bar .key {
    font-family: ui-monospace, SF Mono, monospace;
    background: var(--bg); padding: 1px 5px; border-radius: 3px;
    margin: 0 2px; color: var(--text); font-size: 10px;
  }
  kbd {
    font-family: ui-monospace, SF Mono, monospace;
    background: var(--bg); padding: 1px 5px; border-radius: 3px;
    color: var(--text); font-size: 10px; border: 1px solid var(--panel-border);
  }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <h1>__TITLE__ <span class="badge" id="title-badge"></span></h1>
    <p class="help">
      <kbd>/</kbd> search · <kbd>Esc</kbd> reset · <kbd>e</kbd> edge labels · <kbd>f</kbd> fit
    </p>
    <input id="search" type="text" placeholder="Search nodes..." autocomplete="off" />

    <details open>
      <summary>Stats</summary>
      <div class="section-body"><div id="stats"></div></div>
    </details>

    <details open id="edge-filter-section">
      <summary>Edge types <span class="counter" data-counter="edge"></span></summary>
      <div class="section-body">
        <div class="filter-controls">
          <button data-action="all" data-target="edge">All</button>
          <button data-action="none" data-target="edge">None</button>
        </div>
        <div class="filter-list" id="edge-filters"></div>
      </div>
    </details>

    <details open id="node-filter-section">
      <summary>Node types <span class="counter" data-counter="node"></span></summary>
      <div class="section-body">
        <div class="filter-controls">
          <button data-action="all" data-target="node">All</button>
          <button data-action="none" data-target="node">None</button>
        </div>
        <div class="filter-list" id="node-filters"></div>
      </div>
    </details>

    <details id="community-filter-section" style="display:none">
      <summary>Communities</summary>
      <div class="section-body">
        <div class="filter-controls">
          <button data-action="all" data-target="community">All</button>
          <button data-action="none" data-target="community">None</button>
        </div>
        <div class="filter-list" id="community-filters"></div>
      </div>
    </details>

    <details open>
      <summary>Display</summary>
      <div class="section-body">
        <div class="display-toggles">
          <button id="toggle-edge-labels">Show edge labels (e)</button>
          <button id="toggle-hover-highlight" class="active">Highlight on hover</button>
          <button id="toggle-bold-edges" class="active">Bold edges</button>
          <button id="toggle-arrows" class="active">Show arrows</button>
        </div>
      </div>
    </details>

    <details>
      <summary>Layout</summary>
      <div class="section-body">
        <div class="layout-buttons">
          <button data-layout="cose-bilkent" class="active">Force</button>
          <button data-layout="concentric">Concentric</button>
          <button data-layout="breadthfirst">Tree</button>
          <button data-layout="grid">Grid</button>
          <button data-layout="circle">Circle</button>
          <button data-layout="random">Random</button>
        </div>
      </div>
    </details>

    <details id="selection-section">
      <summary>Selection</summary>
      <div class="section-body">
        <div id="selection-info"></div>
        <div class="selection-actions">
          <button class="primary" id="action-isolate">Isolate</button>
          <button id="action-clear">Clear</button>
        </div>
      </div>
    </details>

    <details>
      <summary>Legend</summary>
      <div class="section-body"><div id="legend"></div></div>
    </details>
  </aside>
  <div id="cy"></div>
</div>
<div id="info">
  <button class="close" id="info-close">&times;</button>
  <div id="info-body"></div>
</div>
<div id="status-bar"></div>
<div id="toolbar">
  <button data-toolbar="fit" title="Fit view (f)">Fit</button>
  <button data-toolbar="reset" title="Reset filters &amp; selection (Esc)">Reset</button>
</div>
<script>
const SPEC = __SPEC__;
const elements = SPEC.nodes.concat(SPEC.edges);

// ---- Cytoscape construction ----------------------------------------
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: elements,
  wheelSensitivity: 0.2,
  style: [
    {
      selector: 'node',
      style: {
        'background-color': 'data(color)',
        'label': 'data(label)',
        'color': '#fff',
        'font-size': 11,
        'text-outline-color': '#000',
        'text-outline-width': 2,
        'text-valign': 'center',
        'text-halign': 'center',
        'width': 'data(size)',
        'height': 'data(size)',
        'border-color': '#0f172a',
        'border-width': 2,
        'transition-property': 'opacity, background-color, border-color, border-width',
        'transition-duration': '0.12s',
      }
    },
    {
      selector: 'node[type="entity"]',
      style: { 'shape': 'ellipse', 'border-color': '#fff', 'border-width': 2 }
    },
    { selector: 'node[type="concept"]', style: { 'shape': 'round-rectangle' } },
    { selector: 'node[type="entity"][?community]',
      style: { 'border-color': '#fff', 'border-width': 2 } },
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.9,
        'width': 'mapData(weight, 0, 5, 2.2, 5)',
        'opacity': 0.78,
        'label': 'data(label)',
        'font-size': 9,
        'color': '#cbd5e1',
        'text-rotation': 'autorotate',
        'text-opacity': 0,
        'text-background-color': '#020617',
        'text-background-opacity': 0.85,
        'text-background-padding': 2,
        'text-background-shape': 'roundrectangle',
        'transition-property': 'opacity, line-color, target-arrow-color, width, text-opacity',
        'transition-duration': '0.12s',
      }
    },
    { selector: 'edge[style="dashed"]', style: { 'line-style': 'dashed' } },

    // Display modifiers (toggleable):
    { selector: '.thin-edges', style: { 'width': 'mapData(weight, 0, 5, 1, 3)', 'opacity': 0.5 } },
    { selector: '.show-edge-labels', style: { 'text-opacity': 1 } },
    { selector: '.no-arrows', style: { 'target-arrow-shape': 'none' } },

    // Filtering / hidden state:
    { selector: '.hidden', style: { 'display': 'none' } },

    // Hover/selection focus mode:
    { selector: '.faded', style: { 'opacity': 0.08, 'text-opacity': 0 } },
    { selector: '.highlighted', style: { 'opacity': 1 } },
    { selector: 'node.highlighted',
      style: { 'border-color': 'var(--highlight)', 'border-width': 4 } },
    { selector: 'edge.highlighted',
      style: { 'width': 'mapData(weight, 0, 5, 3, 6.5)', 'text-opacity': 1 } },

    // Search match (gold glow):
    { selector: 'node.search-match',
      style: { 'border-color': '#fbbf24', 'border-width': 5, 'opacity': 1 } },

    // Selected node:
    {
      selector: ':selected',
      style: { 'border-width': 5, 'border-color': '#3b82f6', 'opacity': 1 }
    },
    {
      selector: 'edge:selected',
      style: { 'line-color': '#3b82f6', 'target-arrow-color': '#3b82f6',
               'opacity': 1, 'text-opacity': 1, 'width': 4 }
    },
  ],
  layout: { name: '__LAYOUT__', padding: 30, animate: false, nodeDimensionsIncludeLabels: true }
});

// ---- Data helpers --------------------------------------------------
const allEdgeTypes = {};
const allNodeTypes = {};
const allCommunities = {};
SPEC.edges.forEach(e => {
  const t = e.data.label || 'unknown';
  if (!allEdgeTypes[t]) allEdgeTypes[t] = { count: 0, color: e.data.color || '#94a3b8',
                                              style: e.data.style || 'solid' };
  allEdgeTypes[t].count++;
});
SPEC.nodes.forEach(n => {
  const t = n.data.type || 'unknown';
  if (!allNodeTypes[t]) allNodeTypes[t] = { count: 0, color: n.data.color || '#94a3b8' };
  allNodeTypes[t].count++;
  if (n.data.community !== undefined) {
    const cid = n.data.community;
    if (!allCommunities[cid]) allCommunities[cid] = { count: 0, color: n.data.color };
    allCommunities[cid].count++;
  }
});

const enabledEdgeTypes = new Set(Object.keys(allEdgeTypes));
const enabledNodeTypes = new Set(Object.keys(allNodeTypes));
const enabledCommunities = new Set(Object.keys(allCommunities));

// ---- Filter UI builder --------------------------------------------
function buildFilterList(containerId, items, enabled, onChange, sortByCount) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  const entries = Object.entries(items);
  if (sortByCount) entries.sort((a, b) => b[1].count - a[1].count);
  else entries.sort((a, b) => a[0].localeCompare(b[0]));
  entries.forEach(([key, info]) => {
    const label = document.createElement('label');
    label.className = 'filter-item';
    label.dataset.key = key;
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = enabled.has(key);
    cb.addEventListener('change', () => {
      if (cb.checked) enabled.add(key); else enabled.delete(key);
      label.classList.toggle('disabled', !cb.checked);
      onChange();
    });
    const swatch = document.createElement('span');
    swatch.className = 'swatch' + (info.style === 'dashed' ? ' dashed' : '');
    swatch.style.setProperty('--swatch-color', info.color);
    swatch.style.background = info.color;
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = key;
    const count = document.createElement('span');
    count.className = 'count';
    count.textContent = info.count;
    label.appendChild(cb); label.appendChild(swatch);
    label.appendChild(name); label.appendChild(count);
    if (!enabled.has(key)) label.classList.add('disabled');
    container.appendChild(label);
  });
}

function applyFilters() {
  let visibleNodes = 0, visibleEdges = 0;
  cy.batch(() => {
    cy.nodes().forEach(n => {
      const t = n.data('type');
      const c = n.data('community');
      const visible = enabledNodeTypes.has(t)
        && (c === undefined || enabledCommunities.has(String(c)));
      n.toggleClass('hidden', !visible);
      if (visible) visibleNodes++;
    });
    cy.edges().forEach(e => {
      const t = e.data('label');
      const visible = enabledEdgeTypes.has(t)
        && !e.source().hasClass('hidden') && !e.target().hasClass('hidden');
      e.toggleClass('hidden', !visible);
      if (visible) visibleEdges++;
    });
  });
  // Update live stats
  document.querySelector('#stats div.live[data-stat="visible-nodes"] span:last-child')
    .textContent = `${visibleNodes} / ${SPEC.metadata.displayed_nodes}`;
  document.querySelector('#stats div.live[data-stat="visible-edges"] span:last-child')
    .textContent = `${visibleEdges} / ${SPEC.metadata.displayed_edges}`;
  if (currentSelection) updateSelectionInfo(currentSelection);
}

// ---- Build all 3 filter lists --------------------------------------
buildFilterList('edge-filters', allEdgeTypes, enabledEdgeTypes, applyFilters, true);
buildFilterList('node-filters', allNodeTypes, enabledNodeTypes, applyFilters, true);
if (Object.keys(allCommunities).length > 1) {
  document.getElementById('community-filter-section').style.display = '';
  buildFilterList('community-filters', allCommunities, enabledCommunities, applyFilters, true);
}

// All / None buttons
document.querySelectorAll('.filter-controls button').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.target;
    const allItems =
      target === 'edge' ? allEdgeTypes :
      target === 'node' ? allNodeTypes : allCommunities;
    const set =
      target === 'edge' ? enabledEdgeTypes :
      target === 'node' ? enabledNodeTypes : enabledCommunities;
    set.clear();
    if (btn.dataset.action === 'all') Object.keys(allItems).forEach(k => set.add(k));
    const containerId =
      target === 'edge' ? 'edge-filters' :
      target === 'node' ? 'node-filters' : 'community-filters';
    document.querySelectorAll(`#${containerId} input`).forEach(cb => {
      cb.checked = set.has(cb.parentElement.dataset.key);
      cb.parentElement.classList.toggle('disabled', !cb.checked);
    });
    applyFilters();
  });
});

// ---- Title badge + stats ------------------------------------------
const stats = SPEC.metadata;
document.getElementById('title-badge').textContent =
  `${stats.displayed_nodes} nodes · ${stats.displayed_edges} edges`;

const statsEl = document.getElementById('stats');
const statRows = [
  ['Visible nodes', `${stats.displayed_nodes} / ${stats.displayed_nodes}`, true, 'visible-nodes'],
  ['Visible edges', `${stats.displayed_edges} / ${stats.displayed_edges}`, true, 'visible-edges'],
  ['Total in graph', `${stats.total_nodes} / ${stats.total_edges}`, false],
  ['Color by', stats.color_by, false],
  ['Snapshot', stats.snapshot_date || 'live', false],
  ['Frozen', stats.frozen ? 'yes' : 'no', false],
];
statRows.forEach(([label, value, isLive, statKey]) => {
  const div = document.createElement('div');
  if (isLive) { div.className = 'live'; div.dataset.stat = statKey; }
  div.innerHTML = `<span>${label}</span><span>${value}</span>`;
  statsEl.appendChild(div);
});

// ---- Legend --------------------------------------------------------
const legendEl = document.getElementById('legend');
(SPEC.metadata.legend || []).forEach(item => {
  const div = document.createElement('div');
  div.className = 'filter-item';
  div.innerHTML =
    `<span class="swatch" style="background:${item.color}"></span>` +
    `<span class="name">${escapeHtml(item.label)}</span>`;
  legendEl.appendChild(div);
});

// ---- Layout switching ---------------------------------------------
document.querySelectorAll('[data-layout]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-layout]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    cy.layout({
      name: btn.dataset.layout, padding: 30,
      animate: true, animationDuration: 500,
      nodeDimensionsIncludeLabels: true,
    }).run();
  });
});

// ---- Display toggles ----------------------------------------------
function toggleClass(btnId, cls, target) {
  const btn = document.getElementById(btnId);
  btn.addEventListener('click', () => {
    btn.classList.toggle('active');
    const on = btn.classList.contains('active');
    cy.elements()[on ? 'addClass' : 'removeClass'](cls);
  });
}
// Edge labels: opt-in
const edgeLabelBtn = document.getElementById('toggle-edge-labels');
edgeLabelBtn.addEventListener('click', () => {
  edgeLabelBtn.classList.toggle('active');
  cy.elements().toggleClass('show-edge-labels', edgeLabelBtn.classList.contains('active'));
});
// Hover highlight (a behavior toggle, not a class). Default ON.
let hoverHighlightOn = true;
document.getElementById('toggle-hover-highlight').addEventListener('click', (e) => {
  hoverHighlightOn = !hoverHighlightOn;
  e.currentTarget.classList.toggle('active', hoverHighlightOn);
  if (!hoverHighlightOn) clearFocusMode();
});
// Bold edges: default ON. When OFF, apply .thin-edges class.
const boldBtn = document.getElementById('toggle-bold-edges');
boldBtn.addEventListener('click', () => {
  boldBtn.classList.toggle('active');
  const on = boldBtn.classList.contains('active');
  cy.elements()[on ? 'removeClass' : 'addClass']('thin-edges');
});
// Arrows: default ON.
const arrowBtn = document.getElementById('toggle-arrows');
arrowBtn.addEventListener('click', () => {
  arrowBtn.classList.toggle('active');
  const on = arrowBtn.classList.contains('active');
  cy.elements()[on ? 'removeClass' : 'addClass']('no-arrows');
});

// ---- Focus mode (hover + isolation) -------------------------------
function focusOn(node) {
  const others = cy.elements().not(node).not(node.neighborhood());
  others.addClass('faded');
  node.neighborhood().removeClass('faded');
  node.addClass('highlighted');
  node.connectedEdges().addClass('highlighted');
}
function clearFocusMode() {
  cy.elements().removeClass('faded').removeClass('highlighted');
}

cy.on('mouseover', 'node', (evt) => {
  if (!hoverHighlightOn || isolatedSubgraph) return;
  focusOn(evt.target);
});
cy.on('mouseout', 'node', () => {
  if (!hoverHighlightOn || isolatedSubgraph) return;
  if (!currentSelection) clearFocusMode();
  else focusOn(currentSelection);
});

// ---- Selection / isolation ----------------------------------------
let currentSelection = null;
let isolatedSubgraph = null;
const infoEl = document.getElementById('info');
const infoBody = document.getElementById('info-body');
const selectionSection = document.getElementById('selection-section');

function updateSelectionInfo(node) {
  const data = node.data();
  let html = `<h3>${escapeHtml(data.label)}</h3>`;
  html += `<div class="row"><span>Type</span><span>${escapeHtml(data.type)}</span></div>`;
  if (data.community !== undefined)
    html += `<div class="row"><span>Community</span><span>${data.community}</span></div>`;
  html += `<div class="row"><span>Connections</span><span>${node.degree()}</span></div>`;
  // Group connections by edge type
  const byType = {};
  node.connectedEdges().forEach(e => {
    const t = e.data('label');
    byType[t] = (byType[t] || 0) + 1;
  });
  const tags = Object.entries(byType)
    .sort((a, b) => b[1] - a[1])
    .map(([t, c]) => `<span class="tag">${escapeHtml(t)} (${c})</span>`)
    .join(' ');
  if (tags) html += `<div style="margin: 8px 0">${tags}</div>`;
  if (data.raw_data && Object.keys(data.raw_data).length > 0) {
    html += `<pre>${escapeHtml(JSON.stringify(data.raw_data, null, 2))}</pre>`;
  }
  infoBody.innerHTML = html;
  infoEl.style.display = 'block';

  document.getElementById('selection-info').innerHTML =
    `<strong>${escapeHtml(data.label)}</strong><br/>` +
    `<span style="color:var(--muted);font-size:11px">` +
    `${node.degree()} connections · ${data.type}</span>`;
  selectionSection.classList.add('visible');
  selectionSection.open = true;
}

cy.on('tap', 'node', (evt) => {
  currentSelection = evt.target;
  if (hoverHighlightOn && !isolatedSubgraph) focusOn(currentSelection);
  updateSelectionInfo(currentSelection);
});

cy.on('tap', (evt) => {
  if (evt.target === cy && !isolatedSubgraph) {
    currentSelection = null;
    infoEl.style.display = 'none';
    selectionSection.classList.remove('visible');
    clearFocusMode();
  }
});

// Isolate: hide everything except the selected node + its k-hop neighborhood.
document.getElementById('action-isolate').addEventListener('click', () => {
  if (!currentSelection) return;
  const keep = currentSelection.closedNeighborhood();
  cy.elements().not(keep).addClass('hidden');
  isolatedSubgraph = currentSelection;
  setStatus(
    `Isolated <strong>${escapeHtml(currentSelection.data('label'))}</strong>. ` +
    `Press <kbd>Esc</kbd> or click <strong>Reset</strong> to restore.`
  );
});
document.getElementById('action-clear').addEventListener('click', () => clearAll());
document.getElementById('info-close').addEventListener('click', () => {
  infoEl.style.display = 'none'; currentSelection = null;
  selectionSection.classList.remove('visible'); clearFocusMode();
});

// ---- Search --------------------------------------------------------
const searchEl = document.getElementById('search');
searchEl.addEventListener('input', () => {
  const q = searchEl.value.toLowerCase().trim();
  cy.elements().removeClass('faded').removeClass('search-match').removeClass('highlighted');
  if (!q) { setStatus(''); return; }
  const matches = cy.nodes().filter(n =>
    !n.hasClass('hidden') && n.data('label').toLowerCase().includes(q)
  );
  if (matches.length === 0) {
    setStatus(`No nodes matching <strong>${escapeHtml(q)}</strong>`);
    return;
  }
  cy.elements().addClass('faded');
  matches.removeClass('faded').addClass('search-match');
  matches.connectedEdges().removeClass('faded').addClass('highlighted');
  matches.neighborhood().removeClass('faded');
  setStatus(
    `${matches.length} match${matches.length === 1 ? '' : 'es'} for ` +
    `<strong>${escapeHtml(q)}</strong>. Press <kbd>Esc</kbd> to clear.`
  );
});

// ---- Toolbar -------------------------------------------------------
document.querySelectorAll('[data-toolbar]').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.dataset.toolbar === 'fit') cy.fit(undefined, 30);
    else if (btn.dataset.toolbar === 'reset') clearAll();
  });
});

// ---- Status bar ----------------------------------------------------
const statusBar = document.getElementById('status-bar');
function setStatus(msg) {
  if (!msg) { statusBar.classList.remove('visible'); statusBar.innerHTML = ''; return; }
  statusBar.innerHTML = msg;
  statusBar.classList.add('visible');
}

// ---- Reset ---------------------------------------------------------
function clearAll() {
  searchEl.value = '';
  cy.elements().removeClass('hidden').removeClass('faded')
    .removeClass('search-match').removeClass('highlighted');
  isolatedSubgraph = null;
  currentSelection = null;
  infoEl.style.display = 'none';
  selectionSection.classList.remove('visible');
  // Re-enable all filters
  Object.keys(allEdgeTypes).forEach(k => enabledEdgeTypes.add(k));
  Object.keys(allNodeTypes).forEach(k => enabledNodeTypes.add(k));
  Object.keys(allCommunities).forEach(k => enabledCommunities.add(k));
  document.querySelectorAll('.filter-list input[type="checkbox"]').forEach(cb => {
    cb.checked = true; cb.parentElement.classList.remove('disabled');
  });
  applyFilters();
  setStatus('');
}

// ---- Keyboard shortcuts -------------------------------------------
document.addEventListener('keydown', (e) => {
  if (e.target === searchEl) {
    if (e.key === 'Escape') { searchEl.value = ''; searchEl.blur();
      cy.elements().removeClass('faded').removeClass('search-match'); setStatus(''); }
    return;
  }
  if (e.key === '/') { e.preventDefault(); searchEl.focus(); searchEl.select(); }
  else if (e.key === 'Escape') clearAll();
  else if (e.key === 'f') cy.fit(undefined, 30);
  else if (e.key === 'e') edgeLabelBtn.click();
});

// Prime the live counts.
applyFilters();

function escapeHtml(s) {
  if (s == null) return '';
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  return String(s).replace(/[&<>"']/g, c => map[c]);
}
</script>
</body>
</html>"""


def _render_html_template(spec: dict[str, Any], *, title: str, layout: str) -> str:
    safe_title = _html.escape(title)
    layout_safe = layout.replace('"', "")
    spec_json = _json.dumps(spec, default=str)
    return (
        _HTML_TEMPLATE.replace("__TITLE__", safe_title)
        .replace("__LAYOUT__", layout_safe)
        .replace("__SPEC__", spec_json)
    )
