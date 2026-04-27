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
    else:
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
        filtered = set()
        for nid in selected:
            ntype = nodes.get(nid, {}).get("type", "")
            if keep_types and ntype not in keep_types:
                continue
            if ntype in skip_types:
                continue
            filtered.add(nid)
        selected = filtered

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
                data["raw_data"] = node.get("data", {})
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
    ) -> dict[str, Any]:
        """Return a D3.js force-graph spec ({nodes:[...], links:[...]})."""
        cy = self.to_cytoscape(color_by=color_by, max_nodes=max_nodes, focus=focus, hops=hops)
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
        """
        spec = self.to_cytoscape(
            color_by=color_by,
            max_nodes=max_nodes,
            focus=focus,
            hops=hops,
            include_node_types=include_node_types,
            exclude_node_types=exclude_node_types,
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
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-border: #334155;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #3b82f6;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg); color: var(--text);
    font: 14px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
  }
  #app { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }
  #sidebar {
    background: var(--panel); border-right: 1px solid var(--panel-border);
    padding: 16px; overflow-y: auto;
  }
  #sidebar h1 { font-size: 16px; margin: 0 0 16px; letter-spacing: 0.02em; }
  #sidebar h2 {
    font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--muted); margin: 18px 0 8px;
  }
  #stats div {
    font-size: 12px; color: var(--muted); display: flex;
    justify-content: space-between; margin-bottom: 4px;
  }
  #stats div span:last-child { color: var(--text); font-weight: 600; }
  .legend-item { display: flex; align-items: center; margin: 4px 0; font-size: 12px; }
  .legend-swatch {
    width: 12px; height: 12px; border-radius: 2px;
    margin-right: 8px; flex-shrink: 0;
  }
  #search {
    width: 100%; padding: 8px; background: var(--bg);
    border: 1px solid var(--panel-border); border-radius: 4px;
    color: var(--text); font-size: 13px; margin-bottom: 8px;
  }
  #search:focus { outline: none; border-color: var(--accent); }
  .layout-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
  .layout-buttons button {
    background: var(--bg); border: 1px solid var(--panel-border);
    color: var(--text); padding: 6px; border-radius: 4px;
    cursor: pointer; font-size: 11px;
  }
  .layout-buttons button:hover { border-color: var(--accent); }
  #cy { width: 100%; height: 100%; background: #020617; }
  #info {
    position: absolute; right: 16px; top: 16px; max-width: 300px;
    background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 6px; padding: 12px; display: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  #info h3 { margin: 0 0 8px; font-size: 14px; }
  #info .row {
    font-size: 12px; margin-bottom: 4px; display: flex;
    justify-content: space-between; gap: 12px;
  }
  #info .row span:first-child { color: var(--muted); }
  #info pre {
    font-size: 11px; background: var(--bg); padding: 6px;
    border-radius: 4px; overflow-x: auto; max-height: 160px;
    margin: 6px 0 0;
  }
  .tag {
    display: inline-block; background: var(--bg); padding: 2px 6px;
    border-radius: 3px; font-size: 11px; color: var(--muted);
    margin-right: 4px;
  }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <h1>__TITLE__</h1>
    <input id="search" type="text" placeholder="Search nodes..." />
    <h2>Stats</h2>
    <div id="stats"></div>
    <h2>Layout</h2>
    <div class="layout-buttons">
      <button data-layout="cose-bilkent">Force</button>
      <button data-layout="concentric">Concentric</button>
      <button data-layout="breadthfirst">Tree</button>
      <button data-layout="grid">Grid</button>
      <button data-layout="circle">Circle</button>
      <button data-layout="random">Random</button>
    </div>
    <h2>Legend</h2>
    <div id="legend"></div>
  </aside>
  <div id="cy"></div>
</div>
<div id="info"></div>
<script>
const SPEC = __SPEC__;
const elements = SPEC.nodes.concat(SPEC.edges);

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: elements,
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
      }
    },
    {
      selector: 'node[type="entity"]',
      style: {
        'shape': 'ellipse',
        'border-color': '#fff',
        'border-width': 1.5,
      }
    },
    {
      selector: 'node[type="concept"]',
      style: { 'shape': 'round-rectangle' }
    },
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.7,
        'width': 'mapData(weight, 0, 5, 1, 4)',
        'opacity': 0.6,
        'label': 'data(label)',
        'font-size': 9,
        'color': '#94a3b8',
        'text-rotation': 'autorotate',
        'text-opacity': 0,
      }
    },
    {
      selector: 'edge[style="dashed"]',
      style: { 'line-style': 'dashed' }
    },
    {
      selector: ':selected',
      style: {
        'border-width': 4,
        'border-color': '#3b82f6',
        'opacity': 1,
      }
    },
    {
      selector: 'edge:selected',
      style: {
        'line-color': '#3b82f6',
        'target-arrow-color': '#3b82f6',
        'opacity': 1,
        'text-opacity': 1,
        'width': 3,
      }
    },
    { selector: '.faded', style: { 'opacity': 0.1 } },
    { selector: '.highlighted', style: { 'opacity': 1 } },
  ],
  layout: { name: '__LAYOUT__', padding: 30, animate: false, nodeDimensionsIncludeLabels: true }
});

// Legend
const legendEl = document.getElementById('legend');
(SPEC.metadata.legend || []).forEach(item => {
  const div = document.createElement('div');
  div.className = 'legend-item';
  div.innerHTML =
    `<span class="legend-swatch" style="background:${item.color}"></span>` +
    `${item.label}`;
  legendEl.appendChild(div);
});

// Stats
const statsEl = document.getElementById('stats');
const stats = SPEC.metadata;
const statRows = [
  ['Total nodes', stats.total_nodes],
  ['Total edges', stats.total_edges],
  ['Displayed nodes', stats.displayed_nodes],
  ['Displayed edges', stats.displayed_edges],
  ['Color by', stats.color_by],
  ['Snapshot', stats.snapshot_date || 'live'],
  ['Frozen', stats.frozen ? 'yes' : 'no'],
];
statRows.forEach(([label, value]) => {
  const div = document.createElement('div');
  div.innerHTML = `<span>${label}</span><span>${value}</span>`;
  statsEl.appendChild(div);
});

// Layout switching
document.querySelectorAll('[data-layout]').forEach(btn => {
  btn.addEventListener('click', () => {
    cy.layout({
      name: btn.dataset.layout,
      padding: 30,
      animate: true,
      animationDuration: 500,
    }).run();
  });
});

// Search
const searchEl = document.getElementById('search');
searchEl.addEventListener('input', () => {
  const q = searchEl.value.toLowerCase().trim();
  if (!q) {
    cy.elements().removeClass('faded').removeClass('highlighted');
    return;
  }
  const matches = cy.nodes().filter(n => n.data('label').toLowerCase().includes(q));
  cy.elements().addClass('faded');
  matches.removeClass('faded').addClass('highlighted');
  matches.connectedEdges().removeClass('faded');
  matches.neighborhood().removeClass('faded');
});

// Info panel on node click
const infoEl = document.getElementById('info');
cy.on('tap', 'node', (evt) => {
  const node = evt.target;
  const data = node.data();
  let html = `<h3>${escapeHtml(data.label)}</h3>`;
  html += `<div class="row"><span>Type</span><span>${escapeHtml(data.type)}</span></div>`;
  if (data.community !== undefined) {
    html += `<div class="row"><span>Community</span><span>${data.community}</span></div>`;
  }
  html += `<div class="row"><span>Connections</span><span>${node.degree()}</span></div>`;
  if (data.raw_data && Object.keys(data.raw_data).length > 0) {
    html += `<pre>${escapeHtml(JSON.stringify(data.raw_data, null, 2))}</pre>`;
  }
  infoEl.innerHTML = html;
  infoEl.style.display = 'block';
});
cy.on('tap', (evt) => {
  if (evt.target === cy) infoEl.style.display = 'none';
});

function escapeHtml(s) {
  if (s == null) return '';
  const map = {
    '&': '&amp;', '<': '&lt;', '>': '&gt;',
    '"': '&quot;', "'": '&#39;',
  };
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
