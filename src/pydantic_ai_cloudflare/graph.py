"""Entity Graph — graph-based feature engineering for ML + LLM analysis.

Builds a typed entity graph from tabular data where each row is an
entity node, column values become feature nodes, and edges represent
real relationships. Then computes graph-derived features (degree,
centrality, community, shared neighbors, embedding similarity) that
export as columns for downstream ML models.

Two purposes:
1. Chat/query: find similar entities, explain connections, explore neighborhoods
2. ML features: export graph metrics as features for XGBoost, LightGBM, etc.

    from pydantic_ai_cloudflare import EntityGraph

    kg = EntityGraph()
    await kg.build_from_records(data, id_column="account_id")

    # Chat use case
    similar = await kg.find_similar("NexaTech", top_k=5)

    # ML feature engineering use case
    features = kg.to_feature_dicts()  # → dict of {entity_id: {feature: value}}

    # Point-in-time training (no leakage):
    kg = EntityGraph()
    await kg.build_from_records(data, id_column="account_id",
                                time_column="event_date", as_of="2024-01-31")

    # Freeze for production scoring:
    kg.freeze(target_columns=["products_owned"], k=5)
    features = await kg.score_one(new_record)  # uses frozen topology

    # Visualize:
    kg.render_html("graph.html", color_by="community")
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .visualization import GraphVisualizer

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response
from .structured import cf_structured

logger = logging.getLogger(__name__)
_GENSIM_WARNED = False

# Sentinel string values that should never become feature nodes.
_NULL_STRINGS = frozenset(["", "none", "null", "nan", "n/a", "na", "unknown", "-"])

# Default LLM concurrency for parallel extraction.
_DEFAULT_LLM_CONCURRENCY = 8


# ============================================================
# Public configuration
# ============================================================


@dataclass
class GraphConfig:
    """Bundle of build-time configuration for ``EntityGraph.build_from_records``.

    Useful when you'd rather pass one object than 20+ keyword arguments
    — every field corresponds 1:1 to a kwarg of ``build_from_records``,
    and defaults match. ``GraphConfig()`` produces the same behavior as
    ``build_from_records(records)`` with no extras.

    Example::

        from pydantic_ai_cloudflare import EntityGraph, GraphConfig

        config = GraphConfig(
            id_column="account_id",
            categorical_columns=["industry", "geo"],
            list_columns={"tech": "USES_TECH", "products": "HAS_PRODUCT"},
            time_column="created_at",
            as_of="2024-01-31",
            extract_entities=False,
            compute_similarity=False,
        )
        kg = EntityGraph()
        await kg.build_from_config(records, config)
    """

    id_column: str | None = None
    data_dict: Any | None = None
    text_columns: list[str] | None = None
    categorical_columns: list[str] | None = None
    numeric_columns: list[str] | None = None
    list_columns: dict[str, str] | None = None
    relationship_columns: dict[str, str] | None = None
    sentinel_zero_columns: list[str] | None = None
    auto_detect_sentinels: bool = True
    sentinel_zero_rate: float = 0.30
    extract_entities: bool = True
    extract_relationships: bool = False
    compute_similarity: bool = True
    summarize_text: bool = False
    similarity_threshold: float = 0.70
    time_column: str | None = None
    as_of: Any | None = None
    temporal_decay: float = 0.0
    outcome_column: str | None = None
    confidence_map: dict[str, float] | None = None
    llm_concurrency: int = _DEFAULT_LLM_CONCURRENCY
    profile: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        """Return the config as a kwargs dict for ``build_from_records``.

        Uses shallow extraction (no deep-copy) so non-dataclass objects
        like ``DataDictionary`` pass through unchanged.
        """
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


# ============================================================
# Graph primitives
# ============================================================


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _adamic_adar(graph: dict[str, set[str]], node_a: str, node_b: str) -> float:
    """Adamic-Adar index: sum of 1/log(degree) for shared neighbors."""
    neighbors_a = graph.get(node_a, set())
    neighbors_b = graph.get(node_b, set())
    shared = neighbors_a & neighbors_b
    score = 0.0
    for n in shared:
        degree = len(graph.get(n, set()))
        if degree > 1:
            score += 1.0 / math.log(degree)
    return score


def _idf_weight(degree: int) -> float:
    """Inverse-frequency weight for traversing through a node of given degree.

    A feature node connected to many entities (a "hub") carries less signal
    than a rare feature node. Returns 1/log(degree+1), so:

        degree=2  -> 0.91  (rare feature, full weight)
        degree=10 -> 0.42
        degree=50 -> 0.26
        degree=170 -> 0.19  (common hub feature, suppressed)

    This is the same idea as Adamic-Adar applied during similarity traversal,
    instead of as a pair-wise feature.
    """
    if degree <= 1:
        return 1.0
    return 1.0 / math.log(degree + 1)


def _parse_date(value: Any) -> datetime | None:
    """Parse a date value to datetime. Returns None if unparseable.

    Accepts ISO strings, common formats, datetime objects, and timestamps.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (ValueError, OSError):
            return None
    s = str(value).strip()
    if not s:
        return None
    # Try fromisoformat first (Python 3.11+ accepts most ISO variants)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _profile_columns_for_warnings(
    records: list[dict[str, Any]],
    *,
    categorical_columns: list[str] | None,
    list_columns: dict[str, str] | None,
    id_column: str | None,
) -> list[str]:
    """Profile columns at build time and return user-readable warnings.

    Catches common footguns:
      - high-null categoricals (mass-link many entities to a single null bucket)
      - very-low-cardinality categoricals (everyone shares the same value)
      - very-high-cardinality categoricals (every entity has its own — useless)
    """
    warnings: list[str] = []
    if not records:
        return warnings
    n = len(records)

    for col in categorical_columns or []:
        if col == id_column:
            continue
        non_null = 0
        unique: set[str] = set()
        for r in records:
            v = r.get(col)
            if v is None:
                continue
            s = str(v).strip().lower()
            if s in _NULL_STRINGS:
                continue
            non_null += 1
            unique.add(s)
        null_rate = 1 - (non_null / n) if n > 0 else 1.0
        cardinality = len(unique)
        if null_rate > 0.5:
            warnings.append(
                f"Categorical column '{col}': {null_rate:.0%} null/sentinel — "
                "may not contribute meaningful similarity signal."
            )
        elif cardinality == 1 and non_null > 0:
            warnings.append(
                f"Categorical column '{col}': only 1 unique value across {non_null} "
                "records — every entity will share this feature node, dominating "
                "similarity. Consider dropping this column."
            )
        elif cardinality > 0 and cardinality / max(non_null, 1) > 0.95 and non_null > 20:
            warnings.append(
                f"Categorical column '{col}': {cardinality}/{non_null} unique "
                "values (almost all distinct) — won't link any entities. "
                "Consider 'text' or 'skip' instead."
            )

    for col in list_columns or {}:
        empty = 0
        for r in records:
            v = str(r.get(col, "") or "").strip().lower()
            if v in _NULL_STRINGS:
                empty += 1
        if n > 0 and empty / n > 0.5:
            warnings.append(
                f"List column '{col}': {empty}/{n} ({empty / n:.0%}) records "
                "have empty/null values for this column."
            )

    return warnings


def _entity_projection(
    adj: dict[str, set[str]],
    typed_adj: dict[str, list[dict[str, Any]]],
    nodes: dict[str, dict[str, Any]],
    entity_ids: list[str],
    weight: str = "adamic_adar",
) -> dict[str, dict[str, float]]:
    """Project a bipartite entity-feature graph onto an entity-only graph.

    For each pair of entities, computes a weighted edge based on shared
    feature nodes. Heavy hubs are downweighted automatically (Adamic-Adar)
    so noise-feature nodes (like "propensity_score=zero" with 170
    accounts attached) don't dominate the entity-entity weights.

    Args:
        adj: Untyped adjacency.
        typed_adj: Typed adjacency (used for entity-to-entity direct edges).
        nodes: Node lookup.
        entity_ids: List of entity node IDs.
        weight: "adamic_adar" (recommended), "jaccard", or "shared_count".

    Returns:
        {entity_nid: {other_entity_nid: weight}} where weight > 0.
    """
    proj: dict[str, dict[str, float]] = defaultdict(dict)
    entity_set = set(entity_ids)

    # Per-entity feature-only neighbor sets (skip entity-to-entity edges,
    # which we handle separately to avoid double-counting).
    feature_neighbors: dict[str, set[str]] = {}
    for nid in entity_ids:
        feats: set[str] = set()
        for n in adj.get(nid, set()):
            if nodes.get(n, {}).get("type") != "entity":
                feats.add(n)
        feature_neighbors[nid] = feats

    n_entities = len(entity_ids)
    for i in range(n_entities):
        a = entity_ids[i]
        feats_a = feature_neighbors[a]
        if not feats_a:
            continue
        for j in range(i + 1, n_entities):
            b = entity_ids[j]
            feats_b = feature_neighbors[b]
            if not feats_b:
                continue
            shared = feats_a & feats_b
            if not shared:
                continue
            if weight == "adamic_adar":
                w = sum(
                    1.0 / math.log(len(adj.get(s, set())))
                    for s in shared
                    if len(adj.get(s, set())) > 1
                )
            elif weight == "jaccard":
                w = len(shared) / len(feats_a | feats_b)
            else:  # shared_count
                w = float(len(shared))
            if w > 0:
                proj[a][b] = w
                proj[b][a] = w

    # Add direct entity-to-entity edges (e.g. COMPETES_WITH) at fixed weight.
    for src, edges in typed_adj.items():
        if src not in entity_set:
            continue
        for e in edges:
            tgt = e.get("target")
            if tgt in entity_set and tgt != src:
                # Boost direct relationships — they're real signal.
                proj[src][tgt] = proj[src].get(tgt, 0.0) + e.get("weight", 1.0) * 2.0

    return dict(proj)


# ============================================================
# Community detection (label propagation — no deps needed)
# ============================================================


def _label_propagation(adj: dict[str, set[str]], max_iter: int = 20) -> dict[str, int]:
    """Simple label propagation for community detection. Pure Python."""
    labels = {node: i for i, node in enumerate(adj)}
    nodes = list(adj.keys())
    rng = random.Random(42)

    for _ in range(max_iter):
        changed = False
        rng.shuffle(nodes)

        for node in nodes:
            neighbors = adj.get(node, set())
            if not neighbors:
                continue

            # Count neighbor labels
            label_counts: dict[int, int] = defaultdict(int)
            for n in neighbors:
                label_counts[labels.get(n, -1)] += 1

            # Pick most common
            best_label = max(label_counts, key=lambda k: label_counts[k])
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True

        if not changed:
            break

    # Renumber communities sequentially
    unique = {}
    counter = 0
    result = {}
    for node, label in labels.items():
        if label not in unique:
            unique[label] = counter
            counter += 1
        result[node] = unique[label]

    return result


# ============================================================
# EntityGraph class
# ============================================================


class EntityGraph:
    """Entity graph for peer-adoption features and ML feature engineering.

    Builds a bipartite graph from tabular data where entity nodes (rows)
    connect to feature-value nodes (column values). Also supports direct
    entity-to-entity relationships via add_relationship().

    Core value: finds structurally similar entities by shared feature
    nodes, then computes peer-adoption rates (knn_rate_*) as ML features.
    "What do my graph peers have that I don't?" — a signal flat tables miss.

    This is NOT a full knowledge graph with ontologies and inference rules.
    It's a structured collaborative filter with typed edges, community
    detection, and co-occurrence lift.
    """

    def __init__(
        self,
        name: str = "default",
        *,
        embedding_model: str = "@cf/baai/bge-base-en-v1.5",
        extraction_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        canonical_map: dict[str, str] | None = None,
        account_id: str | None = None,
        api_key: str | None = None,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._name = name
        self._embedding_model = embedding_model
        self._extraction_model = extraction_model
        self._canonical_map = _build_canonical_lookup(canonical_map or {})
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._timeout = request_timeout
        self._headers = build_headers(self._api_key)

        # Graph storage
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._adj: dict[str, set[str]] = defaultdict(set)
        self._typed_adj: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._embeddings: dict[str, list[float]] = {}
        self._graph_embeddings: dict[str, list[float]] = {}
        self._entity_ids: list[str] = []
        self._outcome_column: str | None = None  # set during build

        # Computed features (lazily populated)
        self._communities: dict[str, int] | None = None
        self._features: dict[str, dict[str, Any]] | None = None

        # Build metadata — captured at build_from_records() time so
        # score_one() / score_batch() can replay the same column logic.
        self._build_meta: dict[str, Any] = {}
        self._snapshot_date: str | None = None  # set if as_of provided
        self._build_warnings: list[str] = []

        # Freeze state — locks the graph for production scoring.
        self._frozen: bool = False
        self._frozen_target_columns: list[str] | None = None
        self._frozen_target_values: list[str] = []
        self._frozen_k: int = 5
        self._frozen_label_to_nid: dict[str, str] = {}

        # Track the most recent target_columns/k passed to knn_rate_features
        # or to_ml_dataset, so save_features can warn loudly if the caller
        # later forgets to pass them and silently saves a smaller feature set.
        self._last_knn_target_columns: list[str] | None = None
        self._last_knn_k: int | None = None

    # -- Repr / containment / iteration --
    #
    # Cheap usability dunders. ``len(kg)`` returns entity count,
    # ``"AcmeCorp" in kg`` checks membership (case-insensitive via
    # ``_resolve_entity``), ``for label in kg`` iterates entity labels.

    def __repr__(self) -> str:
        n_entities = len(self._entity_ids)
        if n_entities == 0:
            return f"<EntityGraph name={self._name!r} (empty — call build_from_records)>"
        state = " frozen" if self._frozen else ""
        snap = f" snapshot={self._snapshot_date!r}" if self._snapshot_date else ""
        return (
            f"<EntityGraph name={self._name!r} entities={n_entities} "
            f"nodes={len(self._nodes)} edges={len(self._edges)}{state}{snap}>"
        )

    def __len__(self) -> int:
        """Number of entity nodes in the graph."""
        return len(self._entity_ids)

    def __contains__(self, entity: object) -> bool:
        """``"AcmeCorp" in kg`` checks if an entity exists (case-insensitive)."""
        if not isinstance(entity, str):
            return False
        return self._resolve_entity(entity) is not None

    def __iter__(self) -> Iterator[str]:
        """Iterate over entity labels in build/insertion order."""
        for nid in self._entity_ids:
            node = self._nodes.get(nid)
            if node is not None:
                yield str(node.get("label", ""))

    @property
    def build_warnings(self) -> list[str]:
        """Read-only copy of warnings collected during the most recent build.

        Each warning is a one-line string explaining a potential data-quality
        issue (high null rates, sentinel zeros, low-cardinality columns, etc.).
        Surfaced automatically via the ``pydantic_ai_cloudflare.graph`` logger
        at WARNING level — to see them, run::

            import logging
            logging.basicConfig(level=logging.WARNING)
        """
        return list(self._build_warnings)

    # -- Internal: warning helper --

    def _warn(self, msg: str) -> None:
        """Record a build-time warning AND emit it to the logger.

        Using this everywhere keeps warnings discoverable: stored on
        ``self._build_warnings`` (also exposed via :pyattr:`build_warnings`)
        so they're inspectable after build, *and* logged so they show up
        in stderr by default.
        """
        self._build_warnings.append(msg)
        logger.warning(msg)

    # -- Node/edge helpers --

    def _canonicalize(self, value: str) -> str:
        """Resolve aliases to canonical form."""
        return self._canonical_map.get(value.lower().strip(), value)

    async def auto_canonicalize(
        self,
        records: list[dict[str, Any]],
        columns: list[str],
        *,
        model: str | None = None,
        min_group_size: int = 2,
    ) -> dict[str, str]:
        """Use LLM to auto-detect and merge entity aliases.

        Scans the specified columns, collects all unique values,
        asks the LLM to group them into canonical forms, and
        updates the internal canonical map.

        CAUTION: The LLM may over-merge distinct entities that share
        a common prefix (e.g. "Magic WAN" and "Magic Transit" are
        different products). Review the output before building the graph.

        Args:
            records: The dataset (same records you'll pass to build_from_records).
            columns: Which columns to scan for aliases.
            model: Override LLM model.
            min_group_size: Minimum aliases in a group to include (2 = only
                merge when there are at least 2 variants). Set higher for
                more conservative merging.

        Returns:
            The generated alias map {alias: canonical}.
        """
        from pydantic import BaseModel as _BM

        # Collect unique values from list/categorical columns
        all_values: set[str] = set()
        for record in records:
            for col in columns:
                raw = str(record.get(col, ""))
                for item in raw.split(","):
                    item = item.strip()
                    if item and item.lower() not in ("", "none", "null", "nan"):
                        all_values.add(item)

        if len(all_values) < 3:
            return {}

        class _Group(_BM):
            canonical: str
            aliases: list[str]

        class _Map(_BM):
            groups: list[_Group]

        values_list = sorted(all_values)
        # Batch if too many (LLM context limit)
        batch_size = 200
        full_map: dict[str, str] = {}

        for i in range(0, len(values_list), batch_size):
            batch = values_list[i : i + batch_size]
            result = await cf_structured(
                f"Group these entity mentions into canonical forms. "
                f"Each group has one canonical name and all its aliases/variants. "
                f"Only group things that are truly the same entity.\n\n"
                f"Entities: {batch}",
                _Map,
                model=model or self._extraction_model,
                account_id=self._account_id,
                api_key=self._api_key,
                max_tokens=4096,
            )
            for g in result.groups:
                if len(g.aliases) < min_group_size:
                    continue  # skip single-item groups (no real aliases)
                for alias in g.aliases:
                    full_map[alias.lower().strip()] = g.canonical

        # Update internal map
        self._canonical_map.update(full_map)
        return full_map

    def _nid(self, ntype: str, value: str) -> str:
        return f"{ntype}:{value}".lower().strip()

    def _add_node(self, ntype: str, value: str, data: dict | None = None) -> str:
        # Canonicalize non-entity nodes (collapse "Zscaler" / "ZS" / "zscaler inc")
        if ntype != "entity":
            value = self._canonicalize(value)
        nid = self._nid(ntype, value)
        if nid not in self._nodes:
            self._nodes[nid] = {"id": nid, "type": ntype, "label": value, "data": data or {}}
        elif data:
            self._nodes[nid]["data"].update(data)
        return nid

    def _add_edge(
        self,
        src: str,
        tgt: str,
        etype: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        outcome: str | None = None,
    ) -> None:
        # Check for duplicate — if exists, accumulate weight (multi-source support)
        for e in self._typed_adj[src]:
            if e["target"] == tgt and e["type"] == etype:
                e["weight"] = max(e["weight"], weight)  # take stronger
                e["confidence"] = max(e["confidence"], confidence)
                return
        edge = {
            "source": src,
            "target": tgt,
            "type": etype,
            "weight": weight,
            "confidence": confidence,
            "outcome": outcome,
        }
        self._edges.append(edge)
        self._typed_adj[src].append(edge)
        self._typed_adj[tgt].append({**edge, "source": tgt, "target": src})
        self._adj[src].add(tgt)
        self._adj[tgt].add(src)
        self._features = None

    # -- Entity-to-entity relationships --

    def add_relationship(
        self,
        source_entity: str,
        relationship: str,
        target_entity: str,
        *,
        weight: float = 1.0,
        confidence: float = 1.0,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Add a direct entity-to-entity relationship.

        This is what makes it more than a bipartite feature graph.
        Enables multi-hop traversal and causal reasoning.

        Examples:
            kg.add_relationship("Cisco", "COMPETES_WITH", "Zscaler")
            kg.add_relationship("Acme Corp", "DISPLACED", "Palo Alto")
            kg.add_relationship("Account A", "REFERRED_BY", "Account B")

        Args:
            source_entity: Source entity ID or name.
            target_entity: Target entity ID or name.
            relationship: Edge type (e.g. COMPETES_WITH, PARTNERS_WITH).
            weight: Edge weight.
            confidence: Confidence score (1.0 = certain, 0.5 = inferred).
            data: Extra metadata on the edge.
        """
        # Resolve or create both entities
        src = self._resolve_entity(source_entity)
        if src is None:
            src = self._add_node("entity", source_entity, data)

        tgt = self._resolve_entity(target_entity)
        if tgt is None:
            tgt = self._add_node("entity", target_entity, data)

        self._add_edge(src, tgt, relationship, weight=weight, confidence=confidence)

    # -- API helpers --

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        all_embs: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"https://api.cloudflare.com/client/v4/accounts/"
                    f"{self._account_id}/ai/run/{self._embedding_model}",
                    headers=self._headers,
                    json={"text": batch},
                )
                resp.raise_for_status()
            data = resp.json()
            check_api_response(data)
            all_embs.extend(data.get("result", {}).get("data", []))
        return all_embs

    async def _extract_entities(self, text: str) -> list[str]:
        from pydantic import BaseModel

        class E(BaseModel):
            entities: list[str]

        try:
            r = await cf_structured(
                f"Extract key entities (companies, technologies, concepts, "
                f"products, industries, tools) from:\n\n{text[:2000]}",
                E,
                model=self._extraction_model,
                account_id=self._account_id,
                api_key=self._api_key,
                max_tokens=512,
                temperature=0.1,
                retries=1,
            )
            return [e.lower().strip() for e in r.entities if e.strip()]
        except Exception:
            # Fallback: capitalized phrases
            import re

            return list(
                set(w.lower() for w in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))
            )[:10]

    async def _extract_relationships(
        self,
        text: str,
        source_entity: str,
    ) -> list[dict[str, str]]:
        """Extract entity-to-entity relationships from text via LLM."""
        from pydantic import BaseModel as _BM

        class _Rel(_BM):
            source: str
            type: str
            target: str

        class _Rels(_BM):
            relationships: list[_Rel]

        try:
            r = await cf_structured(
                f"Extract relationships from text about {source_entity}. "
                f"Types: COMPETES_WITH, PARTNERS_WITH, ACQUIRED, DISPLACED, "
                f"MIGRATED_TO, USES, SUPPLIES.\n\n"
                f"{text[:2000]}",
                _Rels,
                model=self._extraction_model,
                account_id=self._account_id,
                api_key=self._api_key,
                max_tokens=512,
                retries=1,
            )
            return [{"target": rel.target, "type": rel.type} for rel in r.relationships]
        except Exception:
            return []

    async def _summarize_text(self, text: str, entity_name: str = "") -> str:
        """Summarize a long text column value for an entity."""
        from pydantic import BaseModel

        class Summary(BaseModel):
            summary: str
            key_points: list[str]

        try:
            r = await cf_structured(
                f"Summarize this text about {entity_name} into a concise "
                f"summary (2-3 sentences) and 3-5 key points:\n\n{text[:3000]}",
                Summary,
                model=self._extraction_model,
                account_id=self._account_id,
                api_key=self._api_key,
                max_tokens=512,
                temperature=0.1,
                retries=1,
            )
            return f"{r.summary} Key points: {'; '.join(r.key_points)}"
        except Exception:
            # Fallback: take first 500 chars
            return text[:500]

    # ============================================================
    # Build graph
    # ============================================================

    async def build_from_records(
        self,
        records: list[dict[str, Any]],
        *,
        id_column: str | None = None,
        data_dict: Any | None = None,
        text_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        list_columns: dict[str, str] | None = None,
        relationship_columns: dict[str, str] | None = None,
        sentinel_zero_columns: list[str] | None = None,
        auto_detect_sentinels: bool = True,
        sentinel_zero_rate: float = 0.30,
        extract_entities: bool = True,
        extract_relationships: bool = False,
        compute_similarity: bool = True,
        summarize_text: bool = False,
        similarity_threshold: float = 0.70,
        time_column: str | None = None,
        as_of: Any | None = None,
        temporal_column: str | None = None,
        temporal_decay: float = 0.0,
        outcome_column: str | None = None,
        confidence_map: dict[str, float] | None = None,
        llm_concurrency: int = _DEFAULT_LLM_CONCURRENCY,
        profile: bool = True,
    ) -> dict[str, int]:
        """Build graph from tabular records.

        Accepts either explicit column lists OR a DataDictionary
        from profile_data(). If neither provided, auto-detects types.

        Args:
            records: List of row dicts.
            id_column: Entity ID column. Not needed if data_dict provided.
            data_dict: A DataDictionary from profile_data(). Overrides
                individual column parameters.
            text_columns: Columns with long text (embedded + entity-extracted).
            categorical_columns: Columns with categorical values.
            numeric_columns: Columns with numbers.
            list_columns: {column: edge_type} for comma-separated lists.
            relationship_columns: {column: edge_type} for direct entity edges.
            sentinel_zero_columns: Numeric columns where 0 means "missing".
                These zeros are excluded from the graph so they don't create
                fake similarity (e.g. propensity_score=0 for unknown leads).
            auto_detect_sentinels: If True, profile numeric columns and treat
                values that appear in >sentinel_zero_rate of records as
                missing (default 30%).
            sentinel_zero_rate: Threshold for auto-sentinel detection.
            extract_entities: Use LLM to extract entities from text.
            extract_relationships: Use LLM to extract typed entity-to-entity
                relationships from text columns (e.g. COMPETES_WITH).
            compute_similarity: Add SIMILAR_TO edges from embeddings.
            summarize_text: Summarize long text columns per entity.
            similarity_threshold: Min cosine sim for SIMILAR_TO edges.
            time_column: Date column to use for point-in-time filtering.
                When combined with `as_of`, only records on/before that date
                are included in the graph. This eliminates training-time
                data leakage when using these features for ML.
            as_of: Cutoff date (str or datetime). Only records where
                `time_column <= as_of` are included.
            temporal_column: Alias for time_column (back-compat).
                If both provided, time_column wins.
            temporal_decay: Decay rate λ. Edge weight = exp(-λ × days_ago).
                0 = no decay (default), 0.01 = gradual, 0.1 = aggressive.
                Decay is computed relative to as_of (or now if as_of is None).
            outcome_column: Column with the outcome label (Won/Lost/Churned).
            confidence_map: Per-column edge confidence override.
            llm_concurrency: Max concurrent LLM calls for entity/relationship
                extraction (default 8). Lower if hitting rate limits; higher
                for fast accounts. The library batches all extraction jobs
                across records and runs them via asyncio.gather + Semaphore,
                so 1000-record graphs no longer take 30 minutes.
            profile: If True, profiles each column at build time and emits
                warnings about high-null-rate or low-cardinality columns
                that may pollute similarity. Set False to silence.

        Returns:
            {"nodes": int, "edges": int} stats.
        """
        if not records:
            return {"nodes": 0, "edges": 0}

        if self._frozen:
            raise RuntimeError(
                "Cannot rebuild a frozen graph — freeze() locks the topology so "
                "training-time and inference-time features stay identical. "
                "Two ways forward:\n"
                "  • For NEW records (inference): use kg.score_one(record) "
                "or kg.score_batch(records).\n"
                "  • To rebuild from scratch (re-training): call kg.unfreeze() "
                "first, then build_from_records()."
            )

        self._outcome_column = outcome_column
        self._build_warnings = []
        # Confidence per column type (structured > list > text-extracted)
        conf_map = confidence_map or {}

        # Use DataDictionary if provided
        if data_dict is not None:
            id_column = data_dict.id_column
            text_columns = data_dict.text_columns
            categorical_columns = data_dict.categorical_columns
            numeric_columns = data_dict.numeric_columns
            list_columns = data_dict.list_columns
        else:
            text_columns = text_columns or []
            categorical_columns = categorical_columns or []
            numeric_columns = numeric_columns or []
            list_columns = list_columns or {}

        if id_column is None:
            from .data_profiler import _detect_id_column

            id_column = _detect_id_column(records[:100], list(records[0].keys()))

        # Auto-detect if nothing specified
        if not any([text_columns, categorical_columns, numeric_columns, list_columns]):
            sample = records[0]
            for col, val in sample.items():
                if col == id_column:
                    continue
                if isinstance(val, str) and len(val) > 100:
                    text_columns.append(col)
                elif isinstance(val, str) and "," in str(val):
                    list_columns[col] = f"HAS_{col.upper()}"
                elif isinstance(val, str):
                    categorical_columns.append(col)
                elif isinstance(val, (int, float)):
                    numeric_columns.append(col)

        # ----- Point-in-time filter -----
        # time_column wins over deprecated temporal_column.
        effective_time_col = time_column or temporal_column
        cutoff_dt: datetime | None = None
        if as_of is not None and effective_time_col is None:
            self._warn(
                "as_of provided without time_column — point-in-time filter not applied. "
                "Pass time_column='created_at' (or your event-date column) to enable it."
            )
        if as_of is not None and effective_time_col is not None:
            cutoff_dt = _parse_date(as_of)
            if cutoff_dt is None:
                raise ValueError(f"as_of value not parseable as date: {as_of!r}")
            self._snapshot_date = cutoff_dt.isoformat()
            before = len(records)
            records = [
                r
                for r in records
                if (parsed := _parse_date(r.get(effective_time_col))) is not None
                and parsed <= cutoff_dt
            ]
            logger.info(
                f"Point-in-time filter: kept {len(records)}/{before} records "
                f"with {effective_time_col} <= {self._snapshot_date}"
            )

        # ----- Sentinel-zero detection -----
        sentinel_set = set(sentinel_zero_columns or [])
        if auto_detect_sentinels and numeric_columns:
            for col in numeric_columns:
                zero_count = 0
                total = 0
                for r in records:
                    raw = r.get(col)
                    if raw is None:
                        continue
                    try:
                        v = float(
                            str(raw)
                            .replace("$", "")
                            .replace(",", "")
                            .replace("M", "e6")
                            .replace("B", "e9")
                            .replace("K", "e3")
                        )
                    except (ValueError, TypeError):
                        continue
                    total += 1
                    if v == 0:
                        zero_count += 1
                if total > 0 and zero_count / total >= sentinel_zero_rate:
                    if col not in sentinel_set:
                        sentinel_set.add(col)
                        self._warn(
                            f"Numeric column '{col}': {zero_count}/{total} "
                            f"({zero_count / total:.0%}) values are 0. Treating 0 as "
                            f"missing (excluded from graph) to avoid fake similarity. "
                            f"Pass sentinel_zero_columns=[] to disable."
                        )

        # ----- Profile other columns for warnings -----
        if profile:
            for w in _profile_columns_for_warnings(
                records,
                categorical_columns=categorical_columns,
                list_columns=list_columns,
                id_column=id_column,
            ):
                self._warn(w)

        n_records = len(records)
        log_interval = max(n_records // 10, 100)  # log every 10% or 100 records

        # ===== Pass 1: structural edges (no LLM) =====
        for rec_idx, record in enumerate(records):
            if rec_idx > 0 and rec_idx % log_interval == 0:
                logger.info(f"Building graph: {rec_idx}/{n_records} records processed")

            eid = str(record.get(id_column, ""))
            if not eid:
                continue

            row_data = {k: v for k, v in record.items() if k != id_column}
            entity_nid = self._add_node("entity", eid, row_data)
            if entity_nid not in self._entity_ids:
                self._entity_ids.append(entity_nid)

            # Compute temporal decay weight for this record
            edge_weight = 1.0
            if effective_time_col and temporal_decay > 0:
                date_val = record.get(effective_time_col)
                dt = _parse_date(date_val)
                if dt is not None:
                    reference_dt = cutoff_dt or datetime.now()
                    days_ago = (reference_dt - dt).days
                    edge_weight = math.exp(-temporal_decay * max(days_ago, 0))

            # Resolve outcome for this record
            record_outcome = str(record.get(outcome_column, "")) if outcome_column else None

            # Categorical → feature nodes
            for col in categorical_columns:
                val = str(record.get(col, "")).strip()
                if val and val.lower() not in _NULL_STRINGS:
                    fid = self._add_node(col, val)
                    conf = conf_map.get(col, 1.0)
                    self._add_edge(
                        entity_nid,
                        fid,
                        f"HAS_{col.upper()}",
                        weight=edge_weight,
                        confidence=conf,
                        outcome=record_outcome,
                    )

            # List columns → split on comma
            for col, etype in list_columns.items():
                val = str(record.get(col, ""))
                conf = conf_map.get(col, 0.9)
                for item in (x.strip() for x in val.split(",") if x.strip()):
                    if item.lower() in _NULL_STRINGS:
                        continue
                    iid = self._add_node(col, item)
                    self._add_edge(
                        entity_nid,
                        iid,
                        etype,
                        weight=edge_weight,
                        confidence=conf,
                        outcome=record_outcome,
                    )

            # Numeric → bucketed ranges (with sentinel filter)
            for col in numeric_columns:
                raw = record.get(col)
                if raw is None:
                    continue
                try:
                    num = float(
                        str(raw)
                        .replace("$", "")
                        .replace(",", "")
                        .replace("M", "e6")
                        .replace("B", "e9")
                        .replace("K", "e3")
                    )
                except (ValueError, TypeError):
                    continue
                # Skip sentinel zeros — they mean "missing" not "really zero"
                if num == 0 and col in sentinel_set:
                    continue
                bucket = _bucket_number(num)
                bid = self._add_node(f"{col}_range", bucket)
                self._add_edge(
                    entity_nid,
                    bid,
                    f"IN_{col.upper()}_RANGE",
                    weight=edge_weight,
                    outcome=record_outcome,
                )

            # Relationship columns → direct entity-to-entity edges
            if relationship_columns:
                for col, rel_type in relationship_columns.items():
                    val = str(record.get(col, "")).strip()
                    if val and val.lower() not in _NULL_STRINGS:
                        # Each value becomes a target entity with a direct relationship
                        for item in (x.strip() for x in val.split(",") if x.strip()):
                            if item.lower() in _NULL_STRINGS:
                                continue
                            item_canon = self._canonicalize(item)
                            target_nid = self._add_node("entity", item_canon)
                            self._add_edge(
                                entity_nid,
                                target_nid,
                                rel_type,
                                weight=edge_weight,
                                confidence=conf_map.get(col, 0.9),
                                outcome=record_outcome,
                            )

        # ===== Pass 2: parallel LLM extraction =====
        if (extract_entities or extract_relationships or summarize_text) and text_columns:
            await self._extract_text_features_parallel(
                records=records,
                id_column=id_column,
                text_columns=text_columns,
                extract_entities=extract_entities,
                extract_relationships=extract_relationships,
                summarize_text=summarize_text,
                concurrency=llm_concurrency,
            )

        # ===== Pass 3: embeddings + similarity edges =====
        if compute_similarity and text_columns:
            texts_map: dict[str, str] = {}
            for record in records:
                eid = str(record.get(id_column, ""))
                if not eid:
                    continue
                nid = self._nid("entity", eid)
                combined = " ".join(str(record.get(c, "")) for c in text_columns).strip()
                if combined:
                    texts_map[nid] = combined

            if texts_map:
                nids = list(texts_map.keys())
                embs = await self._embed(list(texts_map.values()))
                for nid, emb in zip(nids, embs):
                    self._embeddings[nid] = emb

                # Pairwise SIMILAR_TO (still O(n²) — for >5K entities, prefer
                # using Vectorize and querying it as a separate step).
                if len(nids) > 5000:
                    self._warn(
                        f"Computing all-pairs similarity over {len(nids)} entities is O(n²) "
                        "(~12M comparisons). Consider compute_similarity=False and using "
                        "Vectorize for similarity queries instead."
                    )
                for i in range(len(nids)):
                    for j in range(i + 1, len(nids)):
                        sim = _cosine_sim(embs[i], embs[j])
                        if sim >= similarity_threshold:
                            self._add_edge(nids[i], nids[j], "SIMILAR_TO", weight=round(sim, 4))

        # Capture build metadata for score_one() reproducibility.
        self._build_meta = {
            "id_column": id_column,
            "categorical_columns": list(categorical_columns or []),
            "numeric_columns": list(numeric_columns or []),
            "list_columns": dict(list_columns or {}),
            "text_columns": list(text_columns or []),
            "relationship_columns": dict(relationship_columns or {}),
            "sentinel_zero_columns": list(sentinel_set),
            "outcome_column": outcome_column,
            "time_column": effective_time_col,
            "as_of": self._snapshot_date,
            "n_records": n_records,
        }

        return {"nodes": len(self._nodes), "edges": len(self._edges)}

    async def build_from_config(
        self,
        records: list[dict[str, Any]],
        config: GraphConfig,
    ) -> dict[str, int]:
        """Build the graph using a :class:`GraphConfig` object.

        Equivalent to ``build_from_records(records, **config.to_kwargs())``
        but lets callers bundle the 20+ build parameters into one named
        object — useful for testing, persistence, and CLI tooling.

        Example::

            config = GraphConfig(
                id_column="account_id",
                categorical_columns=["industry"],
                list_columns={"tech": "USES_TECH"},
                extract_entities=False,
                compute_similarity=False,
            )
            await kg.build_from_config(records, config)
        """
        return await self.build_from_records(records, **config.to_kwargs())

    async def quick_build(
        self,
        records: list[dict[str, Any]],
        *,
        id_column: str | None = None,
        use_llm: bool = False,
    ) -> dict[str, int]:
        """One-line graph builder with sensible defaults — for the 80% use case.

        Auto-profiles the dataset via :func:`profile_data` and turns OFF the
        expensive options (LLM entity extraction, summarization, all-pairs
        similarity) by default. Best for fast first results on a CSV — e.g.
        ML feature work where you'll iterate later. Use
        :meth:`build_from_records` when you need full control.

        Args:
            records: List of row dicts (e.g. ``pd.read_csv(...).to_dict("records")``).
            id_column: Entity ID column. Auto-detected from the records if not
                provided.
            use_llm: If True, enables ``extract_entities=True`` and
                ``compute_similarity=True`` (requires Cloudflare credentials,
                slower). Default False — pure structural graph.

        Returns:
            ``{"nodes": int, "edges": int}`` — same as ``build_from_records``.

        Example::

            import pandas as pd
            from pydantic_ai_cloudflare import EntityGraph

            df = pd.read_csv("customers.csv")
            kg = EntityGraph()
            await kg.quick_build(df.to_dict("records"), id_column="customer_id")
            features = kg.to_feature_dicts()  # 22+ ML features per entity

        See Also:
            :meth:`build_from_records` — full-control build
            :meth:`build_from_config` — config-object build
            :func:`profile_data` — explicit profiling
        """
        if not records:
            return {"nodes": 0, "edges": 0}

        # Lazy import — keeps the data_profiler module out of cold-import paths.
        from .data_profiler import profile_data

        dd = profile_data(records, id_column=id_column)

        return await self.build_from_records(
            records,
            data_dict=dd,
            extract_entities=use_llm,
            extract_relationships=False,
            compute_similarity=use_llm,
            summarize_text=False,
            profile=True,
        )

    # -- Parallel LLM extraction helper --

    async def _extract_text_features_parallel(
        self,
        records: list[dict[str, Any]],
        *,
        id_column: str,
        text_columns: list[str],
        extract_entities: bool,
        extract_relationships: bool,
        summarize_text: bool,
        concurrency: int = _DEFAULT_LLM_CONCURRENCY,
    ) -> None:
        """Run all LLM extraction jobs concurrently with a semaphore.

        Replaces the per-record sequential loop. On 300 records with
        extract_entities=True, drops wall time from ~30 min to ~3 min.
        Mutations are applied sequentially after all coroutines resolve,
        keeping graph state consistent.
        """
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _wrap_entities(text: str) -> list[str] | None:
            async with sem:
                try:
                    return await self._extract_entities(text)
                except Exception as e:
                    logger.warning(f"entity extraction failed: {e}")
                    return None

        async def _wrap_rels(text: str, eid: str) -> list[dict[str, str]] | None:
            async with sem:
                try:
                    return await self._extract_relationships(text, eid)
                except Exception as e:
                    logger.warning(f"relationship extraction failed: {e}")
                    return None

        async def _wrap_summary(text: str, eid: str) -> str | None:
            async with sem:
                try:
                    return await self._summarize_text(text, eid)
                except Exception as e:
                    logger.warning(f"summarization failed: {e}")
                    return None

        # Collect (op, eid, col, awaitable) tuples
        jobs: list[tuple[str, str, str, Any]] = []
        for record in records:
            eid = str(record.get(id_column, ""))
            if not eid:
                continue
            for col in text_columns:
                text = str(record.get(col, ""))
                if len(text) < 20:
                    continue
                if extract_entities:
                    jobs.append(("entities", eid, col, _wrap_entities(text)))
                if extract_relationships:
                    jobs.append(("relationships", eid, col, _wrap_rels(text, eid)))
                if summarize_text and len(text) > 500:
                    jobs.append(("summary", eid, col, _wrap_summary(text, eid)))

        if not jobs:
            return

        logger.info(f"Running {len(jobs)} LLM extraction jobs (concurrency={concurrency})...")
        results = await asyncio.gather(*[j[3] for j in jobs])

        # Apply results sequentially — this mutates self._nodes/_edges so
        # we keep it single-threaded.
        for (op, eid, col, _), result in zip(jobs, results):
            if result is None:
                continue
            entity_nid = self._nid("entity", eid)
            if op == "entities" and isinstance(result, list):
                for ent in result:
                    if not isinstance(ent, str) or not ent.strip():
                        continue
                    cid = self._add_node("concept", ent)
                    self._add_edge(entity_nid, cid, "HAS_CONCEPT")
            elif op == "relationships" and isinstance(result, list):
                for rel in result:
                    if not isinstance(rel, dict):
                        continue
                    target = rel.get("target", "")
                    rtype = rel.get("type", "RELATED_TO")
                    if not target:
                        continue
                    target_nid = self._add_node("entity", str(target))
                    self._add_edge(
                        entity_nid,
                        target_nid,
                        str(rtype),
                        weight=0.8,
                        confidence=0.7,
                    )
            elif op == "summary" and isinstance(result, str):
                self._nodes[entity_nid]["data"][f"{col}_summary"] = result

    # ============================================================
    # Graph features for ML
    # ============================================================

    def compute_features(
        self,
        *,
        community_method: str = "entity_projection",
        community_resolution: float = 1.5,
        compute_node2vec: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Compute graph-derived features for every entity node.

        Returns a dict of {entity_id: {feature_name: value}}.

        Args:
            community_method: How to detect communities:
              - "entity_projection" (default): project the bipartite graph
                onto entity-only space using Adamic-Adar weighting, then
                run Louvain. Produces finer-grained, business-meaningful
                clusters (e.g. ~25 communities on 300 entities, vs ~5
                from running Louvain directly on the bipartite graph).
              - "bipartite": run Louvain on the full graph (legacy).
            community_resolution: Louvain resolution. Higher = more,
                smaller communities. Default 1.5.
            compute_node2vec: Whether to compute Node2Vec embeddings
                (requires gensim). Disable for very large graphs.

        Features computed:

        Structural:
          - degree: total edges
          - degree_entity: edges to other entities
          - degree_concept: edges to concept nodes
          - degree_by_type: {edge_type: count}
          - unique_neighbors: distinct connected nodes
          - clustering_coeff: local clustering coefficient

        Community:
          - community_id: Louvain community
          - community_size: how many entities in same community

        Centrality:
          - pagerank: simplified PageRank score (low signal on bipartite
            graphs; included for backward compat)

        Similarity:
          - avg_similarity: mean SIMILAR_TO edge weight
          - max_similarity: highest SIMILAR_TO weight
          - similar_count: number of SIMILAR_TO edges

        Embedding (Node2Vec):
          - n2v_norm: structural-position embedding magnitude
          - n2v_avg_neighbor_dist: mean distance to entity neighbors
        """
        if self._features is not None:
            return self._features

        n_entities = len(self._entity_ids)
        logger.info(f"Computing features for {n_entities} entities...")
        features: dict[str, dict[str, Any]] = {}

        # Community detection
        if community_method == "entity_projection":
            logger.info("Computing entity projection for community detection...")
            proj = _entity_projection(self._adj, self._typed_adj, self._nodes, self._entity_ids)
            # Convert weighted projection to unweighted adjacency for Louvain
            proj_adj: dict[str, set[str]] = {
                nid: set(neighbors.keys()) for nid, neighbors in proj.items()
            }
            # Add isolated entity nodes (no edges in projection)
            for nid in self._entity_ids:
                proj_adj.setdefault(nid, set())
            logger.info("Running Louvain on entity projection...")
            self._communities = _louvain_communities(proj_adj, resolution=community_resolution)
        else:
            logger.info("Running Louvain on bipartite graph...")
            self._communities = _louvain_communities(self._adj, resolution=community_resolution)

        community_sizes: dict[int, int] = defaultdict(int)
        for nid in self._entity_ids:
            cid = self._communities.get(nid, -1)
            community_sizes[cid] += 1

        # PageRank
        pr = _pagerank(self._adj, iterations=20)

        # Node2Vec graph embeddings via random walks + Word2Vec
        n2v: dict[str, list[float]] = {}
        if compute_node2vec:
            logger.info("Computing Node2Vec embeddings...")
            n2v = _node2vec_embeddings(
                self._adj,
                dimensions=32,
                walk_length=10,
                num_walks=5,
            )
            self._graph_embeddings = n2v

        for nid in self._entity_ids:
            f: dict[str, Any] = {}
            label = self._nodes[nid]["label"]

            # Degree features
            neighbors = self._adj.get(nid, set())
            typed_edges = self._typed_adj.get(nid, [])
            f["degree"] = len(typed_edges)
            f["unique_neighbors"] = len(neighbors)

            degree_by_type: dict[str, int] = defaultdict(int)
            entity_neighbors = set()
            concept_neighbors = set()
            similar_weights: list[float] = []

            for e in typed_edges:
                degree_by_type[e["type"]] += 1
                tgt_node = self._nodes.get(e["target"], {})
                if tgt_node.get("type") == "entity":
                    entity_neighbors.add(e["target"])
                elif tgt_node.get("type") == "concept":
                    concept_neighbors.add(e["target"])
                if e["type"] == "SIMILAR_TO":
                    similar_weights.append(e.get("weight", 0))

            f["degree_entity"] = len(entity_neighbors)
            f["degree_concept"] = len(concept_neighbors)
            f["degree_by_type"] = dict(degree_by_type)

            # Similarity features
            f["similar_count"] = len(similar_weights)
            f["avg_similarity"] = (
                sum(similar_weights) / len(similar_weights) if similar_weights else 0.0
            )
            f["max_similarity"] = max(similar_weights) if similar_weights else 0.0

            # Clustering coefficient
            f["clustering_coeff"] = _clustering_coefficient(self._adj, nid)

            # Community
            f["community_id"] = self._communities.get(nid, -1)
            f["community_size"] = community_sizes.get(f["community_id"], 0)

            # Centrality
            # NOTE: On bipartite feature graphs, PageRank measures feature-mediated
            # centrality (how common your feature values are), NOT entity importance.
            # Use community_id and knn_rate_* for more meaningful ML features.
            f["pagerank"] = pr.get(nid, 0.0)

            # Node2Vec embedding features (structural position)
            n2v_emb = n2v.get(nid)
            if n2v_emb:
                # Average distance to entity neighbors in embedding space
                n2v_dists = []
                for en in entity_neighbors:
                    en_emb = n2v.get(en)
                    if en_emb:
                        n2v_dists.append(1.0 - _cosine_sim(n2v_emb, en_emb))
                f["n2v_avg_neighbor_dist"] = sum(n2v_dists) / len(n2v_dists) if n2v_dists else 1.0
                # Embedding norm (how "central" in the structural space)
                f["n2v_norm"] = math.sqrt(sum(x * x for x in n2v_emb))

            features[label] = f

        self._features = features
        return features

    def pairwise_features(self, entity_a: str, entity_b: str) -> dict[str, float]:
        """Compute pair-wise features between two entities.

        Useful for ML models that predict relationships (link prediction,
        match scoring, propensity models).
        """
        nid_a = self._resolve_entity(entity_a) or self._nid("entity", entity_a)
        nid_b = self._resolve_entity(entity_b) or self._nid("entity", entity_b)

        neighbors_a = self._adj.get(nid_a, set())
        neighbors_b = self._adj.get(nid_b, set())

        f: dict[str, float] = {}
        f["shared_neighbors"] = len(neighbors_a & neighbors_b)
        f["jaccard"] = _jaccard(neighbors_a, neighbors_b)
        f["adamic_adar"] = _adamic_adar(self._adj, nid_a, nid_b)

        # Embedding similarity
        emb_a = self._embeddings.get(nid_a)
        emb_b = self._embeddings.get(nid_b)
        f["cosine_similarity"] = _cosine_sim(emb_a, emb_b) if emb_a and emb_b else 0.0

        # Same community?
        if self._communities:
            f["same_community"] = float(
                self._communities.get(nid_a) == self._communities.get(nid_b)
            )

        return f

    def to_feature_dicts(self) -> dict[str, dict[str, Any]]:
        """Export features as flat dicts ready for ML.

        Flattens degree_by_type into separate columns. Returns:
            {entity_id: {"degree": 5, "pagerank": 0.02, "HAS_INDUSTRY_degree": 1, ...}}

        Feed this into pandas DataFrame or directly into sklearn/xgboost.
        """
        raw = self.compute_features()
        flat: dict[str, dict[str, Any]] = {}

        for entity_id, feats in raw.items():
            row: dict[str, Any] = {}
            for k, v in feats.items():
                if k == "degree_by_type" and isinstance(v, dict):
                    for edge_type, count in v.items():
                        row[f"{edge_type}_degree"] = count
                else:
                    row[k] = v
            flat[entity_id] = row

        return flat

    # ============================================================
    # Query / Chat
    # ============================================================

    def _resolve_entity(self, identifier: str) -> str | None:
        """Resolve an entity by ID, label, or data field."""
        # Direct ID match
        nid = self._nid("entity", identifier)
        if nid in self._nodes:
            return nid
        # Search by label or data fields
        for node_id, node in self._nodes.items():
            if node["type"] != "entity":
                continue
            if node["label"].lower() == identifier.lower():
                return node_id
            for v in node.get("data", {}).values():
                if isinstance(v, str) and v.lower() == identifier.lower():
                    return node_id
        return None

    async def find_similar(
        self,
        entity_id: str,
        *,
        top_k: int = 5,
        hops: int = 2,
        edge_type_weights: dict[str, float] | None = None,
        exclude_edge_types: list[str] | None = None,
        use_idf: bool = True,
    ) -> list[dict[str, Any]]:
        """Find similar entities via graph traversal + embeddings.

        Each path source → feature_node → other_entity contributes a score
        weighted by the edge weight, the hop distance, and (when use_idf is
        True) the inverse log-degree of the intermediate feature node.
        Hub feature nodes (e.g. propensity_score=zero with 170 attached
        entities) are automatically suppressed without needing
        edge_type_weights to be set.

        Args:
            top_k: Number of similar entities to return.
            hops: Max graph distance to traverse (2 = friend-of-friend).
            edge_type_weights: Manual weight multipliers per edge type.
                e.g. {"HAS_PRODUCTS_OWNED": 3.0, "COMPETES_WITH": 5.0}.
                Stacks with use_idf when both are set.
            exclude_edge_types: Edge types to skip entirely.
                e.g. ["IN_PROPENSITY_SCORE_RANGE", "IN_DEALS_LOST_RANGE"].
            use_idf: If True (default), discount edges through hub nodes
                using 1/log(degree+1). This is the structural fix for the
                'numeric bucket noise' problem — sentinel-zero feature
                nodes connecting many entities lose their dominance
                automatically. Set False for the legacy uniform behavior.
        """
        source = self._resolve_entity(entity_id)
        if source is None:
            return []

        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)

        # Graph traversal scoring. Multi-path scoring: every distinct
        # path source→...→target_entity adds to the score (stronger
        # similarity = more shared paths). The `expanded` set only
        # blocks BFS re-expansion, NOT score accumulation.
        expanded: set[str] = {source}
        frontier = {source}
        for hop in range(hops):
            nxt: set[str] = set()
            for node in frontier:
                # IDF discount when the path passes through a hub node.
                # At hop=0, `node` is the source — no discount. At hop>=1,
                # `node` is an intermediate node we're walking through.
                idf_factor = 1.0
                if use_idf and hop > 0:
                    intermediate_degree = len(self._adj.get(node, set()))
                    idf_factor = _idf_weight(intermediate_degree)
                for e in self._typed_adj.get(node, []):
                    tgt = e["target"]
                    tgt_node = self._nodes.get(tgt, {})
                    if tgt_node.get("type") == "entity" and tgt != source:
                        etype = e.get("type", "")
                        if exclude_edge_types and etype in exclude_edge_types:
                            pass  # don't score, but still allow expansion below
                        else:
                            w = e.get("weight", 1.0) / (hop + 1)
                            if edge_type_weights and etype in edge_type_weights:
                                w *= edge_type_weights[etype]
                            w *= idf_factor
                            scores[tgt] += w
                            via = self._nodes.get(node, {})
                            if via.get("type") != "entity":
                                reasons[tgt].append(f"{e['type']}:{via.get('label', '')}")
                            else:
                                reasons[tgt].append(e["type"])
                    if tgt not in expanded:
                        expanded.add(tgt)
                        nxt.add(tgt)
            frontier = nxt

        # Embedding boost
        src_emb = self._embeddings.get(source)
        if src_emb:
            for nid, emb in self._embeddings.items():
                if nid == source or nid not in self._entity_ids:
                    continue
                sim = _cosine_sim(src_emb, emb)
                if sim > 0.5:
                    scores[nid] += sim * 0.5
                    if sim > 0.65:
                        reasons[nid].append(f"embedding_sim:{sim:.2f}")

        # Filter to only entities from original records (not relationship targets)
        entity_set = set(self._entity_ids)
        ranked = [
            (nid, sc)
            for nid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if nid in entity_set
        ][:top_k]
        return [
            {
                "entity": self._nodes[nid]["label"],
                "score": round(sc, 3),
                "via": list(set(reasons.get(nid, []))),
                "data": self._nodes[nid].get("data", {}),
                "pair_features": self.pairwise_features(entity_id, self._nodes[nid]["label"]),
            }
            for nid, sc in ranked
        ]

    async def match(self, *, top_k: int = 5, **attrs: Any) -> list[dict[str, Any]]:
        """Find entities matching partial attributes."""
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, list[str]] = defaultdict(list)

        for col, val in attrs.items():
            for item in (x.strip() for x in str(val).split(",") if x.strip()):
                fid = self._nid(col, item)
                if fid in self._adj:
                    for neighbor in self._adj[fid]:
                        if self._nodes.get(neighbor, {}).get("type") == "entity":
                            scores[neighbor] += 1.0
                            matched[neighbor].append(f"{col}={item}")

            if len(str(val)) > 50:
                embs = await self._embed([str(val)])
                if embs:
                    for nid, emb in self._embeddings.items():
                        if nid in self._entity_ids:
                            sim = _cosine_sim(embs[0], emb)
                            if sim > 0.55:
                                scores[nid] += sim
                                matched[nid].append(f"semantic:{col}")

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "entity": self._nodes[nid]["label"],
                "score": round(sc, 3),
                "matched_on": matched.get(nid, []),
                "data": self._nodes[nid].get("data", {}),
            }
            for nid, sc in ranked
        ]

    def neighborhood(self, entity_id: str, hops: int = 1) -> dict[str, Any]:
        """Get local subgraph."""
        src = self._resolve_entity(entity_id)
        if src is None:
            return {"nodes": [], "edges": []}
        visited = {src}
        frontier = {src}
        nodes = [self._nodes[src]]
        edges = []
        for _ in range(hops):
            nxt: set[str] = set()
            for n in frontier:
                for e in self._typed_adj.get(n, []):
                    t = e["target"]
                    if t not in visited:
                        visited.add(t)
                        nxt.add(t)
                        if t in self._nodes:
                            nodes.append(self._nodes[t])
                    edges.append(
                        {
                            "source": e["source"],
                            "target": t,
                            "type": e["type"],
                            "weight": e.get("weight", 1.0),
                        }
                    )
            frontier = nxt
        return {"nodes": nodes, "edges": edges}

    # ============================================================
    # Incremental updates
    # ============================================================

    async def add_records(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, int]:
        """Add new records to an existing graph.

        Same args as build_from_records. New nodes/edges are added,
        existing ones are updated. Recomputes similarity edges for
        new entities against all existing ones.

        Features are invalidated and recomputed on next access.

        Raises:
            RuntimeError: If the graph has been frozen with freeze().
                Use score_one()/score_batch() for inference instead, or
                call unfreeze() to allow mutation again.
        """
        if self._frozen:
            raise RuntimeError(
                "Cannot add_records to a frozen graph — freeze() locks the "
                "topology so adding records would break training/inference parity.\n"
                "  • For inference on new records: use kg.score_one(record) "
                "or kg.score_batch(records) (no mutation).\n"
                "  • To grow the graph: call kg.unfreeze(), add the records, "
                "then re-freeze."
            )

        # Invalidate cached features
        self._features = None
        self._communities = None

        # Just call build — _add_node and _add_edge are idempotent
        return await self.build_from_records(records, **kwargs)

    # ============================================================
    # KNN features
    # ============================================================

    def knn_features(
        self,
        k: int = 5,
        metric: str = "embedding",
    ) -> dict[str, dict[str, Any]]:
        """Compute K-nearest-neighbor features for every entity.

        For each entity, finds K nearest neighbors and returns:
        - knn_entities: list of K nearest entity IDs
        - knn_distances: list of K distances (1 - similarity)
        - knn_avg_distance: mean distance to K neighbors
        - knn_min_distance: distance to nearest neighbor
        - knn_max_distance: distance to K-th neighbor

        Args:
            k: Number of neighbors.
            metric: "embedding" (cosine) or "graph" (path-based).

        Returns:
            {entity_id: {knn features}} for all entities.
        """
        results: dict[str, dict[str, Any]] = {}

        entity_nids = [n for n in self._entity_ids if n in self._nodes]

        if metric == "embedding":
            for nid in entity_nids:
                emb = self._embeddings.get(nid)
                if not emb:
                    label = self._nodes[nid]["label"]
                    results[label] = {
                        "knn_entities": [],
                        "knn_distances": [],
                        "knn_avg_distance": 1.0,
                        "knn_min_distance": 1.0,
                        "knn_max_distance": 1.0,
                    }
                    continue

                # Compute distances to all other entities
                dists: list[tuple[str, float]] = []
                for other_nid in entity_nids:
                    if other_nid == nid:
                        continue
                    other_emb = self._embeddings.get(other_nid)
                    if other_emb:
                        sim = _cosine_sim(emb, other_emb)
                        dists.append((other_nid, 1.0 - sim))  # distance = 1 - similarity

                # Sort by distance, take top K
                dists.sort(key=lambda x: x[1])
                top_k = dists[:k]

                label = self._nodes[nid]["label"]
                knn_labels = [self._nodes[n]["label"] for n, _ in top_k]
                knn_dists = [d for _, d in top_k]

                results[label] = {
                    "knn_entities": knn_labels,
                    "knn_distances": [round(d, 4) for d in knn_dists],
                    "knn_avg_distance": round(sum(knn_dists) / len(knn_dists), 4)
                    if knn_dists
                    else 1.0,
                    "knn_min_distance": round(knn_dists[0], 4) if knn_dists else 1.0,
                    "knn_max_distance": round(knn_dists[-1], 4) if knn_dists else 1.0,
                }

        elif metric == "graph":
            # Graph-based KNN: use shared neighbor count as proximity
            for nid in entity_nids:
                neighbors = self._adj.get(nid, set())
                dists: list[tuple[str, float]] = []
                for other_nid in entity_nids:
                    if other_nid == nid:
                        continue
                    other_neighbors = self._adj.get(other_nid, set())
                    shared = len(neighbors & other_neighbors)
                    union = len(neighbors | other_neighbors)
                    dist = 1.0 - (shared / union if union else 0.0)
                    dists.append((other_nid, dist))

                dists.sort(key=lambda x: x[1])
                top_k = dists[:k]

                label = self._nodes[nid]["label"]
                knn_labels = [self._nodes[n]["label"] for n, _ in top_k]
                knn_dists = [d for _, d in top_k]

                results[label] = {
                    "knn_entities": knn_labels,
                    "knn_distances": [round(d, 4) for d in knn_dists],
                    "knn_avg_distance": round(sum(knn_dists) / len(knn_dists), 4)
                    if knn_dists
                    else 1.0,
                    "knn_min_distance": round(knn_dists[0], 4) if knn_dists else 1.0,
                    "knn_max_distance": round(knn_dists[-1], 4) if knn_dists else 1.0,
                }

        return results

    # ============================================================
    # KNN Rate features (peer adoption → propensity signals)
    # ============================================================

    def knn_rate_features(
        self,
        target_columns: list[str],
        *,
        k: int = 3,
        metric: str = "graph",
    ) -> dict[str, dict[str, Any]]:
        """Compute peer adoption rates for target columns.

        For each entity, finds K nearest neighbors (structurally similar
        accounts) and computes what fraction of those neighbors have each
        value in the target columns.

        This is the core insight: "accounts sharing entity nodes are
        structurally similar — their portfolios become the recommendation."

        Example: if 2/3 of an account's graph neighbors have CASB,
        knn_rate_casb = 0.67. This becomes a propensity signal.

        Args:
            target_columns: Column names whose values become rate features.
                e.g. ["products_owned", "use_cases"] where each is a
                comma-separated list or categorical.
            k: Number of nearest neighbors.
            metric: "graph" (Jaccard distance) or "embedding" (cosine).

        Returns:
            {entity_id: {
                "knn_rate_{value}": float,  # adoption rate among peers
                "knn_peers": [peer_ids],    # who the peers are
                "knn_peer_votes": {value: [bool per peer]},
            }}
        """
        # Track the kwargs so save_features() can detect the
        # "computed knn_rate but forgot target_columns at save" footgun.
        self._last_knn_target_columns = list(target_columns)
        self._last_knn_k = k

        knn_data = self.knn_features(k=k, metric=metric)

        # Collect all possible values across target columns
        all_values: set[str] = set()
        entity_values: dict[str, set[str]] = {}  # entity_nid → set of values

        for nid in self._entity_ids:
            node = self._nodes.get(nid, {})
            data = node.get("data", {})
            vals: set[str] = set()

            for col in target_columns:
                raw = str(data.get(col, ""))
                for item in (x.strip().lower() for x in raw.split(",") if x.strip()):
                    vals.add(item)
                    all_values.add(item)

            entity_values[nid] = vals

        # Build label → nid lookup
        label_to_nid: dict[str, str] = {}
        for nid in self._entity_ids:
            label_to_nid[self._nodes[nid]["label"]] = nid

        results: dict[str, dict[str, Any]] = {}

        for entity_label, kdata in knn_data.items():
            peer_labels = kdata.get("knn_entities", [])
            peers_nids = [label_to_nid.get(p) for p in peer_labels if label_to_nid.get(p)]
            entity_nid = label_to_nid.get(entity_label)

            rates: dict[str, float] = {}
            votes: dict[str, list[bool]] = {}

            for value in sorted(all_values):
                peer_has = [value in entity_values.get(p, set()) for p in peers_nids]
                rate = sum(peer_has) / len(peer_has) if peer_has else 0.0
                rates[f"knn_rate_{value}"] = round(rate, 4)
                votes[value] = peer_has

            # Entity's own values (for identifying gaps)
            own_values = entity_values.get(entity_nid, set()) if entity_nid else set()

            results[entity_label] = {
                **rates,
                "knn_peers": peer_labels,
                "knn_peer_votes": votes,
                "own_values": list(own_values),
            }

        return results

    def recommend(
        self,
        entity_id: str,
        target_columns: list[str],
        *,
        k: int = 3,
        min_rate: float = 0.5,
        metric: str = "graph",
        exclude_dominant: float | None = None,
    ) -> dict[str, Any]:
        """Recommend values (products, use cases) based on peer adoption.

        Args:
            exclude_dominant: If set, skip values present in more than this
                fraction of ALL entities. e.g. 0.7 skips values that 70%+
                of entities already have (too common to be a useful signal).

        Finds the entity's K nearest neighbors, checks which target
        values those peers have but the entity doesn't, and recommends
        the ones with adoption rate >= min_rate.

        This is the "their portfolios become the recommendation" logic.

        Args:
            entity_id: The account to generate recommendations for.
            target_columns: Columns to recommend from.
            k: Number of peers.
            min_rate: Minimum peer adoption rate to recommend.
            metric: "graph" or "embedding".

        Returns:
            List of recommendations sorted by peer adoption rate:
            [{value, rate, peer_votes, reason}]
        """
        rates = self.knn_rate_features(target_columns, k=k, metric=metric)

        # Find the entity
        entity_data = None
        for label, data in rates.items():
            resolved = self._resolve_entity(entity_id)
            if resolved and self._nodes.get(resolved, {}).get("label") == label:
                entity_data = data
                break
        # Fallback: try exact label match
        if entity_data is None and entity_id in rates:
            entity_data = rates[entity_id]

        if entity_data is None:
            return {
                "recommendations": [],
                "entity_found": False,
                "peers_found": 0,
                "own_values": [],
                "reason": "entity not found in graph",
            }

        own_values = set(entity_data.get("own_values", []))
        peers = entity_data.get("knn_peers", [])
        peer_votes = entity_data.get("knn_peer_votes", {})

        # Compute global prevalence to filter dominant values
        dominant_values: set[str] = set()
        if exclude_dominant is not None:
            value_counts: dict[str, int] = defaultdict(int)
            for nid in self._entity_ids:
                data = self._nodes.get(nid, {}).get("data", {})
                for col in target_columns:
                    for item in str(data.get(col, "")).lower().split(","):
                        item = item.strip()
                        if item:
                            value_counts[item] += 1
            n_total = len(self._entity_ids)
            for val, cnt in value_counts.items():
                if cnt / n_total > exclude_dominant:
                    dominant_values.add(val)

        recommendations = []
        for value, votes in peer_votes.items():
            if value in own_values:
                continue  # already has it
            if value in dominant_values:
                continue  # too common to be useful signal
            rate = sum(votes) / len(votes) if votes else 0.0
            if rate >= min_rate:
                # Build human-readable peer vote string
                vote_str = ""
                for i, (peer, has) in enumerate(zip(peers, votes)):
                    vote_str += f"{'✓' if has else '✗'}"
                recommendations.append(
                    {
                        "value": value,
                        "rate": round(rate, 2),
                        "peer_votes": vote_str,
                        "peers": peers,
                        "reason": (
                            f"{sum(votes)}/{len(votes)} structural peers have '{value}'. "
                            f"Graph found a signal a flat table cannot."
                        ),
                    }
                )

        recommendations.sort(key=lambda x: x["rate"], reverse=True)

        reason = "recommendations found"
        if not recommendations and not peer_votes:
            reason = "no peers found or no target values in graph"
        elif not recommendations and own_values:
            reason = "all peers own same values as entity — nothing new to recommend"
        elif not recommendations:
            reason = f"no values exceeded min_rate={min_rate} among peers"

        return {
            "recommendations": recommendations,
            "entity_found": True,
            "peers_found": len(peers),
            "own_values": list(own_values),
            "reason": reason,
        }

    def _format_node_label(self, nid: str) -> str:
        """Format a node ID into a human-readable label.

        Examples:
          industry:saas        → 'industry=SaaS'
          tech:aws             → 'tech=AWS'
          propensity_score_range:zero → 'propensity_score=zero'
          concept:zero trust   → 'zero trust'
          entity:nexatech      → 'NexaTech'
        """
        node = self._nodes.get(nid, {})
        ntype = str(node.get("type", ""))
        label = str(node.get("label", nid))
        if ntype == "entity":
            return label
        if ntype == "concept":
            return label
        if ntype.endswith("_range"):
            col_name = ntype.replace("_range", "")
            return f"{col_name}={label}"
        # Categorical / list-derived feature nodes
        return f"{ntype}={label}"

    def explain(
        self,
        entity_id: str,
        *,
        target_columns: list[str] | None = None,
        k: int = 5,
        min_rate: float = 0.2,
        include_range_nodes: bool = False,
    ) -> dict[str, Any]:
        """Explain WHY an entity gets its recommendations and features.

        Traces the actual graph paths that produce each signal.
        Returns a structured explanation a developer can show to end users.

        Args:
            entity_id: The entity to explain.
            target_columns: Columns to explain recommendations for.
            k: Number of peers.
            min_rate: Minimum peer adoption rate for recommendations.
            include_range_nodes: If True, include numeric range buckets
                (e.g. 'propensity_score=zero') in shared_nodes. Default
                False — these are usually noise from missing-data
                sentinels and shouldn't appear in user-facing output.

        Returns:
            {
                "entity": str,
                "community": {id, size, top_members},
                "peers": [{entity, shared_nodes, jaccard}],
                "recommendations": [{value, rate, peer_evidence: [{peer, has, shared_path}]}],
                "relationships": [{target, type}],  # entity-to-entity edges
            }
        """
        resolved = self._resolve_entity(entity_id)
        if resolved is None:
            return {"entity": entity_id, "error": "not found"}

        node = self._nodes[resolved]
        label = node["label"]
        explanation: dict[str, Any] = {"entity": label}

        # Community explanation
        features = self.compute_features()
        entity_feats = features.get(label, {})
        cid = entity_feats.get("community_id", -1)
        community_members = [
            eid for eid, f in features.items() if f.get("community_id") == cid and eid != label
        ]
        explanation["community"] = {
            "id": cid,
            "size": entity_feats.get("community_size", 0),
            "top_members": community_members[:5],
        }

        # Peer explanation (WHO are the peers and WHY)
        my_neighbors = self._adj.get(resolved, set())
        peer_details = []
        knn = self.knn_features(k=k, metric="graph")
        entity_knn = knn.get(label, {})
        for peer_label in entity_knn.get("knn_entities", []):
            peer_nid = self._resolve_entity(peer_label)
            if peer_nid is None:
                continue
            peer_neighbors = self._adj.get(peer_nid, set())
            shared = my_neighbors & peer_neighbors
            shared_labels: list[str] = []
            for n in shared:
                shared_node = self._nodes.get(n, {})
                ntype = shared_node.get("type", "")
                if ntype == "entity":
                    continue  # entity-to-entity overlap shown elsewhere
                if not shared_node.get("label", ""):
                    continue
                if not include_range_nodes and ntype.endswith("_range"):
                    continue
                shared_labels.append(self._format_node_label(n))
            union = len(my_neighbors | peer_neighbors)
            peer_details.append(
                {
                    "entity": peer_label,
                    "shared_nodes": shared_labels[:10],
                    "jaccard": round(len(shared) / union, 3) if union else 0,
                }
            )
        explanation["peers"] = peer_details

        # Recommendation explanation (WHY each value is recommended)
        if target_columns:
            rec_result = self.recommend(entity_id, target_columns, k=k, min_rate=min_rate)
            recs = (
                rec_result.get("recommendations", [])
                if isinstance(rec_result, dict)
                else rec_result
            )
            rec_explanations = []
            for rec in recs[:5]:
                # Trace which peers have this value
                peer_evidence = []
                for peer in peer_details:
                    peer_nid = self._resolve_entity(peer["entity"])
                    if peer_nid is None:
                        continue
                    peer_data = self._nodes.get(peer_nid, {}).get("data", {})
                    has_value = False
                    for col in target_columns:
                        vals = str(peer_data.get(col, "")).lower()
                        if rec["value"] in vals:
                            has_value = True
                            break
                    peer_evidence.append(
                        {
                            "peer": peer["entity"],
                            "has_value": has_value,
                            "shared_via": peer.get("shared_nodes", [])[:3],
                        }
                    )
                rec_explanations.append(
                    {
                        "value": rec["value"],
                        "rate": rec["rate"],
                        "peer_evidence": peer_evidence,
                    }
                )
            explanation["recommendations"] = rec_explanations
            explanation["recommendation_reason"] = (
                rec_result.get("reason", "") if isinstance(rec_result, dict) else ""
            )

        # Direct entity-to-entity relationships
        relationships = []
        for e in self._typed_adj.get(resolved, []):
            target_node = self._nodes.get(e["target"], {})
            if target_node.get("type") == "entity" and e["target"] != resolved:
                relationships.append(
                    {
                        "target": target_node.get("label", ""),
                        "type": e["type"],
                        "confidence": e.get("confidence", 1.0),
                    }
                )
        explanation["relationships"] = relationships

        return explanation

    def compute_feature(
        self,
        *,
        computation: str,
        node_type: str | None = None,
        target_value: str | None = None,
        reference_filter: dict[str, str] | None = None,
        k: int = 5,
        aggregation: str = "mean",
    ) -> dict[str, float]:
        """Compute a custom feature for every entity. Delegates to feature_engine."""
        from .feature_engine import compute_feature as _compute

        return _compute(
            self,
            computation=computation,
            node_type=node_type,
            target_value=target_value,
            reference_filter=reference_filter,
            k=k,
            aggregation=aggregation,
        )

    def co_occurrence_features(
        self,
        target_column: str,
        *,
        outcome_filter: str | None = None,
        min_support: int = 5,
    ) -> dict[str, dict[str, Any]]:
        """Co-occurrence with lift and outcome-aware rates.

        For each pair (A, B), computes:
        - p_ba: P(B|A) — conditional probability
        - lift: P(A∧B) / (P(A) · P(B)) — statistical lift (>1 = positive signal)
        - co_count: how many entities have both
        - co_won: how many with both have outcome matching outcome_filter
        - co_rate_won: P(B|A, outcome=filter) — outcome-conditional rate

        Args:
            target_column: Column to analyze (e.g. "products_owned").
            outcome_filter: If set, also compute outcome-conditional rates.
                e.g. "Won" computes P(B|A, outcome=Won) separately.
            min_support: Pairs with co_count < min_support are excluded
                from the output. Prevents statistically fragile lift
                scores driven by 1-2 accounts. Default 5; set to 1 for
                exhaustive output (legacy behavior).

        Returns:
            {value_a: {value_b: {p_ba, lift, co_count, co_won, co_rate_won}}}
        """
        if outcome_filter and not self._outcome_column:
            logger.warning(
                f"outcome_filter='{outcome_filter}' passed but no outcome_column "
                "was set in build_from_records(). Outcome data will be missing."
            )

        value_entities: dict[str, set[str]] = defaultdict(set)
        entity_outcomes: dict[str, str] = {}
        n_total = len(self._entity_ids)

        for nid in self._entity_ids:
            data = self._nodes.get(nid, {}).get("data", {})
            raw = str(data.get(target_column, ""))
            for item in (x.strip().lower() for x in raw.split(",") if x.strip()):
                value_entities[item].add(nid)
            if self._outcome_column:
                entity_outcomes[nid] = str(data.get(self._outcome_column, "")).lower()

        result: dict[str, dict[str, Any]] = {}
        values = sorted(value_entities.keys())

        for a in values:
            a_entities = value_entities[a]
            if not a_entities:
                continue
            p_a = len(a_entities) / n_total if n_total else 0
            result[a] = {}

            for b in values:
                if a == b:
                    continue
                b_entities = value_entities[b]
                co_entities = a_entities & b_entities
                co_count = len(co_entities)

                # Filter low-support pairs — they produce statistically
                # fragile lift values dominated by one or two accounts.
                if co_count < min_support:
                    continue

                p_b = len(b_entities) / n_total if n_total else 0
                p_ab = co_count / n_total if n_total else 0
                p_ba = co_count / len(a_entities) if a_entities else 0
                lift = p_ab / (p_a * p_b) if (p_a * p_b) > 0 else 0

                entry: dict[str, Any] = {
                    "p_ba": round(p_ba, 3),
                    "lift": round(lift, 3),
                    "co_count": co_count,
                    "support": co_count,  # alias — clearer term
                }

                # Outcome-conditional rates
                if outcome_filter and entity_outcomes:
                    co_won = sum(
                        1
                        for e in co_entities
                        if outcome_filter.lower() in entity_outcomes.get(e, "")
                    )
                    entry["co_won"] = co_won
                    entry["co_rate_won"] = round(co_won / co_count, 3) if co_count else 0

                result[a][b] = entry

            # Drop empty inner dicts
            if not result[a]:
                del result[a]

        return result

    # ============================================================
    # LLM queries over the graph
    # ============================================================

    async def ask(
        self,
        question: str,
        *,
        context_entities: int = 10,
        include_features: bool = True,
        include_connections: bool = True,
        model: str | None = None,
    ) -> str:
        """Ask an LLM a question using the graph as context.

        Gathers relevant graph context (entities, features, connections)
        and feeds it to the LLM along with the question.

        Args:
            question: Natural language question about the data.
            context_entities: Max entities to include as context.
            include_features: Include computed graph features.
            include_connections: Include edge/neighbor info.
            model: Override LLM model.

        Returns:
            LLM-generated answer grounded in graph data.
        """
        # Build context from graph
        context_parts: list[str] = []

        # Get entities sorted by pagerank (most important first)
        features = self.compute_features()
        sorted_entities = sorted(
            features.items(),
            key=lambda x: x[1].get("pagerank", 0),
            reverse=True,
        )[:context_entities]

        for entity_id, feats in sorted_entities:
            nid = self._nid("entity", entity_id)
            node = self._nodes.get(nid, {})
            data = node.get("data", {})

            parts = [f"Entity: {entity_id}"]

            # Add original data fields
            for k, v in data.items():
                if isinstance(v, str) and len(v) > 200:
                    # Use summary if available
                    summary_key = f"{k}_summary"
                    if summary_key in data:
                        parts.append(f"  {k}: {data[summary_key]}")
                    else:
                        parts.append(f"  {k}: {str(v)[:200]}...")
                else:
                    parts.append(f"  {k}: {v}")

            if include_features:
                parts.append(
                    f"  Graph features: degree={feats.get('degree', 0)}, "
                    f"community={feats.get('community_id', '?')}, "
                    f"pagerank={feats.get('pagerank', 0):.4f}, "
                    f"clustering={feats.get('clustering_coeff', 0):.2f}"
                )

            if include_connections:
                edges = self._typed_adj.get(nid, [])
                connections: list[str] = []
                for e in edges[:10]:  # limit to avoid token explosion
                    tgt = self._nodes.get(e["target"], {})
                    connections.append(f"{e['type']}→{tgt.get('label', e['target'])}")
                if connections:
                    parts.append(f"  Connections: {', '.join(connections)}")

            context_parts.append("\n".join(parts))

        graph_context = "\n\n".join(context_parts)
        stats = self.stats

        prompt = (
            f"You have access to a knowledge graph with {stats['entities']} entities, "
            f"{stats['total_edges']} edges, and {stats.get('communities', '?')} communities.\n\n"
            f"Graph data:\n\n{graph_context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the graph data above. Be specific and cite entities."
        )

        from pydantic import BaseModel

        from .structured import cf_structured

        class Answer(BaseModel):
            answer: str
            entities_referenced: list[str]
            confidence: str

        result = await cf_structured(
            prompt,
            Answer,
            model=model or self._extraction_model,
            account_id=self._account_id,
            api_key=self._api_key,
            max_tokens=4096,
        )

        return result.answer

    def to_ml_dataset(
        self,
        label_column: str,
        *,
        target_columns: list[str] | None = None,
        k: int = 5,
        include_knn_rates: bool = True,
        include_knn_distances: bool = True,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
        """Export (X, y) ready for sklearn/XGBoost.

        Combines all graph features + KNN rates into X.
        Generates binary label columns from label_column into y.

        Args:
            label_column: Column with multi-value labels (e.g. "products_owned").
            target_columns: Columns for KNN rate features.
            k: KNN neighbors.
            include_knn_rates: Add knn_rate_* features.
            include_knn_distances: Add knn distance features.

        Returns:
            (X, y) where X = {entity: {feature: value}} and
            y = {entity: {label_value: 0 or 1}}.
        """
        # Build X
        x: dict[str, dict[str, Any]] = {}
        graph_feats = self.to_feature_dicts()

        knn_rates = {}
        if include_knn_rates and target_columns:
            knn_rates = self.knn_rate_features(target_columns, k=k)

        knn_dists = {}
        if include_knn_distances:
            knn_dists = self.knn_features(k=k, metric="graph")

        for entity, feats in graph_feats.items():
            row = dict(feats)
            # Add KNN rate features
            if entity in knn_rates:
                for key, val in knn_rates[entity].items():
                    if key.startswith("knn_rate_") and isinstance(val, float):
                        row[key] = val
            # Add KNN distance features
            if entity in knn_dists:
                row["knn_avg_distance"] = knn_dists[entity].get("knn_avg_distance", 1.0)
                row["knn_min_distance"] = knn_dists[entity].get("knn_min_distance", 1.0)
            x[entity] = row

        # Build y (binary labels)
        all_labels: set[str] = set()
        entity_labels: dict[str, set[str]] = {}

        for nid in self._entity_ids:
            node = self._nodes.get(nid, {})
            data = node.get("data", {})
            raw = str(data.get(label_column, ""))
            labels = set()
            for item in raw.split(","):
                item = item.strip().lower()
                if item:
                    # Apply canonical map to collapse label variants
                    item = self._canonical_map.get(item, item)
                    labels.add(item)
            entity_labels[node["label"]] = labels
            all_labels |= labels

        y: dict[str, dict[str, int]] = {}
        for entity in graph_feats:
            owned = entity_labels.get(entity, set())
            y[entity] = {f"label_{v}": (1 if v in owned else 0) for v in sorted(all_labels)}

        return x, y

    def save_features(
        self,
        path: str,
        *,
        target_columns: list[str] | None = None,
        k: int = 5,
    ) -> None:
        """Persist the FULL feature matrix (including KNN rates + distances).

        Saves the same features that to_ml_dataset() produces, so a model
        trained on to_ml_dataset() output can be reproduced at inference.

        IMPORTANT: If you trained your model with target_columns, you MUST
        pass the same target_columns here, or the saved feature set will
        be missing the knn_rate_* columns and your inference scoring will
        silently use a smaller feature set than the training did.
        """
        import json as json_mod

        if target_columns:
            x, _ = self.to_ml_dataset("_dummy_", target_columns=target_columns, k=k)
            features = x
        else:
            # Detect whether the user previously computed knn_rate features
            # and is now saving without target_columns — that's the silent
            # bug. Warn loudly.
            if self._last_knn_target_columns:
                logger.warning(
                    "save_features() called without target_columns, but knn_rate_* "
                    f"features were computed earlier with target_columns="
                    f"{self._last_knn_target_columns} (k={self._last_knn_k}). "
                    "The saved feature set will NOT include them. If your model "
                    "was trained on knn_rate features, pass the same "
                    "target_columns=... here so the saved matrix matches."
                )
            features = self.to_feature_dicts()

        payload = {
            "version": 2,
            "snapshot_date": self._snapshot_date,
            "build_meta": self._build_meta,
            "target_columns": target_columns,
            "k": k,
            "features": features,
            "entity_count": len(features),
        }
        with open(path, "w") as f:
            json_mod.dump(payload, f, default=str)

    @classmethod
    def load_features(cls, path: str) -> dict[str, dict[str, Any]]:
        """Load persisted feature matrix.

        Returns just the {entity: {feature: value}} dict. To inspect the
        full payload (snapshot_date, build_meta, target_columns), open
        the file directly with json.load().
        """
        import json as json_mod

        with open(path) as f:
            data = json_mod.load(f)
        return data["features"]

    # ============================================================
    # Freeze / score (production inference without graph mutation)
    # ============================================================

    def freeze(
        self,
        *,
        target_columns: list[str] | None = None,
        k: int = 5,
        community_method: str = "entity_projection",
    ) -> None:
        """Freeze the graph for production scoring.

        After freezing:
          - add_records() raises an error (use score_one/score_batch instead)
          - compute_features() returns the cached feature set
          - score_one() / score_batch() use this exact topology

        This solves the 'feature drift' problem: a model trained on
        last month's graph and scoring on today's graph would silently
        see different feature values for the same entity. Freezing
        guarantees inference uses the same topology that produced the
        training features.

        Args:
            target_columns: Columns whose values become knn_rate_*
                features at scoring time. Pass the same list you passed
                to to_ml_dataset() during training.
            k: Number of nearest neighbors for KNN-rate features.
                Match training-time k.
            community_method: "entity_projection" or "bipartite".
        """
        # Compute and cache features (this also caches communities).
        self._features = None
        self.compute_features(community_method=community_method)

        # Cache target value catalog so score_one returns features in the
        # same column order/set every time.
        if target_columns:
            all_values: set[str] = set()
            for nid in self._entity_ids:
                data = self._nodes.get(nid, {}).get("data", {})
                for col in target_columns:
                    raw = str(data.get(col, ""))
                    for item in (x.strip().lower() for x in raw.split(",") if x.strip()):
                        if item not in _NULL_STRINGS:
                            all_values.add(item)
            self._frozen_target_values = sorted(all_values)
            self._frozen_target_columns = list(target_columns)
            self._frozen_k = k
        else:
            self._frozen_target_values = []
            self._frozen_target_columns = None

        self._frozen_label_to_nid = {self._nodes[n]["label"]: n for n in self._entity_ids}
        self._frozen = True
        logger.info(
            f"Graph frozen: {len(self._entity_ids)} entities, "
            f"{len(self._frozen_target_values)} target values"
        )

    def unfreeze(self) -> None:
        """Reverse freeze() — allow add_records() and feature recomputation."""
        self._frozen = False
        self._features = None
        self._communities = None

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def _record_to_feature_nodes(self, record: dict[str, Any]) -> set[str]:
        """Resolve a record's feature node IDs against the frozen graph.

        Replays the same column logic used during build_from_records,
        but only against existing feature nodes — never mutates the
        graph. Used by score_one/score_batch.
        """
        if not self._build_meta:
            raise RuntimeError(
                "Cannot resolve record to feature nodes — graph has no build "
                "metadata. This usually means freeze() was called on an empty "
                "graph. Call kg.build_from_records(...) (or kg.quick_build(...)) "
                "BEFORE kg.freeze(...)."
            )
        meta = self._build_meta
        sentinel_set = set(meta.get("sentinel_zero_columns") or [])
        nodes: set[str] = set()

        # Categorical
        for col in meta.get("categorical_columns") or []:
            val = str(record.get(col, "")).strip()
            if val and val.lower() not in _NULL_STRINGS:
                nid = self._nid(col, val)
                if nid in self._nodes:
                    nodes.add(nid)

        # List
        for col in meta.get("list_columns") or {}:
            val = str(record.get(col, ""))
            for item in (x.strip() for x in val.split(",") if x.strip()):
                if item.lower() in _NULL_STRINGS:
                    continue
                nid = self._nid(col, item)
                if nid in self._nodes:
                    nodes.add(nid)

        # Numeric (with sentinel filter)
        for col in meta.get("numeric_columns") or []:
            raw = record.get(col)
            if raw is None:
                continue
            try:
                num = float(
                    str(raw)
                    .replace("$", "")
                    .replace(",", "")
                    .replace("M", "e6")
                    .replace("B", "e9")
                    .replace("K", "e3")
                )
            except (ValueError, TypeError):
                continue
            if num == 0 and col in sentinel_set:
                continue
            bucket = _bucket_number(num)
            nid = self._nid(f"{col}_range", bucket)
            if nid in self._nodes:
                nodes.add(nid)

        return nodes

    async def score_one(
        self,
        record: dict[str, Any],
        *,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Score a new record using the frozen graph topology.

        Returns features for `record` AS IF it had been added to the graph,
        but without mutating any graph state. Same feature shape as
        to_ml_dataset() for matching training-time features.

        Args:
            record: A single row dict with the same columns used at build.
            k: Override frozen k. Default uses the k passed to freeze().

        Returns:
            {feature_name: value} including all graph features and the
            knn_rate_* features for any target_columns set in freeze().
        """
        if not self._frozen:
            raise RuntimeError(
                "score_one() requires a frozen graph so scoring-time features "
                "match training-time features (no feature drift).\n"
                "  • If you're scoring NEW records (inference): call "
                "kg.freeze(target_columns=[...], k=...) first.\n"
                "  • If you only need bulk features for already-built entities: "
                "use kg.compute_features() or kg.to_feature_dicts() instead.\n"
                "See the 'Freeze for production scoring' section of the README."
            )

        k_use = k if k is not None else self._frozen_k

        # Resolve which existing feature nodes this record connects to.
        record_features = self._record_to_feature_nodes(record)

        # Find K nearest peers via Jaccard distance over feature-only
        # neighborhoods of existing entities. (We never mutate state.)
        peer_distances: list[tuple[str, float]] = []
        for entity_nid in self._entity_ids:
            entity_neighbors = self._adj.get(entity_nid, set())
            entity_feats = {
                n for n in entity_neighbors if self._nodes.get(n, {}).get("type") != "entity"
            }
            if not entity_feats and not record_features:
                dist = 1.0
            else:
                inter = len(record_features & entity_feats)
                union = len(record_features | entity_feats)
                dist = 1.0 - (inter / union) if union else 1.0
            peer_distances.append((entity_nid, dist))
        peer_distances.sort(key=lambda x: x[1])
        top_peers = peer_distances[:k_use]
        peer_nids = [n for n, _ in top_peers]
        peer_dists = [d for _, d in top_peers]
        peer_labels = [self._nodes[n]["label"] for n in peer_nids]

        # Base structural features on the new record.
        result: dict[str, Any] = {}
        result["degree"] = len(record_features)
        result["unique_neighbors"] = len(record_features)
        result["degree_entity"] = 0
        result["degree_concept"] = 0
        # Edge type degrees from the implied edges
        meta = self._build_meta
        for col in meta.get("categorical_columns") or []:
            etype = f"HAS_{col.upper()}_degree"
            val = str(record.get(col, "")).strip()
            if val and val.lower() not in _NULL_STRINGS:
                result[etype] = result.get(etype, 0) + 1
        for col, etype_raw in (meta.get("list_columns") or {}).items():
            etype = f"{etype_raw}_degree"
            val = str(record.get(col, ""))
            count = 0
            for item in (x.strip() for x in val.split(",") if x.strip()):
                if item.lower() not in _NULL_STRINGS:
                    count += 1
            if count:
                result[etype] = count

        # KNN distance features (scoring-time topology)
        result["knn_avg_distance"] = (
            round(sum(peer_dists) / len(peer_dists), 4) if peer_dists else 1.0
        )
        result["knn_min_distance"] = round(peer_dists[0], 4) if peer_dists else 1.0
        result["knn_max_distance"] = round(peer_dists[-1], 4) if peer_dists else 1.0

        # KNN rate features (against frozen target value catalog)
        if self._frozen_target_columns:
            for value in self._frozen_target_values:
                votes: list[bool] = []
                for peer_nid in peer_nids:
                    peer_data = self._nodes.get(peer_nid, {}).get("data", {})
                    has = False
                    for col in self._frozen_target_columns:
                        raw = str(peer_data.get(col, "")).lower()
                        for item in (x.strip() for x in raw.split(",") if x.strip()):
                            if item == value:
                                has = True
                                break
                        if has:
                            break
                    votes.append(has)
                rate = sum(votes) / len(votes) if votes else 0.0
                result[f"knn_rate_{value}"] = round(rate, 4)

        # Diagnostic info (peers + dominant communities)
        peer_communities = [
            self._communities.get(p, -1) if self._communities else -1 for p in peer_nids
        ]
        # Most common community among peers — assigned community for the new record
        if peer_communities:
            cnt: dict[int, int] = defaultdict(int)
            for c in peer_communities:
                cnt[c] += 1
            assigned = max(cnt, key=lambda k: cnt[k])
            result["community_id"] = assigned
            result["community_size"] = (
                sum(1 for v in self._communities.values() if v == assigned)
                if self._communities
                else 0
            )

        result["knn_peers"] = peer_labels
        return result

    async def score_batch(
        self,
        records: list[dict[str, Any]],
        *,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score a batch of new records using the frozen graph.

        Sequential per-record scoring (cheap, no API calls). For
        thousands of records, consider running concurrent score_one
        calls externally.
        """
        results = []
        for record in records:
            results.append(await self.score_one(record, k=k))
        return results

    # ============================================================
    # Reporting
    # ============================================================

    def feature_report(self) -> dict[str, Any]:
        """Self-describing summary of what's in the graph + warnings.

        Helps users debug their build and understand which features
        are present, which are noise, and what they should worry about.

        Returns a dict with:
          - stats: high-level counts
          - features: which feature groups exist
          - warnings: build-time profiler warnings
          - bipartite_pagerank_caveat: noted if pagerank is in features
        """
        stats = self.stats
        features_present: dict[str, list[str]] = {
            "structural": [],
            "community": [],
            "centrality": [],
            "embedding": [],
            "knn_distance": [],
            "knn_rate": [],
            "by_edge_type": [],
        }
        sample = next(iter((self._features or {}).values()), {})
        for k in sample:
            if k.startswith("knn_rate_"):
                features_present["knn_rate"].append(k)
            elif k.startswith("knn_"):
                features_present["knn_distance"].append(k)
            elif k.startswith("n2v_"):
                features_present["embedding"].append(k)
            elif k in ("community_id", "community_size"):
                features_present["community"].append(k)
            elif k == "pagerank":
                features_present["centrality"].append(k)
            elif k.endswith("_degree") and k != "degree":
                features_present["by_edge_type"].append(k)
            elif k in (
                "degree",
                "degree_entity",
                "degree_concept",
                "unique_neighbors",
                "clustering_coeff",
                "similar_count",
                "avg_similarity",
                "max_similarity",
                "degree_by_type",
            ):
                features_present["structural"].append(k)
            else:
                features_present["structural"].append(k)

        report: dict[str, Any] = {
            "stats": stats,
            "snapshot_date": self._snapshot_date,
            "frozen": self._frozen,
            "build_meta": self._build_meta,
            "features": features_present,
            "warnings": list(self._build_warnings),
        }

        # Add the bipartite-pagerank caveat if applicable
        if "pagerank" in features_present.get("centrality", []):
            n_entity_nodes = len(self._entity_ids)
            n_total = len(self._nodes)
            if n_total > 0 and n_entity_nodes / n_total < 0.5:
                report["pagerank_caveat"] = (
                    "PageRank is computed on the bipartite graph and measures "
                    "feature-mediated centrality (how 'common' your feature values "
                    "are), not entity importance. Treat as low signal for ML."
                )

        return report

    # ============================================================
    # Visualization (delegated to visualization.GraphVisualizer)
    # ============================================================

    def _viz(self) -> GraphVisualizer:
        from .visualization import GraphVisualizer

        return GraphVisualizer(self)

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
        """Cytoscape.js-compliant graph spec — embeddable in any web UI.

        See pydantic_ai_cloudflare.visualization.GraphVisualizer for details.
        """
        return self._viz().to_cytoscape(
            color_by=color_by,
            max_nodes=max_nodes,
            focus=focus,
            hops=hops,
            include_node_types=include_node_types,
            exclude_node_types=exclude_node_types,
        )

    def to_d3_json(
        self,
        *,
        color_by: str | None = None,
        max_nodes: int | None = 300,
        focus: str | None = None,
        hops: int = 2,
    ) -> dict[str, Any]:
        """D3.js force-graph spec ({nodes:[...], links:[...]})."""
        return self._viz().to_d3_json(
            color_by=color_by, max_nodes=max_nodes, focus=focus, hops=hops
        )

    def to_mermaid(
        self,
        *,
        max_nodes: int | None = 50,
        focus: str | None = None,
        hops: int = 2,
        direction: str = "LR",
        color_by: str | None = None,
    ) -> str:
        """Mermaid flowchart string (great for Markdown/GitLab/Confluence)."""
        return self._viz().to_mermaid(
            max_nodes=max_nodes,
            focus=focus,
            hops=hops,
            direction=direction,
            color_by=color_by,
        )

    def to_graphml(
        self,
        path: str | None = None,
        *,
        color_by: str | None = None,
        max_nodes: int | None = None,
    ) -> str:
        """GraphML XML (for Gephi/yEd). If path is None, returns the XML string."""
        return self._viz().to_graphml(path, color_by=color_by, max_nodes=max_nodes)

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
        """Render to a self-contained interactive HTML file (Cytoscape.js).

        Args:
            path: Output file. If None, returns HTML as a string.
            title: Page title.
            color_by: "community" (default), "type", or column name.
            max_nodes: Cap on rendered nodes (default 300; safe for browsers).
            focus: Center the view on this entity's k-hop neighborhood.
            hops: Hops for focus mode.
            include_node_types / exclude_node_types: Type filters.
            layout: Cytoscape layout. "cose" (force, default), "concentric",
                "breadthfirst", "grid", "circle".

        Returns:
            The HTML string. Also writes to `path` if provided.
        """
        return self._viz().render_html(
            path,
            title=title,
            color_by=color_by,
            max_nodes=max_nodes,
            focus=focus,
            hops=hops,
            include_node_types=include_node_types,
            exclude_node_types=exclude_node_types,
            layout=layout,
        )

    def print_report(self) -> None:
        """Pretty-print feature_report() to stdout."""
        report = self.feature_report()
        stats = report["stats"]
        print("EntityGraph report")
        print(f"  Snapshot date:   {report['snapshot_date'] or 'live (no as_of)'}")
        print(f"  Frozen:          {report['frozen']}")
        print(f"  Entities:        {stats['entities']}")
        print(f"  Total nodes:     {stats['total_nodes']}")
        print(f"  Total edges:     {stats['total_edges']}")
        print(f"  Communities:     {stats['communities']}")
        print(f"  Embeddings:      {stats['embeddings']}")
        print(f"  Features/entity: {stats['features_per_entity']}")
        print()
        print("Features generated:")
        for group, feats in report["features"].items():
            if feats:
                preview = feats[:5] + (["..."] if len(feats) > 5 else [])
                print(f"  {group:<14s} ({len(feats):>3d}) {preview}")
        if report.get("pagerank_caveat"):
            print()
            print("PageRank caveat:")
            print(f"  {report['pagerank_caveat']}")
        if report["warnings"]:
            print()
            print("Warnings:")
            for w in report["warnings"]:
                print(f"  - {w}")

    @property
    def stats(self) -> dict[str, Any]:
        nt = defaultdict(int)
        for n in self._nodes.values():
            nt[n["type"]] += 1
        et = defaultdict(int)
        for e in self._edges:
            et[e["type"]] += 1
        feats = self.compute_features() if self._entity_ids else {}
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "entities": len(self._entity_ids),
            "node_types": dict(nt),
            "edge_types": dict(et),
            "embeddings": len(self._embeddings),
            "communities": len(set(self._communities.values())) if self._communities else 0,
            "features_per_entity": len(next(iter(feats.values()), {})) if feats else 0,
        }


# ============================================================
# Graph algorithms (pure Python, no deps)
# ============================================================


def _bucket_number(n: float) -> str:
    if n <= 0:
        return "zero"
    elif n < 1_000:
        return "tiny"
    elif n < 100_000:
        return "small"
    elif n < 1_000_000:
        return "medium"
    elif n < 100_000_000:
        return "large"
    elif n < 1_000_000_000:
        return "very_large"
    else:
        return "massive"


def _clustering_coefficient(adj: dict[str, set[str]], node: str) -> float:
    """Local clustering coefficient: fraction of neighbor pairs that are connected."""
    neighbors = list(adj.get(node, set()))
    k = len(neighbors)
    if k < 2:
        return 0.0
    connected = 0
    for i in range(k):
        for j in range(i + 1, k):
            if neighbors[j] in adj.get(neighbors[i], set()):
                connected += 1
    max_edges = k * (k - 1) / 2
    return connected / max_edges if max_edges else 0.0


def _pagerank(
    adj: dict[str, set[str]],
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Simplified PageRank. Pure Python."""
    nodes = list(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}
    pr = {node: 1.0 / n for node in nodes}
    for _ in range(iterations):
        new_pr: dict[str, float] = {}
        for node in nodes:
            rank = (1 - damping) / n
            for neighbor in adj.get(node, set()):
                out_degree = len(adj.get(neighbor, set()))
                if out_degree > 0:
                    rank += damping * pr[neighbor] / out_degree
            new_pr[node] = rank
        pr = new_pr
    return pr


def _louvain_communities(adj: dict[str, set[str]], resolution: float = 1.0) -> dict[str, int]:
    """Louvain community detection via networkx (much better than label prop)."""
    try:
        import networkx as nx

        G = nx.Graph()
        for node, neighbors in adj.items():
            for n in neighbors:
                G.add_edge(node, n)

        if len(G) == 0:
            return {}

        communities = nx.community.louvain_communities(G, seed=42, resolution=resolution)
        result = {}
        for cid, members in enumerate(communities):
            for m in members:
                result[m] = cid
        return result
    except ImportError:
        # Fallback to label propagation if networkx not installed
        return _label_propagation(adj)


def _node2vec_embeddings(
    adj: dict[str, set[str]],
    dimensions: int = 64,
    walk_length: int = 20,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 2.0,
    window: int = 5,
) -> dict[str, list[float]]:
    """Node2Vec embeddings via biased random walks + Word2Vec.

    Uses gensim Word2Vec on random walk sequences. Scales to 100K+ nodes
    because it's O(num_walks × num_nodes × walk_length), NOT O(n²).

    Falls back to a simpler random-projection method if gensim is unavailable.

    Args:
        adj: Undirected adjacency dict.
        dimensions: Embedding dimensionality.
        walk_length: Length of each random walk.
        num_walks: Walks per node.
        p: Return parameter (1/p = prob of returning to previous node).
        q: In-out parameter (1/q = prob of moving away).
        window: Context window for Word2Vec.

    Returns:
        {node_id: embedding_vector} for all nodes.
    """
    rng = random.Random(42)

    nodes = list(adj.keys())
    if not nodes:
        return {}

    # Generate biased random walks
    walks: list[list[str]] = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = [node]
            current = node
            prev = None

            for _ in range(walk_length - 1):
                neighbors = list(adj.get(current, set()))
                if not neighbors:
                    break

                if prev is None:
                    nxt = rng.choice(neighbors)
                else:
                    prev_neighbors = adj.get(prev, set())
                    weights = []
                    for nb in neighbors:
                        if nb == prev:
                            weights.append(1.0 / p)
                        elif nb in prev_neighbors:
                            weights.append(1.0)
                        else:
                            weights.append(1.0 / q)

                    total = sum(weights)
                    if total == 0:
                        nxt = rng.choice(neighbors)
                    else:
                        r = rng.random() * total
                        cumulative = 0.0
                        nxt = neighbors[-1]
                        for nb, w in zip(neighbors, weights):
                            cumulative += w
                            if r <= cumulative:
                                nxt = nb
                                break

                prev = current
                current = nxt
                walk.append(current)

            walks.append(walk)

    # Train Word2Vec on walks
    try:
        from gensim.models import Word2Vec

        model = Word2Vec(
            sentences=walks,
            vector_size=dimensions,
            window=window,
            min_count=1,
            sg=1,  # skip-gram
            workers=1,
            seed=42,
            epochs=5,
        )
        return {node: model.wv[node].tolist() for node in nodes if node in model.wv}
    except ImportError:
        logger.warning(
            "gensim not installed — Node2Vec embeddings disabled. Install with: pip install gensim"
        )
        return {}  # return empty, don't generate fake embeddings


def _build_canonical_lookup(alias_map: dict[str, str]) -> dict[str, str]:
    """Build a case-insensitive alias → canonical lookup.

    Input: {"ZS": "Zscaler", "zscaler inc": "Zscaler", "PAN": "Palo Alto"}
    Output: {"zs": "Zscaler", "zscaler inc": "Zscaler", "pan": "Palo Alto"}
    """
    lookup: dict[str, str] = {}
    for alias, canonical in alias_map.items():
        lookup[alias.lower().strip()] = canonical
    return lookup


# ============================================================
# Point-in-time training dataset builder
# ============================================================


async def build_temporal_dataset(
    records: list[dict[str, Any]],
    *,
    id_column: str,
    time_column: str,
    snapshot_dates: list[Any],
    label_column: str | None = None,
    label_horizon_days: int = 180,
    target_columns: list[str] | None = None,
    feature_kwargs: dict[str, Any] | None = None,
    label_value: str | None = None,
    k: int = 5,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a point-in-time training dataset with no leakage.

    For each snapshot date `T`:
      1. Build an EntityGraph using only records where time_column <= T.
      2. Compute features for every entity present at T.
      3. Build labels from records where T < time_column <= T + horizon
         (or, if label_column is None, from values present in the
         entity's data on or before T).

    The result is rows of (entity_id, snapshot_date, features) paired
    with (entity_id, snapshot_date, label). A model trained on this
    will see exactly what was knowable at each snapshot — no future
    information leaks in.

    Args:
        records: All records (full history).
        id_column: Entity ID column name.
        time_column: Event timestamp column.
        snapshot_dates: List of cutoff dates to build features at.
            e.g. ["2023-01-01", "2023-04-01", "2023-07-01"].
        label_column: Column whose value AFTER the snapshot becomes the
            label. e.g. "purchased_casb_at". If None, uses the entity's
            current value of `target_columns[0]` (less rigorous; only
            use when records are deduplicated to one-row-per-entity).
        label_horizon_days: Forward window for label (days after snapshot).
        target_columns: Columns whose values become knn_rate_* features.
        feature_kwargs: Extra args passed to build_from_records().
        label_value: If set, label = 1 if the entity has this value in
            label_column within the horizon; else 0. If None, labels
            are dicts of {value: 0|1} for every value seen.
        k: KNN k for knn_rate features.

    Returns:
        (X, y) where each is a list of dicts. Each row has:
            X[i] = {"entity": str, "snapshot_date": str, **features}
            y[i] = {"entity": str, "snapshot_date": str, "label": ...}
    """
    if not records:
        return [], []
    feature_kwargs = dict(feature_kwargs or {})

    snapshots: list[datetime] = []
    for s in snapshot_dates:
        dt = _parse_date(s)
        if dt is None:
            raise ValueError(f"Bad snapshot_date: {s!r}")
        snapshots.append(dt)

    X: list[dict[str, Any]] = []
    y: list[dict[str, Any]] = []

    for snap in snapshots:
        snap_str = snap.isoformat()
        if show_progress:
            logger.info(f"Building snapshot graph as_of={snap_str}...")

        kg = EntityGraph()
        # Force-include time and as_of regardless of caller defaults.
        snap_kwargs = dict(feature_kwargs)
        snap_kwargs.update(
            time_column=time_column,
            as_of=snap_str,
            id_column=id_column,
        )
        await kg.build_from_records(records, **snap_kwargs)

        if not kg._entity_ids:
            continue

        # Compute features (including knn_rate if target_columns given)
        features = kg.to_feature_dicts()
        if target_columns:
            x_part, _ = kg.to_ml_dataset("_dummy_", target_columns=target_columns, k=k)
            features = x_part

        # Future events for label computation
        horizon_end = snap + _timedelta_days(label_horizon_days) if label_horizon_days > 0 else None
        future_events_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if label_column:
            for r in records:
                eid = str(r.get(id_column, ""))
                if not eid:
                    continue
                event_dt = _parse_date(r.get(time_column))
                if event_dt is None or event_dt <= snap:
                    continue
                if horizon_end and event_dt > horizon_end:
                    continue
                future_events_by_entity[eid].append(r)

        for entity_label, feats in features.items():
            row_x = {
                "entity": entity_label,
                "snapshot_date": snap_str,
            }
            row_x.update(feats)
            X.append(row_x)

            # Label
            if label_column:
                events = future_events_by_entity.get(entity_label, [])
                if label_value:
                    label_v: Any = int(
                        any(
                            label_value.lower() in str(ev.get(label_column, "")).lower()
                            for ev in events
                        )
                    )
                else:
                    # collect all distinct values
                    label_v = sorted(
                        {
                            str(ev.get(label_column, "")).strip().lower()
                            for ev in events
                            if ev.get(label_column)
                        }
                    )
            else:
                # Use current entity's data (assumes one-row-per-entity input)
                resolved = kg._resolve_entity(entity_label)
                if resolved and target_columns:
                    data = kg._nodes[resolved].get("data", {})
                    label_v = {col: str(data.get(col, "")) for col in target_columns}
                else:
                    label_v = None

            y.append(
                {
                    "entity": entity_label,
                    "snapshot_date": snap_str,
                    "label": label_v,
                }
            )

    return X, y


def _timedelta_days(days: int) -> Any:  # imported locally to keep top deps minimal
    from datetime import timedelta

    return timedelta(days=days)


# Backward compatibility alias
KnowledgeGraph = EntityGraph
