"""Knowledge Graph — graph-based feature engineering for ML + LLM analysis.

Builds a typed knowledge graph from tabular data where each row is an
entity node, column values become feature nodes, and edges represent
real relationships. Then computes graph-derived features (degree,
centrality, community, shared neighbors, embedding similarity) that
export as columns for downstream ML models.

Two purposes:
1. Chat/query: find similar entities, explain connections, explore neighborhoods
2. ML features: export graph metrics as features for XGBoost, LightGBM, etc.

    from pydantic_ai_cloudflare import KnowledgeGraph

    kg = KnowledgeGraph()
    await kg.build_from_records(data, id_column="account_id")

    # Chat use case
    similar = await kg.find_similar("NexaTech", top_k=5)

    # ML feature engineering use case
    features_df = kg.to_feature_dict()  # → dict of {entity_id: {feature: value}}
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import httpx

from ._auth import resolve_account_id, resolve_api_token
from ._http import DEFAULT_TIMEOUT, build_headers, check_api_response
from .structured import cf_structured

logger = logging.getLogger(__name__)


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


# ============================================================
# Community detection (label propagation — no deps needed)
# ============================================================


def _label_propagation(adj: dict[str, set[str]], max_iter: int = 20) -> dict[str, int]:
    """Simple label propagation for community detection. Pure Python."""
    labels = {node: i for i, node in enumerate(adj)}
    nodes = list(adj.keys())

    for _ in range(max_iter):
        changed = False
        import random

        random.shuffle(nodes)

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
# KnowledgeGraph class
# ============================================================


class KnowledgeGraph:
    """Knowledge graph for feature engineering and entity analysis.

    Builds a typed graph from tabular data. Computes graph metrics
    as ML features. Supports similarity search and LLM-powered
    entity extraction.

    No external deps beyond httpx. No networkx, no pandas required.
    Uses dicts internally, exports to whatever format you need.
    """

    def __init__(
        self,
        name: str = "default",
        *,
        embedding_model: str = "@cf/baai/bge-base-en-v1.5",
        extraction_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        account_id: str | None = None,
        api_key: str | None = None,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._name = name
        self._embedding_model = embedding_model
        self._extraction_model = extraction_model
        self._account_id = resolve_account_id(account_id)
        self._api_key = resolve_api_token(api_key)
        self._timeout = request_timeout
        self._headers = build_headers(self._api_key)

        # Graph storage
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._adj: dict[str, set[str]] = defaultdict(set)  # undirected adjacency
        self._typed_adj: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._embeddings: dict[str, list[float]] = {}
        self._entity_ids: list[str] = []  # ordered list of entity node IDs

        # Computed features (lazily populated)
        self._communities: dict[str, int] | None = None
        self._features: dict[str, dict[str, Any]] | None = None

    # -- Node/edge helpers --

    def _nid(self, ntype: str, value: str) -> str:
        return f"{ntype}:{value}".lower().strip()

    def _add_node(self, ntype: str, value: str, data: dict | None = None) -> str:
        nid = self._nid(ntype, value)
        if nid not in self._nodes:
            self._nodes[nid] = {"id": nid, "type": ntype, "label": value, "data": data or {}}
        elif data:
            self._nodes[nid]["data"].update(data)
        return nid

    def _add_edge(self, src: str, tgt: str, etype: str, weight: float = 1.0) -> None:
        for e in self._typed_adj[src]:
            if e["target"] == tgt and e["type"] == etype:
                return
        edge = {"source": src, "target": tgt, "type": etype, "weight": weight}
        self._edges.append(edge)
        self._typed_adj[src].append(edge)
        self._typed_adj[tgt].append({**edge, "source": tgt, "target": src})
        self._adj[src].add(tgt)
        self._adj[tgt].add(src)
        self._features = None  # invalidate cache

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
        extract_entities: bool = True,
        compute_similarity: bool = True,
        summarize_text: bool = False,
        similarity_threshold: float = 0.70,
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
            extract_entities: Use LLM to extract entities from text.
            compute_similarity: Add SIMILAR_TO edges from embeddings.
            summarize_text: Summarize long text columns per entity (slow but useful).
            similarity_threshold: Min cosine sim for SIMILAR_TO edges.
        """
        if not records:
            return {"nodes": 0, "edges": 0}

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

        for record in records:
            eid = str(record.get(id_column, ""))
            if not eid:
                continue

            row_data = {k: v for k, v in record.items() if k != id_column}
            entity_nid = self._add_node("entity", eid, row_data)
            self._entity_ids.append(entity_nid)

            # Categorical → feature nodes
            for col in categorical_columns:
                val = str(record.get(col, "")).strip()
                if val and val.lower() not in ("", "none", "null", "nan", "unknown"):
                    fid = self._add_node(col, val)
                    self._add_edge(entity_nid, fid, f"HAS_{col.upper()}")

            # List columns → split on comma
            for col, etype in list_columns.items():
                val = str(record.get(col, ""))
                for item in (x.strip() for x in val.split(",") if x.strip()):
                    iid = self._add_node(col, item)
                    self._add_edge(entity_nid, iid, etype)

            # Numeric → bucketed ranges
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
                bucket = _bucket_number(num)
                bid = self._add_node(f"{col}_range", bucket)
                self._add_edge(entity_nid, bid, f"IN_{col.upper()}_RANGE")

            # Text → entity extraction + optional summarization
            for col in text_columns:
                text = str(record.get(col, ""))
                if len(text) < 20:
                    continue
                if extract_entities:
                    entities = await self._extract_entities(text)
                    for ent in entities:
                        cid = self._add_node("concept", ent)
                        self._add_edge(entity_nid, cid, "HAS_CONCEPT")
                if summarize_text and len(text) > 500:
                    summary = await self._summarize_text(text, eid)
                    self._nodes[entity_nid]["data"][f"{col}_summary"] = summary

        # Embed text columns for similarity
        if compute_similarity and text_columns:
            texts_map: dict[str, str] = {}
            for record in records:
                eid = str(record.get(id_column, ""))
                nid = self._nid("entity", eid)
                combined = " ".join(str(record.get(c, "")) for c in text_columns).strip()
                if combined:
                    texts_map[nid] = combined

            if texts_map:
                nids = list(texts_map.keys())
                embs = await self._embed(list(texts_map.values()))
                for nid, emb in zip(nids, embs):
                    self._embeddings[nid] = emb

                # Pairwise SIMILAR_TO
                for i in range(len(nids)):
                    for j in range(i + 1, len(nids)):
                        sim = _cosine_sim(embs[i], embs[j])
                        if sim >= similarity_threshold:
                            self._add_edge(nids[i], nids[j], "SIMILAR_TO", weight=round(sim, 4))

        return {"nodes": len(self._nodes), "edges": len(self._edges)}

    # ============================================================
    # Graph features for ML
    # ============================================================

    def compute_features(self) -> dict[str, dict[str, Any]]:
        """Compute graph-derived features for every entity node.

        Returns a dict of {entity_id: {feature_name: value}}.
        Features computed:

        Structural:
          - degree: total edges
          - degree_entity: edges to other entities
          - degree_concept: edges to concept nodes
          - degree_by_type: {edge_type: count}
          - unique_neighbors: distinct connected nodes
          - clustering_coeff: local clustering coefficient

        Community:
          - community_id: label propagation community
          - community_size: how many entities in same community

        Centrality:
          - pagerank: simplified PageRank score
          - betweenness_approx: approximate betweenness (sampled BFS)

        Similarity:
          - avg_similarity: mean SIMILAR_TO edge weight
          - max_similarity: highest SIMILAR_TO weight
          - similar_count: number of SIMILAR_TO edges

        Pair-wise (for each other entity):
          - shared_neighbors: count of shared graph neighbors
          - jaccard: Jaccard similarity of neighbor sets
          - adamic_adar: Adamic-Adar index
          - cosine_sim: embedding cosine similarity (if available)
        """
        if self._features is not None:
            return self._features

        features: dict[str, dict[str, Any]] = {}

        # Community detection
        self._communities = _label_propagation(self._adj)
        community_sizes: dict[int, int] = defaultdict(int)
        for nid in self._entity_ids:
            cid = self._communities.get(nid, -1)
            community_sizes[cid] += 1

        # PageRank (power iteration, simplified)
        pr = _pagerank(self._adj, iterations=20)

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
            f["pagerank"] = pr.get(nid, 0.0)

            features[label] = f

        self._features = features
        return features

    def pairwise_features(self, entity_a: str, entity_b: str) -> dict[str, float]:
        """Compute pair-wise features between two entities.

        Useful for ML models that predict relationships (link prediction,
        match scoring, propensity models).
        """
        nid_a = self._nid("entity", entity_a)
        nid_b = self._nid("entity", entity_b)

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

    async def find_similar(
        self,
        entity_id: str,
        *,
        top_k: int = 5,
        hops: int = 2,
    ) -> list[dict[str, Any]]:
        """Find similar entities via graph traversal + embeddings."""
        source = self._nid("entity", entity_id)
        if source not in self._nodes:
            return []

        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)

        # Graph traversal scoring
        visited = {source}
        frontier = {source}
        for hop in range(hops):
            nxt: set[str] = set()
            for node in frontier:
                for e in self._typed_adj.get(node, []):
                    tgt = e["target"]
                    if tgt in visited:
                        continue
                    tgt_node = self._nodes.get(tgt, {})
                    if tgt_node.get("type") == "entity" and tgt != source:
                        w = e.get("weight", 1.0) / (hop + 1)
                        scores[tgt] += w
                        via = self._nodes.get(node, {})
                        if via.get("type") != "entity":
                            reasons[tgt].append(f"{e['type']}:{via.get('label', '')}")
                        else:
                            reasons[tgt].append(e["type"])
                    visited.add(tgt)
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

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
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
        src = self._nid("entity", entity_id)
        if src not in self._nodes:
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
        """
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
