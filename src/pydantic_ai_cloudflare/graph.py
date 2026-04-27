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
        self._entity_ids: list[str] = []
        self._outcome_column: str | None = None  # set during build

        # Computed features (lazily populated)
        self._communities: dict[str, int] | None = None
        self._features: dict[str, dict[str, Any]] | None = None

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
    ) -> dict[str, str]:
        """Use LLM to auto-detect and merge entity aliases.

        Scans the specified columns, collects all unique values,
        asks the LLM to group them into canonical forms, and
        updates the internal canonical map.

        Args:
            records: The dataset (same records you'll pass to build_from_records).
            columns: Which columns to scan for aliases (e.g. ["tech_stack", "competitors"]).
            model: Override LLM model.

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
        temporal_column: str | None = None,
        temporal_decay: float = 0.0,
        outcome_column: str | None = None,
        confidence_map: dict[str, float] | None = None,
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
            temporal_column: Date column for temporal decay (e.g. "created_date").
            temporal_decay: Decay rate λ. Edge weight = exp(-λ × days_ago).
                0 = no decay (default), 0.01 = gradual, 0.1 = aggressive.
        """
        if not records:
            return {"nodes": 0, "edges": 0}

        self._outcome_column = outcome_column
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

        for record in records:
            eid = str(record.get(id_column, ""))
            if not eid:
                continue

            row_data = {k: v for k, v in record.items() if k != id_column}
            entity_nid = self._add_node("entity", eid, row_data)
            self._entity_ids.append(entity_nid)

            # Compute temporal decay weight for this record
            edge_weight = 1.0
            if temporal_column and temporal_decay > 0:
                date_val = record.get(temporal_column)
                if date_val:
                    try:
                        from datetime import datetime

                        if isinstance(date_val, str):
                            # Parse common date formats
                            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
                                try:
                                    dt = datetime.strptime(date_val, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                dt = None
                        else:
                            dt = None

                        if dt:
                            now = datetime.now()
                            days_ago = (now - dt).days
                            edge_weight = math.exp(-temporal_decay * max(days_ago, 0))
                    except Exception:
                        pass

            # Resolve outcome for this record
            record_outcome = str(record.get(outcome_column, "")) if outcome_column else None

            # Categorical → feature nodes
            for col in categorical_columns:
                val = str(record.get(col, "")).strip()
                if val and val.lower() not in ("", "none", "null", "nan", "unknown"):
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
                    iid = self._add_node(col, item)
                    self._add_edge(
                        entity_nid,
                        iid,
                        etype,
                        weight=edge_weight,
                        confidence=conf,
                        outcome=record_outcome,
                    )

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
                self._add_edge(entity_nid, bid, f"IN_{col.upper()}_RANGE", weight=edge_weight)

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

        # Community detection (Louvain if networkx available, else label prop)
        self._communities = _louvain_communities(self._adj)
        community_sizes: dict[int, int] = defaultdict(int)
        for nid in self._entity_ids:
            cid = self._communities.get(nid, -1)
            community_sizes[cid] += 1

        # PageRank
        pr = _pagerank(self._adj, iterations=20)

        # Node2Vec graph embeddings via random walks + Word2Vec
        # Scales to 100K+ nodes (O(walks × length), not O(n²))
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
    ) -> list[dict[str, Any]]:
        """Find similar entities via graph traversal + embeddings."""
        source = self._resolve_entity(entity_id)
        if source is None:
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
    ) -> list[dict[str, Any]]:
        """Recommend values (products, use cases) based on peer adoption.

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
            return []

        own_values = set(entity_data.get("own_values", []))
        peers = entity_data.get("knn_peers", [])
        peer_votes = entity_data.get("knn_peer_votes", {})

        recommendations = []
        for value, votes in peer_votes.items():
            if value in own_values:
                continue  # already has it
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
        return recommendations

    def co_occurrence_features(
        self,
        target_column: str,
        *,
        outcome_filter: str | None = None,
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

        Returns:
            {value_a: {value_b: {p_ba, lift, co_count, co_won, co_rate_won}}}
        """
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
                p_b = len(b_entities) / n_total if n_total else 0
                p_ab = co_count / n_total if n_total else 0
                p_ba = co_count / len(a_entities) if a_entities else 0
                lift = p_ab / (p_a * p_b) if (p_a * p_b) > 0 else 0

                entry: dict[str, Any] = {
                    "p_ba": round(p_ba, 3),
                    "lift": round(lift, 3),
                    "co_count": co_count,
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
            labels = {item.strip().lower() for item in raw.split(",") if item.strip()}
            entity_labels[node["label"]] = labels
            all_labels |= labels

        y: dict[str, dict[str, int]] = {}
        for entity in graph_feats:
            owned = entity_labels.get(entity, set())
            y[entity] = {f"label_{v}": (1 if v in owned else 0) for v in sorted(all_labels)}

        return x, y

    def save_features(self, path: str) -> None:
        """Persist feature matrix to JSON for reproducible inference."""
        import json as json_mod

        features = self.to_feature_dicts()
        with open(path, "w") as f:
            json_mod.dump({"features": features, "entity_count": len(features)}, f)

    @classmethod
    def load_features(cls, path: str) -> dict[str, dict[str, Any]]:
        """Load persisted feature matrix."""
        import json as json_mod

        with open(path) as f:
            data = json_mod.load(f)
        return data["features"]

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


def _louvain_communities(adj: dict[str, set[str]]) -> dict[str, int]:
    """Louvain community detection via networkx (much better than label prop)."""
    try:
        import networkx as nx

        G = nx.Graph()
        for node, neighbors in adj.items():
            for n in neighbors:
                G.add_edge(node, n)

        if len(G) == 0:
            return {}

        communities = nx.community.louvain_communities(G, seed=42)
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
    import random as _random

    _random.seed(42)

    nodes = list(adj.keys())
    if not nodes:
        return {}

    # Generate biased random walks
    walks: list[list[str]] = []
    for _ in range(num_walks):
        _random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            current = node
            prev = None

            for _ in range(walk_length - 1):
                neighbors = list(adj.get(current, set()))
                if not neighbors:
                    break

                if prev is None:
                    nxt = _random.choice(neighbors)
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
                        nxt = _random.choice(neighbors)
                    else:
                        r = _random.random() * total
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
        # Fallback: simple hash-based embedding (much worse but no deps)
        logger.warning("gensim not installed — using hash-based Node2Vec fallback")
        embeddings: dict[str, list[float]] = {}
        for node in nodes:
            # Deterministic pseudo-embedding from walk co-occurrence counts
            import hashlib

            h = hashlib.sha256(node.encode()).digest()
            emb = [((b % 200) - 100) / 100.0 for b in h[:dimensions]]
            norm = math.sqrt(sum(x * x for x in emb)) + 1e-10
            embeddings[node] = [x / norm for x in emb]
        return embeddings


def _build_canonical_lookup(alias_map: dict[str, str]) -> dict[str, str]:
    """Build a case-insensitive alias → canonical lookup.

    Input: {"ZS": "Zscaler", "zscaler inc": "Zscaler", "PAN": "Palo Alto"}
    Output: {"zs": "Zscaler", "zscaler inc": "Zscaler", "pan": "Palo Alto"}
    """
    lookup: dict[str, str] = {}
    for alias, canonical in alias_map.items():
        lookup[alias.lower().strip()] = canonical
    return lookup
