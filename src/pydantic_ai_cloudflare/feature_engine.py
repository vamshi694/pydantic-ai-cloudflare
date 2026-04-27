"""Custom feature generation from text descriptions.

Users describe features in natural language. The system parses the
description into a computation plan and executes it against the graph.
One LLM call to parse, then pure graph operations for scoring.

    features = await kg.generate_feature(
        "How many tech stack items does each account share with Zero Trust customers?"
    )
    # → {"A0001": 3, "A0002": 0, "A0003": 5, ...}

Also supports explicit (non-LLM) feature generation for production:

    features = kg.compute_feature(
        computation="shared_count",
        node_type="tech_stack",
        reference_filter={"products_owned": "zero trust"},
    )
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Computation types that can be executed against the graph
COMPUTATION_TYPES = [
    "shared_count",  # count of shared nodes with reference group
    "overlap_rate",  # Jaccard overlap with reference group's nodes
    "adoption_rate",  # % of KNN peers that have a specific value
    "distance_to_nearest",  # graph distance to nearest entity in reference group
    "subgraph_degree",  # degree within a filtered subgraph
    "reference_rate",  # rate of a property in the reference group
    "co_occurrence_lift",  # statistical lift with a specific value
]


def compute_feature(
    graph: Any,
    *,
    computation: str,
    node_type: str | None = None,
    target_value: str | None = None,
    reference_filter: dict[str, str] | None = None,
    k: int = 5,
    aggregation: str = "mean",
) -> dict[str, float]:
    """Compute a custom feature for every entity in the graph.

    This is the explicit (non-LLM) API. For text-based generation,
    use kg.generate_feature().

    Args:
        graph: A KnowledgeGraph instance.
        computation: One of the COMPUTATION_TYPES.
        node_type: The node type to compute over (e.g. "tech_stack").
        target_value: Specific value to target (e.g. "zero trust").
        reference_filter: Filter to define the reference group.
            e.g. {"products_owned": "zero trust"} means "entities that
            have 'zero trust' in their products_owned column."
            e.g. {"stage": "Churned"} means "entities with stage=Churned."
        k: Number of neighbors for KNN-based computations.
        aggregation: How to aggregate multiple values ("mean", "sum", "max", "count").

    Returns:
        {entity_label: float} for every entity in the graph.
    """
    # Resolve reference group
    ref_entities = _resolve_reference_group(graph, reference_filter or {})

    if computation == "shared_count":
        return _shared_count(graph, ref_entities, node_type)

    elif computation == "overlap_rate":
        return _overlap_rate(graph, ref_entities, node_type)

    elif computation == "adoption_rate":
        return _adoption_rate(graph, target_value or "", k)

    elif computation == "distance_to_nearest":
        return _distance_to_nearest(graph, ref_entities)

    elif computation == "subgraph_degree":
        return _subgraph_degree(graph, ref_entities, node_type)

    elif computation == "reference_rate":
        return _reference_rate(graph, ref_entities, node_type, target_value)

    elif computation == "co_occurrence_lift":
        return _co_occurrence_lift(graph, node_type or "", target_value or "")

    else:
        raise ValueError(f"Unknown computation: {computation}. Options: {COMPUTATION_TYPES}")


async def generate_feature_from_text(
    graph: Any,
    description: str,
    *,
    model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    account_id: str | None = None,
    api_key: str | None = None,
) -> dict[str, float]:
    """Generate a feature from a natural language description.

    Uses one LLM call to parse the description into a computation
    spec, then executes it against the graph (no LLM per row).

    Args:
        graph: A KnowledgeGraph instance.
        description: Natural language feature description.
        model: Workers AI model for parsing.

    Returns:
        {entity_label: float} for every entity.
    """
    from pydantic import BaseModel

    from .structured import cf_structured

    # Build context about the graph for the LLM
    node_types = list(graph.stats.get("node_types", {}).keys())
    edge_types = list(graph.stats.get("edge_types", {}).keys())

    # Get sample data values for context
    sample_values: dict[str, list[str]] = {}
    for nid, node in list(graph._nodes.items())[:500]:
        ntype = node.get("type", "")
        if ntype != "entity" and ntype not in sample_values:
            sample_values[ntype] = []
        if ntype != "entity" and len(sample_values.get(ntype, [])) < 5:
            sample_values[ntype].append(node.get("label", ""))

    class FeatureSpec(BaseModel):
        feature_name: str
        computation: str  # one of COMPUTATION_TYPES
        node_type: str  # which node type to operate on
        target_value: str  # specific value if needed, empty string if not
        reference_filter_column: str  # column name for filter, empty if not needed
        reference_filter_value: str  # value for filter, empty if not needed

    spec = await cf_structured(
        f"Parse this feature description into a computation spec.\n\n"
        f"Description: {description}\n\n"
        f"Available node types in the graph: {node_types}\n"
        f"Available edge types: {edge_types}\n"
        f"Sample values per type: {sample_values}\n\n"
        f"Computation types available:\n"
        f"- shared_count: count shared nodes between entity and reference group\n"
        f"- overlap_rate: Jaccard overlap with reference group\n"
        f"- adoption_rate: % of KNN peers that have a value\n"
        f"- distance_to_nearest: graph hops to nearest entity in reference group\n"
        f"- subgraph_degree: degree within filtered subgraph\n"
        f"- reference_rate: rate of a property in the reference group\n"
        f"- co_occurrence_lift: statistical lift with a value\n",
        FeatureSpec,
        model=model,
        account_id=account_id or graph._account_id,
        api_key=api_key or graph._api_key,
        max_tokens=512,
    )

    ref_filter = {}
    if spec.reference_filter_column and spec.reference_filter_value:
        ref_filter = {spec.reference_filter_column: spec.reference_filter_value}

    return compute_feature(
        graph,
        computation=spec.computation,
        node_type=spec.node_type,
        target_value=spec.target_value,
        reference_filter=ref_filter,
    )


# ============================================================
# Internal computation functions
# ============================================================


def _resolve_reference_group(
    graph: Any,
    filters: dict[str, str],
) -> set[str]:
    """Find entity node IDs matching the filter criteria."""
    if not filters:
        return set(graph._entity_ids)

    matched: set[str] = set()
    for nid in graph._entity_ids:
        node = graph._nodes.get(nid, {})
        data = node.get("data", {})
        match = True
        for col, val in filters.items():
            data_val = str(data.get(col, "")).lower()
            if val.lower() not in data_val:
                match = False
                break
        if match:
            matched.add(nid)

    return matched


def _get_entity_neighbors_by_type(
    graph: Any,
    entity_nid: str,
    node_type: str | None,
) -> set[str]:
    """Get an entity's neighbor nodes, optionally filtered by type."""
    neighbors = set()
    for edge in graph._typed_adj.get(entity_nid, []):
        target = edge["target"]
        target_node = graph._nodes.get(target, {})
        if node_type is None or target_node.get("type") == node_type:
            neighbors.add(target)
    return neighbors


def _shared_count(
    graph: Any,
    ref_entities: set[str],
    node_type: str | None,
) -> dict[str, float]:
    """Count shared nodes between each entity and the reference group."""
    # Build the reference group's collective node set
    ref_nodes: set[str] = set()
    for ref_nid in ref_entities:
        ref_nodes |= _get_entity_neighbors_by_type(graph, ref_nid, node_type)

    result: dict[str, float] = {}
    for nid in graph._entity_ids:
        label = graph._nodes[nid]["label"]
        if nid in ref_entities:
            result[label] = 0.0  # don't count against yourself
            continue
        my_nodes = _get_entity_neighbors_by_type(graph, nid, node_type)
        result[label] = float(len(my_nodes & ref_nodes))

    return result


def _overlap_rate(
    graph: Any,
    ref_entities: set[str],
    node_type: str | None,
) -> dict[str, float]:
    """Jaccard overlap between each entity and the reference group's nodes."""
    ref_nodes: set[str] = set()
    for ref_nid in ref_entities:
        ref_nodes |= _get_entity_neighbors_by_type(graph, ref_nid, node_type)

    result: dict[str, float] = {}
    for nid in graph._entity_ids:
        label = graph._nodes[nid]["label"]
        my_nodes = _get_entity_neighbors_by_type(graph, nid, node_type)
        union = len(my_nodes | ref_nodes)
        result[label] = len(my_nodes & ref_nodes) / union if union > 0 else 0.0

    return result


def _adoption_rate(
    graph: Any,
    target_value: str,
    k: int,
) -> dict[str, float]:
    """KNN peer adoption rate for a specific value."""
    rates = graph.knn_rate_features([_guess_column_for_value(graph, target_value)], k=k)
    key = f"knn_rate_{target_value.lower().strip()}"
    result: dict[str, float] = {}
    for label, data in rates.items():
        result[label] = data.get(key, 0.0)
    return result


def _distance_to_nearest(
    graph: Any,
    ref_entities: set[str],
) -> dict[str, float]:
    """BFS distance to nearest entity in the reference group."""
    result: dict[str, float] = {}
    max_dist = 10  # cap

    for nid in graph._entity_ids:
        label = graph._nodes[nid]["label"]
        if nid in ref_entities:
            result[label] = 0.0
            continue

        # BFS
        visited = {nid}
        frontier = {nid}
        found = False
        for dist in range(1, max_dist + 1):
            nxt: set[str] = set()
            for node in frontier:
                for neighbor in graph._adj.get(node, set()):
                    if neighbor in ref_entities:
                        result[label] = float(dist)
                        found = True
                        break
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nxt.add(neighbor)
                if found:
                    break
            if found:
                break
            frontier = nxt
            if not frontier:
                break

        if not found:
            result[label] = float(max_dist)

    return result


def _subgraph_degree(
    graph: Any,
    ref_entities: set[str],
    node_type: str | None,
) -> dict[str, float]:
    """Degree of each entity within a filtered subgraph."""
    # Build the subgraph: only edges between ref_entities and their typed neighbors
    ref_nodes: set[str] = set(ref_entities)
    for ref_nid in ref_entities:
        ref_nodes |= _get_entity_neighbors_by_type(graph, ref_nid, node_type)

    result: dict[str, float] = {}
    for nid in graph._entity_ids:
        label = graph._nodes[nid]["label"]
        my_neighbors = graph._adj.get(nid, set())
        result[label] = float(len(my_neighbors & ref_nodes))

    return result


def _reference_rate(
    graph: Any,
    ref_entities: set[str],
    node_type: str | None,
    target_value: str | None,
) -> dict[str, float]:
    """Rate of a property within the reference group, applied to all entities."""
    if not ref_entities or not target_value:
        return {graph._nodes[nid]["label"]: 0.0 for nid in graph._entity_ids}

    # Count how many in ref group have the target value
    target_nid = graph._nid(node_type or "", target_value)
    count_with = 0
    for ref_nid in ref_entities:
        if target_nid in graph._adj.get(ref_nid, set()):
            count_with += 1

    rate = count_with / len(ref_entities) if ref_entities else 0.0

    # Return same rate for all entities (it's a reference group stat)
    # But modulate by each entity's own proximity to the reference group
    result: dict[str, float] = {}
    ref_nodes: set[str] = set()
    for ref_nid in ref_entities:
        ref_nodes |= graph._adj.get(ref_nid, set())

    for nid in graph._entity_ids:
        label = graph._nodes[nid]["label"]
        my_nodes = graph._adj.get(nid, set())
        proximity = len(my_nodes & ref_nodes) / max(len(my_nodes | ref_nodes), 1)
        result[label] = round(rate * (0.5 + 0.5 * proximity), 4)

    return result


def _co_occurrence_lift(
    graph: Any,
    node_type: str,
    target_value: str,
) -> dict[str, float]:
    """Co-occurrence lift between each entity's values and a target value."""
    target_nid = graph._nid(node_type, target_value)
    if target_nid not in graph._nodes:
        return {graph._nodes[nid]["label"]: 0.0 for nid in graph._entity_ids}

    # Entities that have the target value
    target_entities = set()
    for nid in graph._entity_ids:
        if target_nid in graph._adj.get(nid, set()):
            target_entities.add(nid)

    p_target = len(target_entities) / max(len(graph._entity_ids), 1)

    result: dict[str, float] = {}
    for nid in graph._entity_ids:
        label = graph._nodes[nid]["label"]
        # For each of this entity's neighbor nodes, compute lift with target
        max_lift = 0.0
        for neighbor in graph._adj.get(nid, set()):
            neighbor_node = graph._nodes.get(neighbor, {})
            if neighbor_node.get("type") == "entity":
                continue
            # Entities that have this neighbor
            neighbor_entities = set()
            for en in graph._entity_ids:
                if neighbor in graph._adj.get(en, set()):
                    neighbor_entities.add(en)

            p_neighbor = len(neighbor_entities) / max(len(graph._entity_ids), 1)
            p_both = len(neighbor_entities & target_entities) / max(len(graph._entity_ids), 1)

            if p_neighbor > 0 and p_target > 0:
                lift = p_both / (p_neighbor * p_target)
                max_lift = max(max_lift, lift)

        result[label] = round(max_lift, 4)

    return result


def _guess_column_for_value(graph: Any, value: str) -> str:
    """Try to guess which column a value belongs to."""
    val_lower = value.lower().strip()
    # Check if any node matches
    for nid, node in graph._nodes.items():
        if node.get("label", "").lower() == val_lower and node.get("type") != "entity":
            return node["type"]
    return "products_owned"  # fallback
