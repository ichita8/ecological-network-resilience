from pathlib import Path
import random

import networkx as nx
import numpy as np
import pandas as pd


def build_bipartite_network(matrix):
    """Build an undirected bipartite graph from a Web of Life matrix.

    Every row and column species is retained, including isolated species.
    Positive matrix values represent present interactions.
    """
    if isinstance(matrix, (str, Path)):
        matrix = pd.read_csv(matrix, index_col=0)

    if matrix.empty or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Interaction matrix is empty.")

    matrix = matrix.copy()
    matrix.index = matrix.index.astype(str).str.strip()
    matrix.columns = matrix.columns.astype(str).str.strip()

    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("Duplicate species labels within a partition.")

    if (matrix.index == "").any() or (matrix.columns == "").any():
        raise ValueError("Blank species labels found.")

    if set(matrix.index) & set(matrix.columns):
        raise ValueError("Species labels overlap between bipartite partitions.")

    values = matrix.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Matrix must contain finite, nonnegative values.")

    graph = nx.Graph()

    row_nodes = [("row", name) for name in matrix.index]
    column_nodes = [("column", name) for name in matrix.columns]

    graph.add_nodes_from(row_nodes, bipartite=0)
    graph.add_nodes_from(column_nodes, bipartite=1)

    for i, j in zip(*np.where(values > 0)):
        graph.add_edge(row_nodes[i], column_nodes[j])

    if graph.number_of_edges() == 0:
        raise ValueError("Network contains no observed interactions.")

    graph.graph["original_species"] = graph.number_of_nodes()
    graph.graph["original_interactions"] = graph.number_of_edges()

    return graph


def _removal_curve(graph, removal_order, scenario, replicate):
    original_species = graph.number_of_nodes()
    original_edges = graph.number_of_edges()

    if original_species == 0 or original_edges == 0:
        raise ValueError("Network must contain species and interactions.")

    working = graph.copy()

    def measurements():
        if working.number_of_nodes() == 0:
            largest_component = 0
        else:
            largest_component = max(
                len(component)
                for component in nx.connected_components(working)
            )

        return (
            largest_component / original_species,
            working.number_of_edges() / original_edges,
        )

    rows = []
    lcc, retention = measurements()
    rows.append((0.0, lcc, retention))

    for index, species in enumerate(removal_order, start=1):
        working.remove_node(species)
        lcc, retention = measurements()
        rows.append((index / original_species, lcc, retention))

    result = pd.DataFrame(
        rows,
        columns=[
            "fraction_species_removed",
            "largest_component_fraction",
            "interaction_retention",
        ],
    )
    result["scenario"] = scenario
    result["replicate"] = replicate
    return result


def random_removal_curves(graph, replicates=100, seed=2026):
    """Generate reproducible random-removal curves."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1.")

    nodes = sorted(graph.nodes(), key=str)
    rng = random.Random(seed)
    curves = []

    for replicate in range(replicates):
        order = nodes.copy()
        rng.shuffle(order)
        curves.append(
            _removal_curve(graph, order, "random", replicate)
        )

    return curves


def targeted_removal_curve(graph):
    """Adaptively remove the currently highest-degree species."""
    working = graph.copy()
    order = []

    while working.number_of_nodes():
        degrees = dict(working.degree())
        chosen = min(
            degrees,
            key=lambda node: (-degrees[node], str(node)),
        )
        order.append(chosen)
        working.remove_node(chosen)

    return _removal_curve(
        graph, order, "targeted_highest_degree", 0
    )


def curve_auc(curve, metric="largest_component_fraction"):
    """Calculate normalized trapezoidal AUC over removal fraction [0, 1]."""
    if metric not in {
        "largest_component_fraction",
        "interaction_retention",
    }:
        raise ValueError(f"Unknown curve metric: {metric}")

    ordered = curve.sort_values("fraction_species_removed")
    x = ordered["fraction_species_removed"].to_numpy(dtype=float)
    y = ordered[metric].to_numpy(dtype=float)

    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Curve contains insufficient or invalid data.")

    if not np.isclose(x[0], 0.0) or not np.isclose(x[-1], 1.0):
        raise ValueError("Curve must cover the full removal interval [0, 1].")

    if np.any(np.diff(x) <= 0):
        raise ValueError("Removal fractions must be strictly increasing.")

    if np.any(y < 0) or np.any(y > 1):
        raise ValueError("Curve values must be between 0 and 1.")

    return float(np.sum(
        (y[:-1] + y[1:]) * np.diff(x) / 2
    ))


def summarize_curves(curves):
    """Calculate both robustness outcomes for every simulation replicate."""
    records = []

    for curve in curves:
        records.append({
            "scenario": curve["scenario"].iloc[0],
            "replicate": int(curve["replicate"].iloc[0]),
            "robustness_auc": curve_auc(
                curve, "largest_component_fraction"
            ),
            "interaction_retention_auc": curve_auc(
                curve, "interaction_retention"
            ),
        })

    return pd.DataFrame(records)
