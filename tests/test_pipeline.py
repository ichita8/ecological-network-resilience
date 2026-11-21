import pandas as pd
import pytest

from ecorobust.robustness import (
    build_bipartite_network,
    curve_auc,
    random_removal_curves,
    targeted_removal_curve,
    summarize_curves,
)


def sample_matrix():
    return pd.DataFrame(
        [[1, 2, 0], [0, 1, 1], [0, 0, 0]],
        index=["r1", "r2", "isolated_row"],
        columns=["c1", "c2", "isolated_column"],
    )


def test_bipartite_graph_preserves_isolated_species():
    graph = build_bipartite_network(sample_matrix())
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 4
    assert all(graph.nodes[n]["bipartite"] == 0 for n in [
        ("row", "r1"), ("row", "r2"), ("row", "isolated_row")
    ])


def test_all_edges_connect_opposite_partitions():
    graph = build_bipartite_network(sample_matrix())
    for a, b in graph.edges():
        assert graph.nodes[a]["bipartite"] != graph.nodes[b]["bipartite"]


def test_random_removal_is_reproducible():
    graph = build_bipartite_network(sample_matrix())
    a = random_removal_curves(graph, 5, 17)
    b = random_removal_curves(graph, 5, 17)
    for first, second in zip(a, b):
        pd.testing.assert_frame_equal(first, second)


def test_random_curve_has_correct_endpoints():
    graph = build_bipartite_network(sample_matrix())
    curve = random_removal_curves(graph, 1, 17)[0]
    assert curve.iloc[0]["largest_component_fraction"] > 0
    assert curve.iloc[0]["interaction_retention"] == 1
    assert curve.iloc[-1]["largest_component_fraction"] == 0
    assert curve.iloc[-1]["interaction_retention"] == 0


def test_targeted_removal_is_complete():
    graph = build_bipartite_network(sample_matrix())
    curve = targeted_removal_curve(graph)
    assert len(curve) == graph.number_of_nodes() + 1
    assert curve.iloc[-1]["largest_component_fraction"] == 0
    assert curve.iloc[-1]["interaction_retention"] == 0


def test_auc_values_are_bounded():
    graph = build_bipartite_network(sample_matrix())
    curve = targeted_removal_curve(graph)
    assert 0 <= curve_auc(curve) <= 1
    assert 0 <= curve_auc(curve, "interaction_retention") <= 1


def test_summary_contains_both_outcomes():
    graph = build_bipartite_network(sample_matrix())
    curves = random_removal_curves(graph, 3, 42)
    summary = summarize_curves(curves)
    assert len(summary) == 3
    assert "robustness_auc" in summary.columns
    assert "interaction_retention_auc" in summary.columns


def test_rejects_overlapping_partition_labels():
    matrix = pd.DataFrame([[1]], index=["same"], columns=["same"])
    with pytest.raises(ValueError, match="overlap"):
        build_bipartite_network(matrix)


def test_rejects_negative_values():
    matrix = pd.DataFrame([[-1]], index=["r"], columns=["c"])
    with pytest.raises(ValueError, match="nonnegative"):
        build_bipartite_network(matrix)

