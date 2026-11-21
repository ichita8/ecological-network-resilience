from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from ecorobust.robustness import (
    build_bipartite_network,
    random_removal_curves,
    targeted_removal_curve,
    summarize_curves,
)

ROOT = Path(".")
CACHE = ROOT / "data/web_of_life/full_cache"
INPUT = ROOT / "results/web_of_life/validated_networks.csv"
DEFAULT_OUTPUT = ROOT / "results/web_of_life/experiment_10000"
BASE_SEED = 20261115
GRID = np.linspace(0, 1, 101)
METRICS = [
    "largest_component_fraction",
    "interaction_retention",
]


def stable_seed(network_id):
    digest = hashlib.sha256(network_id.encode("utf-8")).hexdigest()
    return (BASE_SEED + int(digest[:8], 16)) % (2**32)


def update_curve_statistics(curve, count, means, m2):
    x = curve["fraction_species_removed"].to_numpy()

    for metric in METRICS:
        values = np.interp(
            GRID,
            x,
            curve[metric].to_numpy(),
        )

        count[metric] += 1
        delta = values - means[metric]
        means[metric] += delta / count[metric]
        m2[metric] += delta * (values - means[metric])


def curve_statistic_records(network_id, scenario, count, means, m2):
    records = []

    for metric in METRICS:
        n = count[metric]
        sd = (
            np.sqrt(m2[metric] / (n - 1))
            if n > 1
            else np.zeros_like(means[metric])
        )

        for fraction, mean_value, sd_value in zip(
            GRID, means[metric], sd
        ):
            records.append({
                "network": network_id,
                "scenario": scenario,
                "fraction_species_removed": float(fraction),
                "metric": metric,
                "mean_value": float(mean_value),
                "sd_value": float(sd_value),
                "curves_averaged": int(n),
            })

    return records


def append_csv(records, path):
    if not records:
        return

    frame = pd.DataFrame(records)
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.replicates < 1 or args.batch_size < 1:
        raise ValueError("Replicates and batch size must be positive.")

    start = time.time()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(INPUT)
    catalog = catalog[catalog["status"] == "ACCEPTED"].copy()
    catalog = catalog.sort_values("network").reset_index(drop=True)

    if args.limit is not None:
        catalog = catalog.head(args.limit).copy()

    network_ids = [str(value) for value in catalog["network"]]
    checkpoint_path = output / "checkpoint.json"

    config = {
        "replicates": args.replicates,
        "batch_size": args.batch_size,
        "base_seed": BASE_SEED,
        "network_ids": network_ids,
    }

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config") != config:
            raise RuntimeError(
                "This output folder belongs to a different run configuration. "
                "Choose a new --output folder."
            )
    else:
        existing = list(output.glob("*.csv"))
        if existing:
            raise RuntimeError(
                "Output folder already contains CSV files but has no checkpoint. "
                "Choose a new --output folder."
            )
        checkpoint = {
            "config": config,
            "completed_networks": [],
            "failures": [],
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2),
            encoding="utf-8",
        )

    completed = set(checkpoint["completed_networks"])
    failures = checkpoint["failures"]

    print(f"Networks in this run: {len(catalog)}", flush=True)
    print(f"Random replicates per network: {args.replicates}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)
    print(f"Output folder: {output}", flush=True)

    for number, row in catalog.iterrows():
        network_id = str(row["network"])

        if network_id in completed:
            print(
                f"[{number + 1}/{len(catalog)}] {network_id}: already completed",
                flush=True,
            )
            continue

        matrix_path = CACHE / f"{network_id}_matrix.csv"

        try:
            matrix = pd.read_csv(matrix_path, index_col=0)
            graph = build_bipartite_network(matrix)

            if graph.number_of_nodes() != int(row["species_richness"]):
                raise ValueError(
                    "Species count differs from validated catalog."
                )

            if graph.number_of_edges() != int(row["observed_interactions"]):
                raise ValueError(
                    "Interaction count differs from validated catalog."
                )

            seed = stable_seed(network_id)
            replicate_records = []

            counts = {metric: 0 for metric in METRICS}
            means = {
                metric: np.zeros(len(GRID), dtype=float)
                for metric in METRICS
            }
            m2 = {
                metric: np.zeros(len(GRID), dtype=float)
                for metric in METRICS
            }

            completed_replicates = 0

            for batch_start in range(0, args.replicates, args.batch_size):
                batch_count = min(
                    args.batch_size,
                    args.replicates - batch_start,
                )
                batch_seed = (seed + batch_start) % (2**32)

                curves = random_removal_curves(
                    graph,
                    replicates=batch_count,
                    seed=batch_seed,
                )

                batch_summary = summarize_curves(curves)

                for record in batch_summary.to_dict("records"):
                    record["replicate"] = (
                        batch_start + int(record["replicate"])
                    )
                    replicate_records.append({
                        "network": network_id,
                        "interaction_type": row["interaction_type"],
                        "study_reference_id": row["study_reference_id"],
                        "species_richness": row["species_richness"],
                        "bipartite_connectance": row["bipartite_connectance"],
                        "matrix_weighted": row["matrix_weighted"],
                        "seed": seed,
                        "batch_seed": batch_seed,
                        **record,
                    })

                for curve in curves:
                    update_curve_statistics(curve, counts, means, m2)

                completed_replicates += batch_count

                if (
                    completed_replicates % 1000 == 0
                    or completed_replicates == args.replicates
                ):
                    print(
                        f"    {network_id}: "
                        f"{completed_replicates:,}/{args.replicates:,} "
                        "random simulations",
                        flush=True,
                    )

                del curves
                del batch_summary

            random_summary = pd.DataFrame(replicate_records)

            targeted_curve = targeted_removal_curve(graph)
            targeted_summary = summarize_curves([targeted_curve])

            targeted_records = []
            for record in targeted_summary.to_dict("records"):
                targeted_records.append({
                    "network": network_id,
                    "interaction_type": row["interaction_type"],
                    "study_reference_id": row["study_reference_id"],
                    "species_richness": row["species_richness"],
                    "bipartite_connectance": row["bipartite_connectance"],
                    "matrix_weighted": row["matrix_weighted"],
                    "seed": seed,
                    "batch_seed": None,
                    **record,
                })

            all_records = replicate_records + targeted_records

            summary = (
                pd.DataFrame(all_records)
                .groupby(
                    [
                        "network",
                        "interaction_type",
                        "study_reference_id",
                        "species_richness",
                        "bipartite_connectance",
                        "matrix_weighted",
                        "scenario",
                    ],
                    dropna=False,
                )
                .agg(
                    robustness_auc_mean=("robustness_auc", "mean"),
                    robustness_auc_sd=("robustness_auc", "std"),
                    interaction_retention_auc_mean=(
                        "interaction_retention_auc", "mean"
                    ),
                    interaction_retention_auc_sd=(
                        "interaction_retention_auc", "std"
                    ),
                    simulations=("replicate", "count"),
                )
                .reset_index()
            )

            curve_records = curve_statistic_records(
                network_id,
                "random",
                counts,
                means,
                m2,
            )

            targeted_count = {metric: 1 for metric in METRICS}
            targeted_means = {}
            targeted_m2 = {}

            for metric in METRICS:
                targeted_means[metric] = np.interp(
                    GRID,
                    targeted_curve["fraction_species_removed"].to_numpy(),
                    targeted_curve[metric].to_numpy(),
                )
                targeted_m2[metric] = np.zeros(len(GRID), dtype=float)

            curve_records.extend(
                curve_statistic_records(
                    network_id,
                    "targeted_highest_degree",
                    targeted_count,
                    targeted_means,
                    targeted_m2,
                )
            )

            # Write a network's outputs only after all its simulations succeed.
            append_csv(
                replicate_records + targeted_records,
                output / "experiment_replicates.csv",
            )
            append_csv(
                summary.to_dict("records"),
                output / "network_summary.csv",
            )
            append_csv(
                curve_records,
                output / "mean_robustness_curves.csv",
            )

            completed.add(network_id)
            checkpoint["completed_networks"] = sorted(completed)
            checkpoint["failures"] = failures
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2),
                encoding="utf-8",
            )

            print(
                f"[{number + 1}/{len(catalog)}] {network_id}: COMPLETE "
                f"({graph.number_of_nodes()} species, "
                f"{graph.number_of_edges()} interactions)",
                flush=True,
            )

        except Exception as error:
            failure = {
                "network": network_id,
                "error": repr(error),
            }
            failures = [
                item for item in failures
                if item.get("network") != network_id
            ]
            failures.append(failure)
            checkpoint["failures"] = failures
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2),
                encoding="utf-8",
            )
            print(
                f"[FAILED {number + 1}/{len(catalog)}] "
                f"{network_id}: {error}",
                flush=True,
            )

    report = {
        "networks_in_run": len(catalog),
        "successful_networks": len(completed),
        "failed_networks": len(failures),
        "random_replicates_per_network": args.replicates,
        "successful_random_simulations": (
            len(completed) * args.replicates
        ),
        "successful_targeted_simulations": len(completed),
        "base_seed": BASE_SEED,
        "batch_size": args.batch_size,
        "elapsed_seconds_this_session": round(time.time() - start, 2),
        "completed_network_ids": sorted(completed),
        "failures": failures,
    }

    (output / "experiment_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT SESSION FINISHED")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print(f"\nResults folder: {output}", flush=True)


if __name__ == "__main__":
    main()
