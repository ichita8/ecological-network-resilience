from __future__ import annotations

import json
import time
import re
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CATALOG_PATH = Path("data/web_of_life/networks.csv")
CACHE_DIR = Path("data/web_of_life/full_cache")
OUTPUT_DIR = Path("results/web_of_life")

BASE_URL = "https://www.web-of-life.es/api/v1/networks"
MAX_RETRIES = 4
REQUEST_DELAY = 0.15

EXCLUDED_CATEGORY = "Food Webs"

REQUIRED_CATALOG_COLUMNS = {
    "network", "interaction_type", "weighted",
    "species", "rows", "columns", "interactions",
    "connectance", "author", "reference", "doi"
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_reference(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[.,;:\s]+$", "", value)

    return value


def is_missing(value):
    return pd.isna(value) or str(value).strip() == ""


def to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def download_file(url, destination):
    """
    Download a file with retries.
    Existing cached files are reused.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return "CACHE"

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "EcologicalNetworkResearch/1.0"}
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()

            if not content:
                raise ValueError("Downloaded file was empty.")

            temporary = destination.with_suffix(
                destination.suffix + ".tmp"
            )

            temporary.write_bytes(content)
            temporary.replace(destination)

            time.sleep(REQUEST_DELAY)

            return "DOWNLOADED"

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Download failed after {MAX_RETRIES} attempts: {last_error}"
    )


def download_network_files(network_id):
    safe_id = quote(str(network_id), safe="")

    paths = {
        "matrix": CACHE_DIR / f"{network_id}_matrix.csv",
        "edges": CACHE_DIR / f"{network_id}_edges.csv",
        "species": CACHE_DIR / f"{network_id}_species.csv",
    }

    urls = {
        name: f"{BASE_URL}/{safe_id}/{name}.csv"
        for name in paths
    }

    statuses = {}

    for name in paths:
        statuses[name] = download_file(
            urls[name],
            paths[name]
        )

    return paths, statuses


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_network(catalog_record):
    network_id = str(catalog_record["network"])

    result = {
        "network": network_id,
        "interaction_type": catalog_record["interaction_type"],
        "weighted_original": catalog_record["weighted"],
        "author": catalog_record["author"],
        "reference": catalog_record["reference"],
        "doi": catalog_record["doi"],
        "status": "REJECTED",
        "reason": "",
        "validation_flags": "",
        "download_status": "",
    }

    try:
        paths, download_status = download_network_files(network_id)

        result["download_status"] = json.dumps(download_status)

        # ----------------------------------------------------
        # Read all three data files
        # ----------------------------------------------------

        matrix_df = pd.read_csv(paths["matrix"], index_col=0)
        edges = pd.read_csv(paths["edges"])
        species = pd.read_csv(paths["species"])

        required_edge_columns = {
            "network", "row_species", "column_species",
            "row_role", "column_role", "strength"
        }

        required_species_columns = {
            "species", "side", "role"
        }

        if not required_edge_columns.issubset(edges.columns):
            raise ValueError("Edge file is missing required columns.")

        if not required_species_columns.issubset(species.columns):
            raise ValueError("Species file is missing required columns.")

        # ----------------------------------------------------
        # Matrix validation
        # ----------------------------------------------------

        if matrix_df.empty:
            raise ValueError("Matrix is empty.")

        row_labels = matrix_df.index.astype(str).str.strip()
        column_labels = matrix_df.columns.astype(str).str.strip()

        if any(label == "" for label in row_labels):
            raise ValueError("Matrix contains blank row labels.")

        if any(label == "" for label in column_labels):
            raise ValueError("Matrix contains blank column labels.")

        if row_labels.duplicated().any():
            raise ValueError("Duplicate matrix row labels.")

        if column_labels.duplicated().any():
            raise ValueError("Duplicate matrix column labels.")

        matrix_df.index = row_labels
        matrix_df.columns = column_labels

        try:
            matrix = matrix_df.to_numpy(dtype=float)
        except (ValueError, TypeError):
            raise ValueError("Matrix contains non-numeric values.")

        if not np.isfinite(matrix).all():
            raise ValueError("Matrix contains missing or infinite values.")

        if (matrix < 0).any():
            raise ValueError("Matrix contains negative interaction strengths.")

        observed = matrix > 0

        row_count, column_count = matrix.shape
        species_count = row_count + column_count
        interaction_count = int(observed.sum())

        if interaction_count == 0:
            raise ValueError("Network has no observed interactions.")

        # A bipartite network must have distinct species on each side.
        if set(row_labels).intersection(set(column_labels)):
            raise ValueError(
                "Species labels overlap between matrix sides."
            )

        # ----------------------------------------------------
        # Species table validation
        # ----------------------------------------------------

        species["species"] = species["species"].astype(str).str.strip()
        species["side"] = (
            species["side"].astype(str).str.strip().str.lower()
        )

        if species["species"].eq("").any():
            raise ValueError("Species table contains blank labels.")

        if not species["side"].isin(["row", "column"]).all():
            raise ValueError(
                "Species table contains unexpected side labels."
            )

        if species.duplicated(["species", "side"]).any():
            raise ValueError(
                "Duplicate species records within the same side."
            )

        row_species = set(
            species.loc[species["side"] == "row", "species"]
        )

        column_species = set(
            species.loc[species["side"] == "column", "species"]
        )

        if row_species != set(row_labels):
            raise ValueError(
                "Species table row-side labels do not match matrix rows."
            )

        if column_species != set(column_labels):
            raise ValueError(
                "Species table column-side labels do not match matrix columns."
            )

        if len(species) != species_count:
            raise ValueError(
                "Species table record count does not match matrix dimensions."
            )

        # ----------------------------------------------------
        # Edge table validation
        # ----------------------------------------------------

        if edges.empty:
            raise ValueError("Edge table is empty.")

        edges["row_species"] = (
            edges["row_species"].astype(str).str.strip()
        )
        edges["column_species"] = (
            edges["column_species"].astype(str).str.strip()
        )

        if edges[["row_species", "column_species"]].isna().any().any():
            raise ValueError("Edge table has missing endpoint labels.")

        if edges.duplicated(
            ["row_species", "column_species"]
        ).any():
            raise ValueError("Edge table contains duplicate interactions.")

        if not edges["network"].astype(str).eq(network_id).all():
            raise ValueError("Edge table contains an unexpected network ID.")

        try:
            edge_strengths = pd.to_numeric(
                edges["strength"], errors="raise"
            ).to_numpy(dtype=float)
        except (ValueError, TypeError):
            raise ValueError("Edge strengths contain non-numeric values.")

        if not np.isfinite(edge_strengths).all():
            raise ValueError("Edge strengths contain missing or infinite values.")

        if (edge_strengths <= 0).any():
            raise ValueError("Edge table contains zero or negative strengths.")

        # Every edge must point to a valid matrix cell.
        row_index = {name: i for i, name in enumerate(row_labels)}
        column_index = {name: i for i, name in enumerate(column_labels)}

        edge_pairs = set()

        for row in edges.itertuples(index=False):
            r = row.row_species
            c = row.column_species

            if r not in row_index or c not in column_index:
                raise ValueError(
                    "Edge table contains a species absent from the matrix."
                )

            i = row_index[r]
            j = column_index[c]

            if matrix[i, j] <= 0:
                raise ValueError(
                    "Edge table contains an interaction absent from the matrix."
                )

            if not np.isclose(
                matrix[i, j],
                float(row.strength),
                rtol=1e-9,
                atol=1e-12
            ):
                raise ValueError(
                    "Edge strength does not match its matrix cell."
                )

            edge_pairs.add((r, c))

        # Every positive matrix cell must appear in the edge table.
        matrix_pairs = set()

        positive_rows, positive_columns = np.where(observed)

        for i, j in zip(positive_rows, positive_columns):
            matrix_pairs.add((row_labels[i], column_labels[j]))

        if matrix_pairs != edge_pairs:
            raise ValueError(
                "Matrix and edge table do not contain exactly the same interactions."
            )

        # ----------------------------------------------------
        # Catalog comparisons: flag discrepancies, don't
        # automatically exclude otherwise valid networks.
        # ----------------------------------------------------

        flags = []

        catalog_rows = to_number(catalog_record["rows"])
        catalog_columns = to_number(catalog_record["columns"])
        catalog_species = to_number(catalog_record["species"])
        catalog_edges = to_number(catalog_record["interactions"])
        catalog_connectance = to_number(catalog_record["connectance"])

        if not np.isnan(catalog_rows) and catalog_rows != row_count:
            flags.append("CATALOG_ROW_COUNT_MISMATCH")

        if not np.isnan(catalog_columns) and catalog_columns != column_count:
            flags.append("CATALOG_COLUMN_COUNT_MISMATCH")

        if not np.isnan(catalog_species) and catalog_species != species_count:
            flags.append("CATALOG_SPECIES_COUNT_MISMATCH")

        if not np.isnan(catalog_edges) and catalog_edges != interaction_count:
            flags.append("CATALOG_INTERACTION_COUNT_MISMATCH")

        connectance = interaction_count / (row_count * column_count)

        if (
            not np.isnan(catalog_connectance)
            and not np.isclose(
                catalog_connectance,
                connectance,
                rtol=0.01,
                atol=0.0001
            )
        ):
            flags.append("CATALOG_CONNECTANCE_MISMATCH")

        # ----------------------------------------------------
        # Primary analysis variables
        # ----------------------------------------------------

        result.update({
            "status": "ACCEPTED",
            "reason": "",
            "validation_flags": ";".join(flags),
            "rows": row_count,
            "columns": column_count,
            "species_richness": species_count,
            "observed_interactions": interaction_count,
            "possible_interactions": row_count * column_count,
            "bipartite_connectance": connectance,
            "matrix_weighted": bool(
                not np.all(np.isin(matrix, [0, 1]))
            ),
            "catalog_connectance": catalog_connectance,
            "catalog_interactions": catalog_edges,
            "normalized_reference": normalize_reference(
                catalog_record["reference"]
            ),
        })

        return result

    except Exception as exc:
        result["reason"] = str(exc)
        return result


# ============================================================
# MAIN IMPORT
# ============================================================

def main():
    print("=" * 70)
    print("WEB OF LIFE FULL NETWORK IMPORT AND VALIDATION")
    print("=" * 70)

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found: {CATALOG_PATH}"
        )

    catalog = pd.read_csv(CATALOG_PATH)

    missing = REQUIRED_CATALOG_COLUMNS - set(catalog.columns)

    if missing:
        raise ValueError(
            f"Catalog is missing columns: {sorted(missing)}"
        )

    if catalog["network"].isna().any():
        raise ValueError("Catalog contains missing network identifiers.")

    if catalog["network"].duplicated().any():
        raise ValueError("Catalog contains duplicate network identifiers.")

    print(f"\nCatalog records: {len(catalog)}")
    print(f"Unique identifiers: {catalog['network'].nunique()}")

    # Exclude food webs from the primary bipartite candidate sample.
    candidates = catalog[
        catalog["interaction_type"] != EXCLUDED_CATEGORY
    ].copy()

    print(f"Primary bipartite candidates: {len(candidates)}")
    print(f"Intentionally excluded food webs: {len(catalog) - len(candidates)}")

    results = []

    for number, (_, record) in enumerate(
        candidates.iterrows(), start=1
    ):
        network_id = str(record["network"])

        print(
            f"[{number:03d}/{len(candidates):03d}] "
            f"Validating {network_id} ... ",
            end="",
            flush=True
        )

        result = validate_network(record)
        results.append(result)

        if result["status"] == "ACCEPTED":
            print("ACCEPTED")

            if result["validation_flags"]:
                print(
                    "    REVIEW FLAGS: "
                    + result["validation_flags"]
                )
        else:
            print("REJECTED")
            print("    REASON: " + result["reason"])

    # --------------------------------------------------------
    # Build publication groups from all candidate records.
    # Group IDs are provisional; references still require
    # manual review before being called independent studies.
    # --------------------------------------------------------

    candidate_references = candidates[
        ["network", "reference"]
    ].copy()

    candidate_references["normalized_reference"] = (
        candidate_references["reference"].map(normalize_reference)
    )

    unique_references = sorted(
        candidate_references["normalized_reference"].unique()
    )

    reference_ids = {
        reference: f"REF_{i:03d}"
        for i, reference in enumerate(unique_references, start=1)
    }

    reference_map = dict(
        zip(
            candidate_references["network"],
            candidate_references["normalized_reference"]
        )
    )

    for result in results:
        normalized = reference_map.get(result["network"], "")
        result["study_reference_id"] = reference_ids.get(
            normalized, ""
        )

    # --------------------------------------------------------
    # Include intentional food-web exclusions in the audit.
    # --------------------------------------------------------

    accepted = pd.DataFrame(results)

    food_webs = catalog[
        catalog["interaction_type"] == EXCLUDED_CATEGORY
    ].copy()

    intentional_exclusions = pd.DataFrame({
        "network": food_webs["network"].astype(str),
        "interaction_type": food_webs["interaction_type"],
        "status": "EXCLUDED_BY_DESIGN",
        "reason": "Food webs excluded from primary bipartite analysis."
    })

    rejected = accepted[accepted["status"] != "ACCEPTED"].copy()
    validated = accepted[accepted["status"] == "ACCEPTED"].copy()

    audit = pd.concat(
        [accepted, intentional_exclusions],
        ignore_index=True,
        sort=False
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validated_path = OUTPUT_DIR / "validated_networks.csv"
    excluded_path = OUTPUT_DIR / "excluded_networks.csv"
    audit_path = OUTPUT_DIR / "network_validation_audit.csv"
    reference_path = OUTPUT_DIR / "publication_reference_groups.csv"

    validated.to_csv(validated_path, index=False)

    all_exclusions = pd.concat(
        [rejected, intentional_exclusions],
        ignore_index=True,
        sort=False
    )

    all_exclusions.to_csv(excluded_path, index=False)
    audit.to_csv(audit_path, index=False)

    reference_groups = (
        candidate_references
        .merge(
            pd.DataFrame(results)[
                ["network", "status", "study_reference_id"]
            ],
            on="network",
            how="left"
        )
        .groupby(
            ["study_reference_id", "normalized_reference"],
            dropna=False
        )
        .agg(
            candidate_networks=("network", "count"),
            accepted_networks=(
                "status",
                lambda values: int((values == "ACCEPTED").sum())
            )
        )
        .reset_index()
        .sort_values("candidate_networks", ascending=False)
    )

    reference_groups.to_csv(reference_path, index=False)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("IMPORT AND VALIDATION SUMMARY")
    print("=" * 70)

    print(f"Catalog records: {len(catalog)}")
    print(f"Bipartite candidates: {len(candidates)}")
    print(f"Accepted networks: {len(validated)}")
    print(f"Rejected candidate networks: {len(rejected)}")
    print(f"Food webs excluded by design: {len(food_webs)}")

    print(
        "Accepted networks with catalog review flags:",
        int(validated["validation_flags"].fillna("").ne("").sum())
    )

    print(
        "Normalized reference groups:",
        len(reference_groups)
    )

    print("\nACCEPTED NETWORKS BY INTERACTION CATEGORY:")

    if not validated.empty:
        print(
            validated["interaction_type"]
            .value_counts()
            .to_string()
        )
    else:
        print("No networks passed validation.")

    print("\nOUTPUT FILES:")
    print(validated_path)
    print(excluded_path)
    print(audit_path)
    print(reference_path)

    print("\nIMPORT COMPLETE")


if __name__ == "__main__":
    main()
