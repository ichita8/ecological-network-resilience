from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/web_of_life/experiment_10000/network_summary.csv"
OLD_INPUT = ROOT / "results/web_of_life/experiment/network_summary.csv"
CATALOG = ROOT / "results/web_of_life/validated_networks.csv"
OLD_COEFFICIENTS = ROOT / "results/web_of_life/statistical_analysis/model_coefficients.csv"
OUT = ROOT / "results/web_of_life/statistical_analysis_10000"
OUT.mkdir(parents=True, exist_ok=True)


def fit_model(data, formula, name):
    groups = data["study_reference_id"]
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
    )
    ci = model.conf_int()
    table = pd.DataFrame({
        "model": name,
        "term": model.params.index,
        "coefficient": model.params.values,
        "clustered_se": model.bse.values,
        "p_value": model.pvalues.values,
        "ci_lower": ci[0].values,
        "ci_upper": ci[1].values,
    })
    print("\n" + "=" * 68)
    print(name)
    print("=" * 68)
    print(table.to_string(index=False))
    print("Networks:", int(model.nobs))
    print("Provisional publication groups:", groups.nunique())
    print("R-squared:", round(model.rsquared, 4))
    return model, table


def make_model_diagnostics(name, model, data):
    influence = model.get_influence()
    fitted = np.asarray(model.fittedvalues)
    residuals = np.asarray(model.resid)
    cooks = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    studentized = influence.resid_studentized_internal

    try:
        bp_lm, bp_p, bp_f, bp_f_p = het_breuschpagan(
            residuals, model.model.exog
        )
    except Exception:
        bp_lm = bp_p = bp_f = bp_f_p = np.nan

    jb = sm.stats.jarque_bera(residuals)
    diagnostic = {
        "model": name,
        "n": int(model.nobs),
        "publication_clusters": data["study_reference_id"].nunique(),
        "breusch_pagan_lm": bp_lm,
        "breusch_pagan_p": bp_p,
        "breusch_pagan_f": bp_f,
        "breusch_pagan_f_p": bp_f_p,
        "jarque_bera": float(jb[0]),
        "jarque_bera_p": float(jb[1]),
        "residual_skewness": float(skew(residuals, bias=False)),
        "residual_excess_kurtosis": float(kurtosis(residuals, fisher=True, bias=False)),
        "max_cooks_distance": float(np.nanmax(cooks)),
        "max_leverage": float(np.nanmax(leverage)),
        "studentized_residual_abs_max": float(np.nanmax(np.abs(studentized))),
        "cooks_over_4_over_n": int(np.sum(cooks > 4 / len(data))),
    }

    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    pd.DataFrame({
        "network": data["network"].to_numpy(),
        "study_reference_id": data["study_reference_id"].to_numpy(),
        "fitted": fitted,
        "residual": residuals,
        "studentized_residual": studentized,
        "leverage": leverage,
        "cooks_distance": cooks,
    }).to_csv(OUT / f"residuals_{slug}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted, residuals, alpha=0.7)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals vs fitted: {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"residuals_vs_fitted_{slug}.png", dpi=250)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    sm.qqplot(residuals, line="45", ax=ax, fit=True)
    ax.set_title(f"Normal Q-Q: {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"qq_plot_{slug}.png", dpi=250)
    plt.close(fig)
    return diagnostic


def make_vif_table(name, model):
    exog = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    rows = []
    for index, column in enumerate(exog.columns):
        if column.lower() in ("intercept", "const"):
            continue
        try:
            vif = variance_inflation_factor(exog.to_numpy(), index)
        except Exception:
            vif = np.nan
        rows.append({"model": name, "term": column, "vif": vif})
    return rows


def publication_audit():
    catalog = pd.read_csv(CATALOG)
    accepted = catalog[catalog["status"] == "ACCEPTED"].copy()
    accepted["study_reference_id"] = accepted["study_reference_id"].astype(str)
    rows = []

    for study_id, group in accepted.groupby("study_reference_id", dropna=False):
        refs = sorted(group["reference"].dropna().astype(str).unique())
        authors = sorted(group["author"].dropna().astype(str).unique())
        dois = sorted(group["doi"].dropna().astype(str).unique())
        normalized = sorted(group["normalized_reference"].dropna().astype(str).unique())
        categories = group["interaction_type"].value_counts().to_dict()

        if len(normalized) > 1:
            flag = "REVIEW: multiple normalized references"
        elif len(dois) > 1:
            flag = "REVIEW: multiple DOIs"
        elif len(refs) > 1:
            flag = "REVIEW: multiple raw reference strings"
        else:
            flag = "CHECK IDENTIFICATION"

        rows.append({
            "study_reference_id": study_id,
            "accepted_networks": len(group),
            "unique_raw_reference_strings": len(refs),
            "unique_author_strings": len(authors),
            "unique_dois": len(dois),
            "unique_normalized_references": len(normalized),
            "authors": " || ".join(authors),
            "dois": " || ".join(dois),
            "raw_references_for_manual_review": " || ".join(refs),
            "normalized_reference_values": " || ".join(normalized),
            "interaction_category_counts": "; ".join(
                f"{key}: {value}" for key, value in sorted(categories.items())
            ),
            "manual_review_flag": flag,
            "network_ids": "; ".join(sorted(group["network"].astype(str))),
        })

    audit = pd.DataFrame(rows).sort_values(
        ["accepted_networks", "study_reference_id"],
        ascending=[False, True],
    )
    audit.to_csv(OUT / "publication_cluster_manual_audit.csv", index=False)

    audit_summary = pd.DataFrame([{
        "accepted_networks": len(accepted),
        "provisional_clusters": accepted["study_reference_id"].nunique(),
        "clusters_with_multiple_raw_reference_strings":
            int((audit["unique_raw_reference_strings"] > 1).sum()),
        "clusters_with_multiple_dois": int((audit["unique_dois"] > 1).sum()),
        "clusters_with_multiple_normalized_references":
            int((audit["unique_normalized_references"] > 1).sum()),
    }])
    audit_summary.to_csv(OUT / "publication_cluster_audit_summary.csv", index=False)
    return audit_summary


def compare_replicate_levels(random):
    if OLD_INPUT.exists():
        old = pd.read_csv(OLD_INPUT)
        old = old[old["scenario"] == "random"].copy()
        merged = old.merge(
            random,
            on="network",
            suffixes=("_100rep", "_10000rep"),
            validate="one_to_one",
        )
        rows = []
        for outcome in ["robustness_auc_mean", "interaction_retention_auc_mean"]:
            a = merged[f"{outcome}_100rep"]
            b = merged[f"{outcome}_10000rep"]
            rows.append({
                "outcome": outcome,
                "matched_networks": len(merged),
                "mean_100rep": a.mean(),
                "mean_10000rep": b.mean(),
                "mean_absolute_network_difference": (b - a).abs().mean(),
                "maximum_absolute_network_difference": (b - a).abs().max(),
                "pearson_correlation": a.corr(b),
                "spearman_correlation": a.corr(b, method="spearman"),
            })
        pd.DataFrame(rows).to_csv(
            OUT / "network_outcome_stability_100_vs_10000.csv", index=False
        )
    else:
        pd.DataFrame([{
            "note": "Original 100-replicate summary file not found."
        }]).to_csv(OUT / "network_outcome_stability_100_vs_10000.csv", index=False)

    if OLD_COEFFICIENTS.exists():
        old_coef = pd.read_csv(OLD_COEFFICIENTS)
        new_coef = pd.read_csv(OUT / "model_coefficients.csv")
        old_coef["comparison_model"] = old_coef["model"].replace({
            "Weighted-matrix sensitivity":
                "Source-matrix weighting-status sensitivity"
        })
        new_coef["comparison_model"] = new_coef["model"]
        merged = old_coef.merge(
            new_coef,
            on=["comparison_model", "term"],
            suffixes=("_100rep", "_10000rep"),
            how="inner",
        )
        if not merged.empty:
            merged["coefficient_change"] = (
                merged["coefficient_10000rep"] - merged["coefficient_100rep"]
            )
        merged.to_csv(
            OUT / "model_coefficient_stability_100_vs_10000.csv", index=False
        )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"10,000-replicate summary not found: {INPUT}")

    summary = pd.read_csv(INPUT)
    random = summary[summary["scenario"] == "random"].copy()
    targeted = summary[summary["scenario"] == "targeted_highest_degree"].copy()

    if len(random) != 285 or random["network"].nunique() != 285:
        raise ValueError("Expected exactly 285 unique random-removal networks.")
    if len(targeted) != 285 or targeted["network"].nunique() != 285:
        raise ValueError("Expected exactly 285 unique targeted-removal networks.")
    if not (random["simulations"] == 10000).all():
        raise ValueError("Not every random network has exactly 10,000 simulations.")
    if not (targeted["simulations"] == 1).all():
        raise ValueError("Expected one deterministic targeted run per network.")

    random["log_richness"] = np.log(random["species_richness"])
    random["richness_z"] = (
        random["log_richness"] - random["log_richness"].mean()
    ) / random["log_richness"].std(ddof=1)
    random["connectance_z"] = (
        random["bipartite_connectance"] - random["bipartite_connectance"].mean()
    ) / random["bipartite_connectance"].std(ddof=1)
    random["weighted_original"] = (
        random["matrix_weighted"].astype(str).str.lower()
        .isin(["true", "1", "yes"])
    ).astype(int)

    category = random.groupby("interaction_type").agg(
        networks=("network", "count"),
        mean_robustness=("robustness_auc_mean", "mean"),
        sd_robustness=("robustness_auc_mean", "std"),
        mean_connectance=("bipartite_connectance", "mean"),
        mean_richness=("species_richness", "mean"),
        mean_retention=("interaction_retention_auc_mean", "mean"),
    ).reset_index()
    category.to_csv(OUT / "category_descriptives.csv", index=False)
    print("\nCATEGORY DESCRIPTIVES\n", category.to_string(index=False))

    correlation = pd.DataFrame([{
        "pearson_r_log_richness_connectance": random["log_richness"].corr(
            random["bipartite_connectance"], method="pearson"
        ),
        "spearman_richness_connectance": random["species_richness"].corr(
            random["bipartite_connectance"], method="spearman"
        ),
    }])
    correlation.to_csv(OUT / "predictor_correlation.csv", index=False)
    print("\nPREDICTOR CORRELATION\n", correlation.to_string(index=False))

    random["robustness_mc_se"] = (
        random["robustness_auc_sd"] / np.sqrt(random["simulations"])
    )
    random["retention_mc_se"] = (
        random["interaction_retention_auc_sd"] / np.sqrt(random["simulations"])
    )
    random["robustness_mc_ci_lower"] = (
        random["robustness_auc_mean"] - 1.96 * random["robustness_mc_se"]
    )
    random["robustness_mc_ci_upper"] = (
        random["robustness_auc_mean"] + 1.96 * random["robustness_mc_se"]
    )

    mc = pd.DataFrame({
        "measure": ["Robustness AUC", "Interaction-retention AUC"],
        "median_mc_se": [
            random["robustness_mc_se"].median(),
            random["retention_mc_se"].median(),
        ],
        "maximum_mc_se": [
            random["robustness_mc_se"].max(),
            random["retention_mc_se"].max(),
        ],
        "between_network_sd": [
            random["robustness_auc_mean"].std(),
            random["interaction_retention_auc_mean"].std(),
        ],
    })
    mc.to_csv(OUT / "monte_carlo_uncertainty.csv", index=False)
    random[[
        "network", "robustness_auc_mean", "robustness_auc_sd", "simulations",
        "robustness_mc_se", "robustness_mc_ci_lower", "robustness_mc_ci_upper",
        "interaction_retention_auc_mean", "interaction_retention_auc_sd",
        "retention_mc_se",
    ]].to_csv(OUT / "network_monte_carlo_uncertainty.csv", index=False)
    print("\nMONTE CARLO UNCERTAINTY\n", mc.to_string(index=False))

    specifications = [
        (random, "robustness_auc_mean ~ richness_z + connectance_z",
         "Primary random-removal robustness"),
        (random, "robustness_auc_mean ~ richness_z + connectance_z + weighted_original",
         "Source-matrix weighting-status sensitivity"),
        (random, "interaction_retention_auc_mean ~ richness_z + connectance_z",
         "Interaction-retention outcome"),
    ]

    counts = random["interaction_type"].value_counts()
    eligible = counts[counts >= 10].index
    category_data = random[random["interaction_type"].isin(eligible)].copy()
    specifications.append((
        category_data,
        "robustness_auc_mean ~ richness_z + connectance_z + C(interaction_type)",
        "Category-adjusted sensitivity",
    ))

    targeted = targeted.merge(
        random[["network", "richness_z", "connectance_z"]],
        on="network", how="inner", validate="one_to_one",
    )
    specifications.append((
        targeted,
        "robustness_auc_mean ~ richness_z + connectance_z",
        "Targeted-removal robustness",
    ))

    models, coefficient_tables, diagnostics, vif_rows, fit_rows = [], [], [], [], []
    for data, formula, name in specifications:
        model, table = fit_model(data, formula, name)
        models.append((name, model, data))
        coefficient_tables.append(table)
        diagnostics.append(make_model_diagnostics(name, model, data))
        vif_rows.extend(make_vif_table(name, model))
        fit_rows.append({
            "model": name,
            "networks": int(model.nobs),
            "provisional_publication_clusters":
                data["study_reference_id"].nunique(),
            "r_squared": model.rsquared,
            "adjusted_r_squared": model.rsquared_adj,
        })

    pd.concat(coefficient_tables, ignore_index=True).to_csv(
        OUT / "model_coefficients.csv", index=False
    )
    pd.DataFrame(fit_rows).to_csv(OUT / "model_fit_statistics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    pd.DataFrame(vif_rows).to_csv(OUT / "variance_inflation_factors.csv", index=False)

    for predictor, xlabel, filename in [
        ("log_richness", "Log species richness", "robustness_vs_richness.png"),
        ("bipartite_connectance", "Bipartite connectance", "robustness_vs_connectance.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(random[predictor], random["robustness_auc_mean"], alpha=0.65)
        slope, intercept = np.polyfit(
            random[predictor], random["robustness_auc_mean"], 1
        )
        x = np.linspace(random[predictor].min(), random[predictor].max(), 100)
        ax.plot(x, intercept + slope * x)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Random-removal robustness AUC")
        ax.set_title("Structural robustness across networks")
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=300)
        plt.close(fig)

    groups = list(random.groupby("interaction_type", sort=False))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([group["robustness_auc_mean"].to_numpy() for _, group in groups])
    ax.set_xticks(
        range(1, len(groups) + 1),
        [name for name, _ in groups],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Random-removal robustness AUC")
    ax.set_title("Robustness by interaction category")
    fig.tight_layout()
    fig.savefig(OUT / "robustness_by_category.png", dpi=300)
    plt.close(fig)

    audit_summary = publication_audit()
    print("\nPUBLICATION CLUSTER AUDIT SUMMARY\n", audit_summary.to_string(index=False))
    compare_replicate_levels(random)

    print("\nMODEL DIAGNOSTICS\n", pd.DataFrame(diagnostics).to_string(index=False))
    print("\nVIF\n", pd.DataFrame(vif_rows).to_string(index=False))
    print("\nANALYSIS COMPLETE")
    print("Results saved to:", OUT)
    print("Publication clusters are provisional and require manual verification.")


if __name__ == "__main__":
    main()
