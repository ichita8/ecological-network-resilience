
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results" / "web_of_life"
EXPERIMENT = RESULTS / "experiment_10000"
ANALYSIS = RESULTS / "statistical_analysis_10000"

OUTPUT = RESULTS / "poster_figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = EXPERIMENT / "network_summary.csv"
CURVES_FILE = EXPERIMENT / "mean_robustness_curves.csv"
CATEGORY_FILE = ANALYSIS / "category_descriptives.csv"
COEFFICIENT_FILE = ANALYSIS / "model_coefficients.csv"


# ---------------------------------------------------------
# VISUAL STYLE
# ---------------------------------------------------------

NAVY = "#10283F"
TEAL = "#167D91"
SAGE = "#62A6A0"
GRAY = "#657786"
LIGHT = "#DCE8EF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# ---------------------------------------------------------
# LOAD AND VALIDATE DATA
# ---------------------------------------------------------

for path in [
    SUMMARY_FILE,
    CURVES_FILE,
    CATEGORY_FILE,
    COEFFICIENT_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

summary = pd.read_csv(SUMMARY_FILE)
curves = pd.read_csv(CURVES_FILE)
categories = pd.read_csv(CATEGORY_FILE)
coefficients = pd.read_csv(COEFFICIENT_FILE)

required_summary = {
    "network",
    "interaction_type",
    "species_richness",
    "bipartite_connectance",
    "scenario",
    "robustness_auc_mean",
}

if not required_summary.issubset(summary.columns):
    raise ValueError(
        f"Missing summary columns: "
        f"{required_summary - set(summary.columns)}"
    )

random_data = summary[
    summary["scenario"] == "random"
].copy()

targeted_data = summary[
    summary["scenario"] == "targeted_highest_degree"
].copy()

if len(random_data) != 285:
    raise ValueError(
        f"Expected 285 random network summaries; found {len(random_data)}"
    )

if len(targeted_data) != 285:
    raise ValueError(
        f"Expected 285 targeted summaries; found {len(targeted_data)}"
    )

if random_data["network"].nunique() != 285:
    raise ValueError("Random dataset does not contain 285 unique networks.")

if random_data["robustness_auc_mean"].isna().any():
    raise ValueError("Missing random-removal robustness values.")

print("DATA VALIDATION PASSED")
print(f"Random networks: {len(random_data)}")
print(f"Targeted networks: {len(targeted_data)}")
print(f"Categories: {categories['interaction_type'].nunique()}")


# ---------------------------------------------------------
# HELPER: SAVE HIGH-RESOLUTION PNG AND VECTOR SVG
# ---------------------------------------------------------

def save_figure(fig, name):
    fig.savefig(
        OUTPUT / f"{name}.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUTPUT / f"{name}.svg",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Created: {name}.png and {name}.svg")


def get_model_row(model_text, term):
    selected = coefficients[
        coefficients["model"].str.contains(
            model_text, case=False, regex=False
        )
        & (coefficients["term"] == term)
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Could not uniquely identify model={model_text!r}, "
            f"term={term!r}. Found {len(selected)} rows."
        )

    return selected.iloc[0]


# ---------------------------------------------------------
# FIGURE 1: SPECIES RICHNESS VS ROBUSTNESS
# ---------------------------------------------------------

x = random_data["species_richness"].to_numpy(dtype=float)
y = random_data["robustness_auc_mean"].to_numpy(dtype=float)

if np.any(x <= 0):
    raise ValueError("Species richness must be positive.")

log_x = np.log10(x)
slope, intercept = np.polyfit(log_x, y, 1)

line_x = np.linspace(log_x.min(), log_x.max(), 200)
line_y = intercept + slope * line_x

model = get_model_row(
    "Primary random-removal robustness",
    "richness_z",
)

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(
    x,
    y,
    s=35,
    alpha=0.62,
    color=TEAL,
    edgecolors="white",
    linewidths=0.4,
)

ax.plot(
    10 ** line_x,
    line_y,
    color=NAVY,
    linewidth=2.5,
)

ax.set_xscale("log")
ax.set_xlabel("Species richness (log scale)")
ax.set_ylabel("Random-removal structural robustness AUC")
ax.set_title("Species richness and structural robustness", pad=14)

ax.text(
    0.04,
    0.95,
    f"Standardized β = {model['coefficient']:.4f}\n"
    f"p = {model['p_value']:.2e}",
    transform=ax.transAxes,
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor=LIGHT,
        edgecolor="none",
    ),
)

ax.grid(axis="y", alpha=0.2)
save_figure(fig, "figure_1_species_richness")


# ---------------------------------------------------------
# FIGURE 2: CONNECTANCE VS ROBUSTNESS
# ---------------------------------------------------------

x = random_data["bipartite_connectance"].to_numpy(dtype=float)
y = random_data["robustness_auc_mean"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
line_x = np.linspace(x.min(), x.max(), 200)
line_y = intercept + slope * line_x

model = get_model_row(
    "Primary random-removal robustness",
    "connectance_z",
)

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(
    x,
    y,
    s=35,
    alpha=0.62,
    color=SAGE,
    edgecolors="white",
    linewidths=0.4,
)

ax.plot(
    line_x,
    line_y,
    color=NAVY,
    linewidth=2.5,
)

ax.set_xlabel("Bipartite connectance")
ax.set_ylabel("Random-removal structural robustness AUC")
ax.set_title("Connectance and structural robustness", pad=14)

ax.text(
    0.04,
    0.95,
    f"Standardized β = {model['coefficient']:.4f}\n"
    f"p = {model['p_value']:.2e}",
    transform=ax.transAxes,
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor=LIGHT,
        edgecolor="none",
    ),
)

ax.grid(axis="y", alpha=0.2)
save_figure(fig, "figure_2_connectance")


# ---------------------------------------------------------
# FIGURE 3: CATEGORY COMPARISON
# ---------------------------------------------------------

categories = categories.sort_values(
    "mean_robustness",
    ascending=True,
).copy()

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.barh(
    categories["interaction_type"],
    categories["mean_robustness"],
    color=TEAL,
    height=0.65,
    xerr=categories["sd_robustness"],
    error_kw={
        "ecolor": NAVY,
        "capsize": 3,
        "elinewidth": 1.2,
    },
)

for bar, (_, row) in zip(bars, categories.iterrows()):
    ax.text(
        row["mean_robustness"] + row["sd_robustness"] + 0.006,
        bar.get_y() + bar.get_height() / 2,
        f"{row['mean_robustness']:.3f}  (n={int(row['networks'])})",
        va="center",
        fontsize=9,
    )

ax.set_xlabel("Mean random-removal structural robustness AUC")
ax.set_title("Structural robustness across interaction categories", pad=14)
ax.set_xlim(0, 0.65)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

fig.text(
    0.5,
    0.015,
    "Error bars represent ±1 standard deviation across networks.",
    ha="center",
    fontsize=9,
    color=GRAY,
)

fig.tight_layout(rect=[0, 0.05, 1, 1])
save_figure(fig, "figure_3_interaction_categories")


# ---------------------------------------------------------
# FIGURE 4: RANDOM VS TARGETED REMOVAL CURVES
# ---------------------------------------------------------

curve_data = curves[
    (curves["metric"] == "largest_component_fraction")
    & curves["scenario"].isin(
        ["random", "targeted_highest_degree"]
    )
].copy()

if curve_data.empty:
    raise ValueError("No largest-component robustness curves found.")

# Average network-level mean curves across the 285 networks.
# SD here describes variation between networks, not simulation runs.

curve_summary = (
    curve_data.groupby(
        ["scenario", "fraction_species_removed"]
    )["mean_value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5.5))

curve_styles = {
    "random": {
        "color": TEAL,
        "label": "Random removal",
    },
    "targeted_highest_degree": {
        "color": NAVY,
        "label": "Adaptive highest-degree removal",
    },
}

for scenario, style in curve_styles.items():
    subset = curve_summary[
        curve_summary["scenario"] == scenario
    ].sort_values("fraction_species_removed")

    if subset.empty:
        continue

    fraction = subset["fraction_species_removed"].to_numpy()
    mean = subset["mean"].to_numpy()
    sd = subset["std"].fillna(0).to_numpy()

    ax.plot(
        fraction,
        mean,
        color=style["color"],
        linewidth=2.5,
        label=style["label"],
    )

    ax.fill_between(
        fraction,
        np.clip(mean - sd, 0, 1),
        np.clip(mean + sd, 0, 1),
        color=style["color"],
        alpha=0.13,
    )

ax.set_xlabel("Proportion of species removed")
ax.set_ylabel("Largest connected component / original species")
ax.set_title("Structural robustness under species removal", pad=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.03)
ax.legend(frameon=False)
ax.grid(alpha=0.2)

fig.text(
    0.5,
    0.015,
    "Lines show means across networks; shaded bands show ±1 network-level SD.",
    ha="center",
    fontsize=9,
    color=GRAY,
)

fig.tight_layout(rect=[0, 0.05, 1, 1])
save_figure(fig, "figure_4_removal_curves")


# ---------------------------------------------------------
# FIGURE 5: STANDARDIZED MODEL COEFFICIENTS
# ---------------------------------------------------------

models = [
    (
        "Random removal",
        "Primary random-removal robustness",
        "richness_z",
        "connectance_z",
    ),
    (
        "Targeted removal",
        "Targeted-removal robustness",
        "richness_z",
        "connectance_z",
    ),
]

fig, ax = plt.subplots(figsize=(8, 5.5))

positions = []
values = []
lower_errors = []
upper_errors = []
labels = []
colors = []

position = 0

for label, model_name, *terms in models:
    for term in terms:
        row = get_model_row(model_name, term)

        coefficient = float(row["coefficient"])
        low = float(row["ci_lower"])
        high = float(row["ci_upper"])

        positions.append(position)
        values.append(coefficient)
        lower_errors.append(coefficient - low)
        upper_errors.append(high - coefficient)

        predictor = (
            "Species richness"
            if term == "richness_z"
            else "Connectance"
        )

        labels.append(f"{label}: {predictor}")
        colors.append(
            TEAL if label == "Random removal" else NAVY
        )

        position += 1

y_positions = np.arange(len(values))

for i in range(len(values)):
    ax.errorbar(
        values[i],
        y_positions[i],
        xerr=[
            [lower_errors[i]],
            [upper_errors[i]],
        ],
        fmt="o",
        color=colors[i],
        markersize=8,
        capsize=4,
        linewidth=2,
    )

ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)

ax.set_yticks(y_positions)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Standardized regression coefficient (β)")
ax.set_title("Predictor associations across removal scenarios", pad=14)
ax.grid(axis="x", alpha=0.2)

save_figure(fig, "figure_5_model_coefficients")


# ---------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------

print("\n" + "=" * 55)
print("POSTER FIGURE GENERATION COMPLETE")
print("=" * 55)
print(f"Output folder: {OUTPUT}")
print("Figures created: 5")
print("Formats: 400-DPI PNG and vector SVG")
print("Simulation reruns: None")
print("=" * 55)
