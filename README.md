# Structural Robustness of Bipartite Ecological Networks Under Simulated Species Loss

## Research Question

How are species richness and connectance associated with the structural robustness of bipartite ecological networks under simulated species loss across different ecological interaction categories?

## Project Overview

This project investigates the structural robustness of empirical bipartite ecological networks under simulated species removal.

Using network data from the [Web of Life database](https://www.web-of-life.es/), the study examines whether network properties—particularly species richness and connectance—are associated with the ability of a network to retain its largest connected component as species are removed.

The final dataset contains **285 validated ecological networks** across six interaction categories.

| Interaction Category | Networks |
|---|---:|
| Pollination | 174 |
| Host-Parasite | 51 |
| Seed Dispersal | 34 |
| Anemone-Fish | 17 |
| Plant-Ant | 5 |
| Plant-Herbivore | 4 |
| **Total** | **285** |

Each network underwent 10,000 random-removal simulations, producing **2,850,000 random-removal replicates** across the complete dataset. Targeted-removal simulations were also conducted.

## Main Findings

The primary statistical model found positive associations between both species richness and connectance and structural robustness under random species removal.

| Predictor | Standardized Coefficient | p-value |
|---|---:|---:|
| Species richness | +0.0485 | < 0.001 |
| Connectance | +0.0542 | < 0.001 |

Additional model information:

- Networks analyzed: 285
- Provisional publication-reference clusters: 108
- Model R²: 0.535

These findings describe statistical associations, not causal relationships. They concern structural responses in simulated networks and should not be interpreted as direct predictions of ecosystem survival, population viability, or ecological recovery.

## Research Method

The study represents each ecological network as a bipartite graph with two distinct node partitions.

The primary analysis uses binary interaction matrices. Positive interaction values indicate an existing interaction, while zero values indicate its absence.

Two important network properties are examined:

**Species richness:** The total number of species represented by nodes in both partitions.

**Connectance:** The proportion of possible cross-partition interactions that are observed.

\[
C = \frac{E}{N_1 \times N_2}
\]

where:
- \(E\) is the number of observed interactions.
- \(N_1\) and \(N_2\) are the numbers of species in the two partitions.

Robustness is measured using the area under the curve describing the fraction of original species retained in the largest connected component as species are removed.

Both random and targeted species-removal strategies are evaluated. Statistical models examine associations between network properties and robustness, with sensitivity analyses and model diagnostics used to assess the findings.

See [`methods.md`](methods.md) for the complete methodology.

## Repository Structure

```text
ecological-network-resilience/
│
├── data/
│   └── Web of Life catalog and data inputs
│
├── references/
│   └── REFERENCES.md
│
├── results/
│   └── web_of_life/
│       ├── validated network records
│       ├── experiment_10000/
│       ├── statistical_analysis_10000/
│       └── poster_figures/
│
├── scripts/
│   ├── run_web_of_life_experiment.py
│   ├── analyze_web_of_life.py
│   └── create_poster_figures.py
│
├── src/
│   └── ecorobust/
│
├── tests/
│
├── import_web_of_life.py
├── methods.md
├── pyproject.toml
└── README.md
```

The repository includes final experiment summaries, statistical outputs, and research figures. The full replicate-level simulation file and downloaded data caches are excluded because of their size.

## Installation

Python 3.11 or newer is required.

Clone the repository:

```bash
git clone https://github.com/ichita8/ecological-network-resilience.git
cd ecological-network-resilience
```

Create and activate a virtual environment:

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Reproducing the Analysis

The project is organized into four main stages:

1. **Data retrieval and validation:** `import_web_of_life.py` retrieves and validates ecological network records from Web of Life.
2. **Robustness simulations:** `scripts/run_web_of_life_experiment.py` performs random and targeted species-removal experiments.
3. **Statistical analysis:** `scripts/analyze_web_of_life.py` evaluates relationships between network properties and robustness.
4. **Figure generation:** `scripts/create_poster_figures.py` generates figures for research presentation.

The final experiment outputs are included in the repository. Reproducing the complete 10,000-replicate experiment requires the original network inputs and substantial computational time.

## Testing

The automated test suite can be run with:

```bash
python -m pytest -q
```

The current test suite passes all **9 tests**.

## Limitations

- Structural connectivity is not equivalent to ecosystem health, population viability, or ecological recovery.
- Simulated species removal is a computational stress test, not an extinction forecast.
- The primary analysis binarizes interaction matrices and therefore does not directly measure interaction-strength effects.
- Ecological networks can differ in sampling effort, geographic scale, and construction.
- Publication-reference groupings are provisional normalized groups, not independently verified study identities.
- Some interaction categories have relatively small sample sizes.
- Cluster-robust standard errors address some dependence among networks but do not eliminate all potential sources of bias or model misspecification.
- Monte Carlo replicates improve simulation precision but do not increase the number of independent empirical networks.

## Data and References

The ecological network data were obtained from the [Web of Life database](https://www.web-of-life.es/).

The bibliography is available in [`references/REFERENCES.md`](references/REFERENCES.md). It contains source-reference information extracted from the validated network records, alongside methodological references.

When reusing individual network data, researchers should consult the bibliography and cite the original studies where identifiable, as well as credit the Web of Life database.

Some source records do not contain complete citation information or DOI identifiers. The bibliography preserves the available database metadata and identifies this limitation.

## License and Data Attribution

The source data remain subject to the terms and attribution requirements of their original providers. Inclusion of analysis code and derived research outputs in this repository does not imply ownership of the original ecological network datasets.

Please consult Web of Life and the original publications before redistributing source datasets.
