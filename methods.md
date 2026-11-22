<CodeBlock language="markdown" editable>

## 1. Research Design

This study uses a computational comparative design to investigate associations between structural properties and the robustness of empirical bipartite ecological networks under simulated species removal.

The analysis uses network records obtained from the Web of Life database.

## 2. Network Representation

Each ecological network is represented as an undirected bipartite graph with two distinct node partitions.

The original interaction matrices are converted into binary representations. Positive matrix values indicate an observed interaction, while zero values indicate its absence.

All species represented in the original matrix are retained, including species with no recorded interactions.

## 3. Network Properties

The study examines species richness and connectance.

Species richness is the total number of species across both node partitions.

Connectance is the proportion of possible cross-partition interactions that are observed:

$$
C = \frac{E}{N_1 \times N_2}
$$

where E is the number of observed interactions and N₁ and N₂ are the numbers of species in the two partitions.

## 4. Species-Removal Simulations

Two removal strategies are evaluated.

**Random removal:** Species are removed in a randomly shuffled order. The experiment performs 10,000 random-removal replicates for each network using a documented random seed.

**Targeted removal:** Species with the highest current degree are removed first. Degree is recalculated after each removal, and ties are resolved alphabetically by species identifier.

## 5. Robustness Measurements

The primary outcome is structural robustness, measured as the normalized area under the curve (AUC) of the fraction of original species retained in the largest connected component as species are removed.

At each removal step, the largest connected component is divided by the original number of species.

A secondary outcome is interaction retention, calculated as the proportion of original interaction edges remaining after each removal step. Its AUC is also calculated.

Both AUC measures range from zero to one.

## 6. Statistical Analysis

Statistical models evaluate the associations between species richness, connectance, and structural robustness.

Additional analyses examine interaction categories, model diagnostics, sensitivity to analytical choices, and Monte Carlo uncertainty.

Statistical associations are interpreted as relationships within the analyzed dataset, not as evidence of causation.

## 7. Reproducibility

The project includes data-validation scripts, simulation code, statistical-analysis scripts, figures, and automated tests.

The experiment uses a documented random seed to support reproducibility.

## 8. Limitations

Structural connectivity is not equivalent to ecosystem functioning, population viability, or ecological recovery.

Simulated species removal is a computational stress test rather than a prediction of real-world extinction.

The analysis uses binary interaction matrices and does not directly model the ecological effects of interaction strength.

Differences in sampling effort, geographic scale, and network construction may affect comparisons between networks.

Monte Carlo replicates improve the precision of simulation estimates but do not increase the number of independent empirical networks.
</CodeBlock>

