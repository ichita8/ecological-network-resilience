# Methods — working draft

## Scope of inference

The initial outcome is structural interaction persistence in recorded ecological networks. It is not a direct measure of ecosystem functioning or population persistence.

## Network representation

Each interaction record is represented as a directed edge from `source` to `target`. A directed multigraph is used so that multiple recorded interaction types between the same pair of species are not collapsed.

## Perturbation scenarios

- Random removal: species are removed in a randomly shuffled order. The process is repeated using a documented random seed.
- Targeted removal: at each step, the currently highest-total-degree species is removed. Ties are resolved alphabetically by species identifier.

## Primary outcome

At each removal step, interaction retention is the number of remaining recorded interaction edges divided by the original number of edges.

The robustness AUC is the trapezoidal area under interaction retention versus fraction of species removed, over the interval from zero to complete removal. Values range from 0 to 1.

## Interaction diversity

Interaction-type richness is the number of distinct labels. Shannon entropy is calculated from the proportions of interaction records in each type.

## Important limitations to address before empirical inference

Network sampling, species count, interaction count, connectance, taxonomic scope, and interaction coding may affect comparisons. The empirical analysis plan must account for these factors. The targeted-removal scenario is a structural stress test, not a claim that real ecological losses always follow degree order.
