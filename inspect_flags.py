import pandas as pd

p = "results/web_of_life/"

df = pd.read_csv(p + "validated_networks.csv")

ids = ["M_PL_021", "M_PL_054", "M_PL_055"]

columns = [
    "network",
    "species_richness",
    "observed_interactions",
    "possible_interactions",
    "bipartite_connectance",
    "catalog_connectance",
    "validation_flags"
]

print(df[df["network"].isin(ids)][columns].to_string(index=False))
