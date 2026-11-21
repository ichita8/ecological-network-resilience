import pandas as pd
from pathlib import Path

folder = Path("data/web_of_life/sample_inspection")

for network in [
    "A_HP_001", "A_PH_004", "FW_001",
    "M_AF_001", "M_PA_001", "M_PL_001", "M_SD_001"
]:
    matrix = pd.read_csv(folder / f"{network}_matrix.csv", index_col=0)
    species = pd.read_csv(folder / f"{network}_species.csv")
    edges = pd.read_csv(folder / f"{network}_edges.csv")

    row_species = set(matrix.index.astype(str))
    column_species = set(matrix.columns.astype(str))
    listed_species = set(species["species"].astype(str))

    print("\n" + "=" * 55)
    print("NETWORK:", network)
    print("Matrix rows:", len(row_species))
    print("Matrix columns:", len(column_species))
    print("Matrix total species:", len(row_species | column_species))
    print("Species file records:", len(species))
    print("Unique species file names:", len(listed_species))
    print("Matrix labels missing from species file:",
          len((row_species | column_species) - listed_species))
    print("Species file entries absent from matrix:",
          len(listed_species - (row_species | column_species)))

    print("Species side counts:")
    print(species["side"].value_counts(dropna=False).to_string())

    print("Matrix nonzero cells:", int((matrix != 0).sum().sum()))
    print("Edge records:", len(edges))
    print("Calculated matrix connectance:",
          round((matrix != 0).sum().sum() / matrix.size, 6))
    print("Catalog species:", int(
        pd.read_csv("data/web_of_life/networks.csv")
        .set_index("network").loc[network, "species"]
    ))
    print("Catalog connectance:", pd.read_csv(
        "data/web_of_life/networks.csv"
    ).set_index("network").loc[network, "connectance"])
