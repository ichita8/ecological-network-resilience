import pandas as pd
import urllib.request
from io import StringIO
from pathlib import Path

BASE = "https://www.web-of-life.es/api/v1/networks"
OUT = Path("data/web_of_life/sample_inspection")
OUT.mkdir(parents=True, exist_ok=True)

catalog = pd.read_csv("data/web_of_life/networks.csv")

print("NETWORKS BY WEIGHTED STATUS:")
print(catalog.groupby(["interaction_type", "weighted"]).size().to_string())

print("\nREPRESENTATIVE NETWORK INSPECTION")

for category in catalog["interaction_type"].dropna().unique():
    row = catalog[catalog["interaction_type"] == category].iloc[0]
    network_id = row["network"]

    print("\n" + "=" * 65)
    print("CATEGORY:", category)
    print("NETWORK:", network_id)
    print("WEIGHTED:", row["weighted"])
    print("CATALOG SPECIES:", row["species"])
    print("CATALOG INTERACTIONS:", row["interactions"])
    print("CATALOG CONNECTANCE:", row["connectance"])
    print("AUTHOR:", row["author"])
    print("REFERENCE:", row["reference"])

    for filename in ["matrix.csv", "edges.csv", "species.csv"]:
        url = f"{BASE}/{network_id}/{filename}"
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read().decode("utf-8-sig")

            (OUT / f"{network_id}_{filename}").write_text(
                content, encoding="utf-8"
            )

            if filename == "matrix.csv":
                matrix = pd.read_csv(StringIO(content), index_col=0)
                values = pd.to_numeric(
                    matrix.stack(), errors="coerce"
                )
                print("MATRIX SHAPE:", matrix.shape)
                print("MATRIX MISSING CELLS:", int(matrix.isna().sum().sum()))
                print("MATRIX ZERO CELLS:", int((matrix == 0).sum().sum()))
                print("MATRIX NONZERO CELLS:", int((matrix != 0).sum().sum()))
                print("MATRIX UNIQUE VALUES:",
                      sorted(values.dropna().unique().tolist())[:20])

            elif filename == "edges.csv":
                edges = pd.read_csv(StringIO(content))
                print("EDGE COLUMNS:", edges.columns.tolist())
                print("EDGE RECORDS:", len(edges))
                print(edges.head(3).to_string(index=False))

            elif filename == "species.csv":
                species = pd.read_csv(StringIO(content))
                print("SPECIES COLUMNS:", species.columns.tolist())
                print("SPECIES RECORDS:", len(species))
                if "role" in species.columns:
                    print("SPECIES ROLES:")
                    print(species["role"].value_counts(dropna=False).to_string())
                print(species.head(3).to_string(index=False))

        except Exception as error:
            print("DOWNLOAD ERROR:", filename, repr(error))

print("\nINSPECTION FILES SAVED TO:", OUT)
