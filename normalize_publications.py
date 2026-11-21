import pandas as pd
import re
from pathlib import Path

catalog_path = Path("data/web_of_life/networks.csv")

if not catalog_path.exists():
    raise FileNotFoundError(f"Catalog not found: {catalog_path}")

d = pd.read_csv(catalog_path)

required = {"interaction_type", "reference", "author"}
missing = required - set(d.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

b = d[d["interaction_type"] != "Food Webs"].copy()

def normalize_reference(value):
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[.,;:\s]+$", "", value)
    return value

b["NormalizedReference"] = b["reference"].map(normalize_reference)

print("=" * 65)
print("WEB OF LIFE PUBLICATION AUDIT")
print("=" * 65)

print("\nNETWORKS:", len(b))
print("RAW REFERENCE STRINGS:", b["reference"].nunique())
print("NORMALIZED REFERENCE GROUPS:", b["NormalizedReference"].nunique())

print("\nPOSSIBLE DUPLICATE REFERENCE STRINGS:")

duplicates_found = False

for normalized, group in b.groupby("NormalizedReference"):
    originals = sorted(group["reference"].dropna().unique())

    if len(originals) > 1:
        duplicates_found = True
        print("\nNORMALIZED:", normalized)
        for original in originals:
            print("  ", repr(original))

if not duplicates_found:
    print("None found through basic punctuation normalization.")

print("\nNETWORKS PER NORMALIZED REFERENCE:")
print(
    b.groupby("NormalizedReference")
    .size()
    .sort_values(ascending=False)
    .head(25)
    .to_string()
)

print("\nINTERACTION CATEGORY COUNTS:")
print(b["interaction_type"].value_counts().to_string())

print("\nAUDIT COMPLETE")
