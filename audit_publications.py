import pandas as pd

d = pd.read_csv("data/web_of_life/networks.csv")

# Exclude food webs from the primary bipartite sample
b = d[d["interaction_type"] != "Food Webs"].copy()

print("TOTAL NETWORKS:", len(d))
print("PRIMARY BIPARTITE CANDIDATES:", len(b))
print("EXCLUDED FOOD WEBS:", len(d) - len(b))

print("\nNETWORKS BY CATEGORY:")
print(b.groupby("interaction_type").size().to_string())

print("\nDISTINCT REFERENCES:")
print("All networks:", d["reference"].nunique())
print("Bipartite candidates:", b["reference"].nunique())

print("\nDISTINCT AUTHORS:")
print("Bipartite candidates:", b["author"].nunique())

print("\nNETWORKS PER REFERENCE:")
counts = b.groupby("reference").size().sort_values(ascending=False)
print(counts.head(15).to_string())

print("\nWEIGHTED STATUS:")
print(b.groupby(["interaction_type", "weighted"]).size().to_string())

print("\nMISSING REFERENCE VALUES:")
print("References:", b["reference"].isna().sum())
print("Authors:", b["author"].isna().sum())
