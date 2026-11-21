import pandas as pd

df = pd.read_csv("data/web_of_life/networks.csv")

print("COLUMNS:")
print(df.columns.tolist())

print("\nFIRST 10 NETWORK IDENTIFIERS:")
print(df[["network", "interaction_type", "weighted"]].head(10).to_string(index=False))

print("\nIDENTIFIER CHECK:")
print("Total records:", len(df))
print("Unique identifiers:", df["network"].nunique())
print("Missing identifiers:", df["network"].isna().sum())
