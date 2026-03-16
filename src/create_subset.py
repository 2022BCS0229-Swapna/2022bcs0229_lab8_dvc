import pandas as pd

df = pd.read_csv("data/housing.csv")

df_subset = df.head(5000)

df_subset.to_csv("data/housing.csv", index=False)