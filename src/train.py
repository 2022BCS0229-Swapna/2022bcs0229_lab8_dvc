import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/housing.csv")
df = df.dropna()

df = pd.get_dummies(df, columns=["ocean_proximity"])

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, pred)

print("===================================")
print("Roll No: 2022BCS0229")
print("Dataset Version: Version 1")
print("Dataset Size:", len(df))
print("RMSE:", rmse)
print("R2:", r2)
print("===================================")

with open("metrics.txt", "w") as f:
    f.write("Roll No: 2022BCS0229\n")
    f.write("Dataset Version: Version 1\n")
    f.write(f"Dataset Size: {len(df)}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")