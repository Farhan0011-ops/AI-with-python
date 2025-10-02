import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

# 0) Read dataset
# Note: The file usually uses ',' as delimiter, but we ensure correct reading
df = pd.read_csv("50_Startups.csv", delimiter=",")
print("First few rows of dataset:\n", df.head(), "\n")

# 1) Identify the variables
print("Columns in dataset:", df.columns.tolist())

# 2) Investigate correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True).round(2), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3) Choose explanatory variables

# Encode categorical "State"
df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)

# 4) Plot explanatory variables vs Profit
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(df["R&D Spend"], df["Profit"])
plt.xlabel("R&D Spend")
plt.ylabel("Profit")

plt.subplot(1,3,2)
plt.scatter(df["Administration"], df["Profit"])
plt.xlabel("Administration")
plt.ylabel("Profit")

plt.subplot(1,3,3)
plt.scatter(df["Marketing Spend"], df["Profit"])
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")

plt.tight_layout()
plt.show()

"""
Plots show:
- R&D Spend has clear linear dependence with Profit.
- Marketing Spend shows weaker linear dependence.
- Administration appears to have little/no direct relationship.
"""

# 5) Train-test split (80/20)
X = df_encoded.drop(columns=["Profit"])
y = df_encoded["Profit"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6) Train linear regression
lm = LinearRegression()
lm.fit(x_train, y_train)

# 7) Compute performance metrics
y_train_pred = lm.predict(x_train)
y_test_pred = lm.predict(x_test)

print("Training Performance")
print("Train RMSE:", root_mean_squared_error(y_train, y_train_pred))
print("Train R²:", r2_score(y_train, y_train_pred))

print("\nTesting Performance")
print("Test RMSE:", root_mean_squared_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

""" 
                      Answers
1) R&D spend is the most important predictor for profit.
2) When i have added the marketing spend and administration it helped but not helped much.
3) State variables have minimal effect once R&D and Marketing are included.
4) The Linear Regression model achieved high R², which showed  good explanatory power.
5) Train and Test R² values are close, indicating the model generalizes well.

"""