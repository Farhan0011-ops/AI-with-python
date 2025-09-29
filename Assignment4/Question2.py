import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# loading dataset
data = pd.read_csv("weight-height.csv")

X = data['Height'].values.reshape(-1, 1)
y = data['Weight'].values

# scatter plot
plt.scatter(X, y, c="orange", alpha=0.4, s=8)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height vs Weight Scatter Plot")
plt.show()

# linear regression
model = LinearRegression()
model.fit(X, y)

# predictions
x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_line = model.predict(x_line)

# plotting regression line
plt.scatter(X, y, c="lightgray", alpha=0.5, s=8)
plt.plot(x_line, y_line, color="blue", linewidth=2, label="Regression Line")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.title("Linear Regression: Height → Weight")
plt.show()

# evaluating model
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE = {rmse:.3f}")
print(f"R² = {r2:.3f}")

# This code loads height and weight data from the CSV file
# It is making a scatter plot to see the relation between weight and height
# We are training a linear regression model to predict weight and height
# It is drawing a regression line on the top of the scatter plot
# At the end it is calculating the RMSE and R2