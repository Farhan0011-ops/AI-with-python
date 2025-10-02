import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# Loading diabetes dataset
data = load_diabetes(as_frame=True)
df = data['frame']

# Plot target distribution
plt.hist(df["target"], 25)
plt.xlabel("target")
plt.show()

# Correlation heatmap to see which variables are most correlated with target
sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

"""
Basic model with only 'bmi' and 's5'
These two are known to be strongly correlated with diabetes progression.
"""
X_base = df[['bmi', 's5']]
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X_base, y, random_state=5, test_size=0.2)
lm = LinearRegression()
lm.fit(x_train, y_train)

y_train_pred = lm.predict(x_train)
y_test_pred = lm.predict(x_test)

print("Baseline Model (bmi + s5)")
print("Train RMSE:", root_mean_squared_error(y_train, y_train_pred))
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test RMSE:", root_mean_squared_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

"""
now we are adding one more variable (e.g., 'bp' - average blood pressure.
because blood pressure is a known risk factor for diabetes complications,
and it has moderate correlation with the target.
"""
X_plus1 = df[['bmi', 's5', 'bp']]
x_train, x_test, y_train, y_test = train_test_split(X_plus1, y, random_state=5, test_size=0.2)
lm = LinearRegression()
lm.fit(x_train, y_train)

y_train_pred = lm.predict(x_train)
y_test_pred = lm.predict(x_test)

print("\nModel with bmi + s5 + bp ")
print("Train RMSE:", root_mean_squared_error(y_train, y_train_pred))
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test RMSE:", root_mean_squared_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

"""
Use ALL available explanatory variables to see if adding more helps.
"""
X_all = df.drop(columns=['target'])  # all predictors
x_train, x_test, y_train, y_test = train_test_split(X_all, y, random_state=5, test_size=0.2)
lm = LinearRegression()
lm.fit(x_train, y_train)

y_train_pred = lm.predict(x_train)
y_test_pred = lm.predict(x_test)

print("\n Model with ALL variables")
print("Train RMSE:", root_mean_squared_error(y_train, y_train_pred))
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test RMSE:", root_mean_squared_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

"""
1) Going with bmi and S5 gives us a string basline model.
2) When we added the third value which is bp it slightly improved the performance.
3)By using all variables somtimes gives better R² and lower RMSE, but not always by a huge margin.
4) Too many values can also lead us to overfitting, so model performance should always be tested on unseen data.
"""