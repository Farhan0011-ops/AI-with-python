import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Reading dataset
df = pd.read_csv('Auto.csv')
#print(df.head())

# Features and target variable
X = df[['cylinders','displacement','horsepower','weight','year','acceleration']]
y = df['mpg']

# Splitting dataset into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Defining alpha values to test
alphas = np.linspace(0.001, 120, 800)

# Lists for storing results
r2ValuesRidge = []
r2ValuesLasso = []

# Training with different alpha values
for alp in alphas:
    # Ridge regression
    rr = linear_model.Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    r2_testRidge = r2_score(y_test, rr.predict(X_test))
    r2ValuesRidge.append(r2_testRidge)

    # Lasso regression
    lr = linear_model.Lasso(alpha=alp)
    lr.fit(X_train, y_train)
    r2_testLasso = r2_score(y_test, lr.predict(X_test))
    r2ValuesLasso.append(r2_testLasso)

# Finding best alpha and R² for Ridge
best_r2Ridge = max(r2ValuesRidge)
idxRidge = r2ValuesRidge.index(best_r2Ridge)
best_alphaRidge = alphas[idxRidge]

# Finding best alpha and R² for Lasso
best_r2Lasso = max(r2ValuesLasso)
idxLasso = r2ValuesLasso.index(best_r2Lasso)
best_alphaLasso = alphas[idxLasso]

# Printing results
print('--------------------------------------------')
print('           Values     Ridge         Lasso')
print('--------------------------------------------')
print(f'Best R²   : {best_r2Ridge:10.4f}   {best_r2Lasso:12.4f}')
print(f'Best alpha: {best_alphaRidge:8.4f}   {best_alphaLasso:11.4f}')

# Plotting alpha vs R² for Ridge
plt.subplot(1,2,1)
plt.plot(alphas, r2ValuesRidge)
plt.title('Ridge regression')
plt.xlabel('alpha')
plt.ylabel('R²')

# Plotting alpha vs R² for Lasso
plt.subplot(1,2,2)
plt.plot(alphas, r2ValuesLasso)
plt.title('Lasso regression')
plt.xlabel('alpha')
plt.ylabel('R²')

plt.show()

"""
              Findings 
1) Ridge regression works well for handling multicollinearity and shrinking coefficients smoothly.
2) Lasso regression performs both regularization and feature selection, which made it perform slightly 
better on the Auto dataset.
3)Choosing the correct alpha is crucial, and the optimal alpha values were found by testing multiple
candidates and comparing their R² scores.
"""