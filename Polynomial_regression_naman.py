# Polynomial Regression

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X_naman = dataset.iloc[: , 1:2].values
Y_naman = dataset.iloc[: , 2].values

# Splitting the datasets into training and testing sets
"""from sklearn.model_selection import train_test_split
X_naman_train , X_naman_test , Y_naman_train , Y_naman_test = train_test_split(X_naman , Y_naman, test_size = 0.2 , random_state = 0)"""

# Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_naman, Y_naman)

# Fitting Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_naman)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y_naman)

# Visualizing the linear regression model
plt.scatter(X_naman, Y_naman, color = "red")
plt.plot(X_naman, lin_reg.predict(X_naman), color = "blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynomial regression model
X_grid = np.arange(min(X_naman), max(X_naman), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X_naman, Y_naman, color = "red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Graph Comparison
plt.scatter(X_naman, Y_naman, color = "red")
plt.plot(X_naman, lin_reg.predict(X_naman), color = "blue")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = "black")
plt.title("Truth or Bluff (Comparison)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Predicting the results using both models
lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

