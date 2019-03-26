# Simple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")
X_naman = dataset.iloc[: , :-1].values
Y_naman = dataset.iloc[: , 1].values

from sklearn.model_selection import train_test_split
X_naman_train , X_naman_test , Y_naman_train , Y_naman_test = train_test_split(X_naman , Y_naman , test_size = 1/3 , random_state = 0)

# Fitting Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_naman_train , Y_naman_train)

# Predicting the Test Set Result

Y_naman_pred = regressor.predict(X_naman_test)

# Plotting the Training Set

plt.scatter(X_naman_train , Y_naman_train , color = "black")
plt.plot(X_naman_train , regressor.predict(X_naman_train) , color = "yellow")
plt.title("Salary vs Experience(Training Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Plotting the Test Set

plt.scatter(X_naman_test , Y_naman_test , color = "black")
plt.plot(X_naman_train , regressor.predict(X_naman_train) , color = "yellow")
plt.title("Salary vs Experience(Test Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()