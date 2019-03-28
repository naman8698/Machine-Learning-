# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X_naman = dataset.iloc[: , :-1].values
Y_naman = dataset.iloc[: , 4].values

# Encoding the categorical features
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X_naman = LabelEncoder()
X_naman[: , 3] = labelencoder_X_naman.fit_transform(X_naman[: , 3])

from sklearn.compose import make_column_transformer
colT = make_column_transformer((OneHotEncoder(categories='auto'),[3]), remainder='passthrough')
X_naman = colT.fit_transform(X_naman)

# Avoiding Dummy Variable
X_naman = X_naman[: , 1:]

df = pd.DataFrame(X_naman)

# Splitting into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_naman_train , X_naman_test , Y_naman_train , Y_naman_test = train_test_split(X_naman , Y_naman , test_size = 0.2 , random_state = 0)

# Fitting Multiple Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_naman_train , Y_naman_train)

# Predicting the Test Set Results
Y_pred = regressor.predict(X_naman_test)

# Building optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_naman = np.append(arr = np.ones((50 , 1)).astype(int) , values = X_naman , axis = 1)
X_opt = X_naman[: , [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype = float)
regressor_OLS = sm.OLS(Y_naman, X_opt).fit()
regressor_OLS.summary()

X_opt = X_naman[: , [0, 1, 3, 4, 5]]
X_opt = np.array(X_opt, dtype = float)
regressor_OLS = sm.OLS(Y_naman, X_opt).fit()
regressor_OLS.summary()

X_opt = X_naman[: , [0, 3, 4, 5]]
X_opt = np.array(X_opt, dtype = float)
regressor_OLS = sm.OLS(Y_naman, X_opt).fit()
regressor_OLS.summary()

X_opt = X_naman[: , [0, 3, 5]]
X_opt = np.array(X_opt, dtype = float)
regressor_OLS = sm.OLS(Y_naman, X_opt).fit()
regressor_OLS.summary()

X_opt = X_naman[: , [0, 3]]
X_opt = np.array(X_opt, dtype = float)
regressor_OLS = sm.OLS(Y_naman, X_opt).fit()
regressor_OLS.summary()

# Automated Backward Elimination
"""import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)"""
