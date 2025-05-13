# Simple Linear Regression

# Simple Linear Regression is a statistical method that models the relationship between a dependent variable and one independent variable by fitting a linear equation to observed data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/studentscores.csv')
""" print("Original Dataset:\n", dataset) 
    Hours  Scores
0     2.5      21
1     5.1      47
2     3.2      27
3     8.5      75
4     3.5      30
5     1.5      20
"""
X = dataset.iloc[ : ,   : 1 ].values # Hours
Y = dataset.iloc[ : , 1 ].values # Scores
# The independent variable (often called x) is the input variable that you use to make predictions
# The dependent variable (often called y) is the outcome or target variable you're trying to predict
print("Original Dataset:\n", dataset)

# Step 1: Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 

# Step 2: Fitting Simple Linear Regression Model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train) # Fit the model to the training data
# The fit method estimates the coefficients of the linear regression model using the training data.

# Step 3: Predicting the Result
Y_pred = regressor.predict(X_test) # Predicting the test set results
# The predict method uses the fitted model to make predictions on the test data.
 
# Step 4: Visualization 
## Visualising the Training results
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.title(f'Hours vs Scores (Training set), MSE = {np.mean((Y_train - regressor.predict(X_train))**2):.2f}')
plt.xlabel('Hours of Study')
plt.ylabel('Scores')
plt.legend(['Actual', 'Predicted'])
plt.show()

## Visualizing the test results
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , Y_pred, color ='blue')
plt.title(f'Hours vs Scores (Test set), MSE = {np.mean((Y_test - Y_pred)**2):.2f}')
plt.xlabel('Hours of Study')
plt.ylabel('Scores')
plt.legend(['Actual', 'Predicted'])
plt.show()
