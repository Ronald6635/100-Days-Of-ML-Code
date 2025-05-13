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

# Add after model fitting section
print(f"Model Coefficient (Slope): {regressor.coef_[0]:.4f}")
print(f"Model Intercept: {regressor.intercept_:.4f}")
print(f"This means: Score = {regressor.coef_[0]:.4f} × Hours + {regressor.intercept_:.4f}")

# Step 3: Predicting the Result
Y_pred = regressor.predict(X_test) # Predicting the test set results
# The predict method uses the fitted model to make predictions on the test data.
 
 # Add after predictions
from sklearn.metrics import r2_score, mean_absolute_error

# Training metrics
train_pred = regressor.predict(X_train)
print("\nTraining Set Metrics:")
print(f"R² Score: {r2_score(Y_train, train_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(Y_train, train_pred):.4f}")
print(f"Mean Squared Error: {np.mean((Y_train - train_pred)**2):.4f}")

# Test metrics
print("\nTest Set Metrics:")
print(f"R² Score: {r2_score(Y_test, Y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred):.4f}")
print(f"Mean Squared Error: {np.mean((Y_test - Y_pred)**2):.4f}")

# Step 4: Visualization
## Visualising the Training results
plt.scatter(X_train, Y_train, color='red', label='Actual')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted')
plt.legend()
plt.title(f'Hours vs Scores (Training set), MSE = {np.mean((Y_train - regressor.predict(X_train))**2):.2f}')
plt.xlabel('Hours of Study')
plt.ylabel('Scores')
plt.show()

## Visualizing the test results
plt.scatter(X_test , Y_test, color = 'red', label='Actual')
plt.plot(X_test , Y_pred, color ='blue', label='Predicted')
plt.title(f'Hours vs Scores (Test set), MSE = {np.mean((Y_test - Y_pred)**2):.2f}')
plt.xlabel('Hours of Study')
plt.ylabel('Scores')
plt.legend()
plt.show()

# Add a section for making new predictions
# Example: Predict score for a student studying for 9.5 hours
new_hours = np.array([[9.5]])
predicted_score = regressor.predict(new_hours)
print(f"\nA student studying for {new_hours[0][0]} hours is predicted to score: {predicted_score[0]:.2f}")

# Create a combined plot to compare training and test results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, Y_train, color='red', label='Training data')
plt.scatter(X_test, Y_test, color='green', label='Test data')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression line')
plt.title('Hours vs Scores (Combined Training & Test data)')
plt.xlabel('Hours of Study')
plt.ylabel('Scores')
plt.legend()
plt.savefig('linear_regression_combined.png')  # Save the visualization
plt.show()


# Add code to check residuals
residuals = Y_train - regressor.predict(X_train)

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(regressor.predict(X_train), residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Check normality of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=15)
plt.title('Histogram of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# Save the trained model for later use
import pickle

# Save the model
with open('study_score_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

# Example of how to load the model
with open('study_score_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Verify it works
print(f"Original model prediction for 9.5 hours: {regressor.predict(np.array([[9.5]]))[0]:.2f}")
print(f"Loaded model prediction for 9.5 hours: {loaded_model.predict(np.array([[9.5]]))[0]:.2f}")