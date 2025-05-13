print("Data Preprocessing Template")

print("\nImporting libraries...")
# Importing libraries
import numpy as np # NumPy is used for numerical operations
import pandas as pd # Pandas is used for data manipulation and analysis

# Importing dataset
print("\nLoading dataset...")
dataset = pd.read_csv('../datasets/Data.csv')
""" 
   Country   Age   Salary Purchased
0   France  44.0  72000.0        No
1    Spain  27.0  48000.0       Yes
2  Germany  30.0  54000.0        No
3    Spain  38.0  61000.0        No
4  Germany  40.0      NaN       Yes
5   France  35.0  58000.0       Yes
6    Spain   NaN  52000.0        No
7   France  48.0  79000.0       Yes
8  Germany  50.0  83000.0        No
9   France  37.0  67000.0       Yes
"""
X = dataset.iloc[:, :-1].values  # Selects all rows and all columns except the last one; Purely integer-location based indexing for selection by position
Y = dataset.iloc[:, 3].to_numpy()    # Selects all rows but only column index 3
print("Original Dataset:")
print("X: ", X)
print("Y: ", Y)

print("\nData before preprocessing:")
print("X before preprocessing: ", X)
print("Y before preprocessing: ", Y)
print({type(X): "X is a NumPy array", type(Y): "Y is a NumPy array"})
# Handling missing data
from sklearn.impute import SimpleImputer # Importing SimpleImputer for handling missing values
# The SimpleImputer class is used to handle missing values in the dataset.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3]) # Fit the imputer to the data
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Transform the data
print("Data after handling missing values:")
print("X after imputation: ", X)

print("\nEncoding categorical data...")
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit(X[:, 0])  # Learn the categories
X[:, 0] = labelencoder_X.transform(X[:, 0])  # Apply the transformation
print("X after encoding: ", X)

# Creating dummy variables for categorical data
# Better Alternative: For categorical variables without inherent order (like countries), one-hot encoding is often preferred.
# Example of one-hot encoding result
""" France:  [1, 0, 0]
    Germany: [0, 1, 0]
    Spain:   [0, 0, 1]
"""
# Updated OneHotEncoder without deprecated parameter
onehotencoder = OneHotEncoder()
# Apply one-hot encoding to the first column only (reshaped as required)
X_encoded = onehotencoder.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
# More explicit but equivalent
# X[:, 0].reshape(len(X), 1)
# Concatenate the one-hot encoded features with the original features (excluding first column)
# The reshape(-1, 1) operation converts a 1D array into a 2D array (column vector) which is often required by scikit-learn estimators.
X = np.concatenate((X_encoded, X[:, 1:]), axis=1)

""" print("X after one-hot encoding: ")
# Example output of one-hot encoding; X_encoded will be a 2D array with shape (10, 3)
# The first column will be the one-hot encoded version of the 'Country' column. 
[[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
 """
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("X after one-hot encoding: ", X)
print("Y after encoding: ", Y)

print("\nSplitting the dataset into the training set and test set...")
# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# The test_size parameter specifies the proportion of the dataset to include in the test split (20% in this case).
# The random_state parameter ensures reproducibility of the split.
print("Data after splitting by inspecting its shape:")
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("Y_train: ", Y_train.shape)
print("Y_test: ", Y_test.shape)

print("\nFeature scaling...")
# Feature scaling
""" Why It's Important
Algorithm Performance: Many ML algorithms perform better with scaled features
Convergence: Helps gradient descent converge faster
Feature Importance: Prevents features with larger scales from dominating
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Fit the scaler to the training data and transform it    
X_test = sc_X.transform(X_test) # Transform the test data using the same scaler
# Note: The StandardScaler is used to standardize features by removing the mean and scaling to unit variance.
# Never use fit_transform() on test data to avoid data leakage.

print("\nData preprocessing completed.")
print("X_train: ", X_train)
print("X_test: ", X_test)
print("Y_train: ", Y_train)
print("Y_test: ", Y_test)
print("Data Preprocessing Template Completed")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression  # Importing a common classifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit the pipeline to the training data
pipeline.fit(X_train, Y_train)
# Make predictions on the test data
predictions = pipeline.predict(X_test)
print("Predictions of: ", predictions)
print("Pipeline completed.")

# Note: The pipeline allows you to chain together multiple processing steps and a final estimator, making it easier to manage the workflow.


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(Y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Generate detailed classification report
print("\nClassification Report:")
print(classification_report(Y_test, predictions))

# Create confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, predictions))
# The confusion matrix provides a summary of the prediction results on a classification problem.

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Create pipelines with different classifiers
pipelines = {
    'logistic': Pipeline([('scaler', StandardScaler()),
                         ('classifier', LogisticRegression())]),
    'random_forest': Pipeline([('scaler', StandardScaler()),
                             ('classifier', RandomForestClassifier())]),
    'svm': Pipeline([('scaler', StandardScaler()),
                    ('classifier', SVC())])
}

# Compare models
for name, pipe in pipelines.items():
    cv_scores = cross_val_score(pipe, X_train, Y_train, cv=5)
    print(f"\n{name} - Average CV Score: {cv_scores.mean():.2f}")

# The above code demonstrates how to preprocess data, train a model, and evaluate its performance using cross-validation.


from sklearn.model_selection import GridSearchCV

# Example for LogisticRegression
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, Y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)


import matplotlib.pyplot as plt
import seaborn as sns # For better visualization

# Confusion matrix visualization
cm = confusion_matrix(Y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()