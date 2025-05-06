print("Data Preprocessing Template")

print("\nImporting libraries...")
# Importing libraries
import numpy as np
import pandas as pd

# Importing dataset
print("\nLoading dataset...")
dataset = pd.read_csv('../datasets/Data.csv')
X = dataset.iloc[:, :-1].values  # Selects all rows and all columns except the last one
Y = dataset.iloc[:, 3].values    # Selects all rows but only column index 3
print("Original Dataset:")
print("X: ", X)
print("Y: ", Y)

print("\nData before preprocessing:")
print("X before preprocessing: ", X)
print("Y before preprocessing: ", Y)
# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3]) # Fit the imputer to the data
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Transform the data
print("Data after handling missing values:")
print("X after imputation: ", X)

print("\nEncoding categorical data...")
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # Encode the country column
print("X after encoding: ", X)

# Creating dummy variables for categorical data
# Updated OneHotEncoder without deprecated parameter
onehotencoder = OneHotEncoder()
# Apply one-hot encoding to the first column only (reshaped as required)
X_encoded = onehotencoder.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
# Concatenate the one-hot encoded features with the original features (excluding first column)
X = np.concatenate((X_encoded, X[:, 1:]), axis=1)
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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Fit the scaler to the training data and transform it    
X_test = sc_X.transform(X_test) # Transform the test data using the same scaler
# Note: The StandardScaler is used to standardize features by removing the mean and scaling to unit variance.

print("\nData preprocessing completed.")
print("X_train: ", X_train)
print("X_test: ", X_test)
print("Y_train: ", Y_train)
print("Y_test: ", Y_test)
print("Data Preprocessing Template Completed")