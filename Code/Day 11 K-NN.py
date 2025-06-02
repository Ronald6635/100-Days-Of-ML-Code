"""
K-Nearest Neighbors (K-NN) Module

This module demonstrates the implementation of the K-Nearest Neighbors (K-NN)
algorithm for classification. It uses the Social Network Ads dataset to predict
whether a user will purchase a product based on their age and estimated salary.

Key features:
- Loads and preprocesses the dataset.
- Splits the dataset into training and testing sets.
- Applies feature scaling to normalize data.
- Trains a K-NN classifier.
- Predicts results on the test set.
- Evaluates the model using a confusion matrix.
"""

# Imports grouped logically
import numpy as np
import matplotlib.pyplot as plt # NOTE: Not used in the script directly, but often useful for visualization
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# =============================================================================
# IMPORTING THE DATASET
# =============================================================================
# Load the dataset from a CSV file
dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
""" 
      User ID  Gender  Age  EstimatedSalary  Purchased
0    15624510    Male   19            19000          0
1    15810944    Male   35            20000          0
2    15668575  Female   26            43000          0
... and so on
"""
# Extracting features (Age, EstimatedSalary)
X = dataset.iloc[:, [2, 3]].values
# Extracting the target variable (Purchased)
y = dataset.iloc[:, 4].values

# =============================================================================
# SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
# =============================================================================
# Splitting data: 75% for training, 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# =============================================================================
# FEATURE SCALING
# =============================================================================
# Initialize the StandardScaler
sc = StandardScaler()
# Fit the scaler on the training data and transform it
X_train = sc.fit_transform(X_train)
# Transform the test data using the fitted scaler
X_test = sc.transform(X_test)

# =============================================================================
# FITTING K-NN TO THE TRAINING SET
# =============================================================================
# Initialize the K-NN classifier
# n_neighbors=5: Use 5 nearest neighbors
# metric='minkowski', p=2: Use Euclidean distance
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# Train the classifier on the training data
classifier.fit(X_train, y_train)

# =============================================================================
# PREDICTING THE TEST SET RESULTS
# =============================================================================
# Predict the outcomes for the test set
y_pred = classifier.predict(X_test)

# =============================================================================
# MAKING THE CONFUSION MATRIX
# =============================================================================
# Generate the confusion matrix to evaluate model performance
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix to see the results
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {(cm[0,0] + cm[1,1]) / sum(sum(cm)) * 100:.2f}%")

# =============================================================================
# VISUALIZATION OF RESULTS
# =============================================================================
# Visualizing the training set results
from matplotlib.colors import ListedColormap

# Create a figure with two subplots for training and test data
plt.figure(figsize=(12, 6))

# Subplot 1: Visualizing the Training set results
plt.subplot(1, 2, 1)

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                                stop = X_set[:, 0].max() + 1,
                                step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                                stop = X_set[:, 1].max() + 1,
                                step = 0.01))
# Define colormap once
cmap = ListedColormap(('red', 'green'))

# Use in contourf and scatter plots
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
         alpha = 0.75, cmap = cmap)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=cmap(i), label=j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Subplot 2: Visualizing the Test set results
plt.subplot(1, 2, 2)
# Create a meshgrid for the test set
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                                stop = X_set[:, 0].max() + 1,
                                step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                                stop = X_set[:, 1].max() + 1,
                                step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
         alpha = 0.75, cmap = cmap)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=cmap(i), label=j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# The above code visualizes the decision boundary of the K-NN classifier on both the training and test sets.
# The decision boundary is created by predicting the class for each point in the meshgrid and plotting it.
# The points are colored based on their predicted class, and the actual data points are overlaid with their true labels.

# =============================================================================
# END OF K-NN MODULE