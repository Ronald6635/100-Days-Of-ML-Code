"""
Logistic Regression Module

This module demonstrates the implementation of a Logistic Regression model
to predict purchases based on age and estimated salary from the Social Network Ads dataset.

Key features:
- Data loading and pre-processing with feature scaling
- Training a Logistic Regression classifier
- Model evaluation with confusion matrix
- Performance metrics calculation (accuracy, precision, recall)
- Visualization of results with 3D confusion matrix and decision boundaries

This implementation follows a linear workflow: data preprocessing → model training → 
prediction → evaluation → visualization.
"""

# Imports grouped logically
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# =============================================================================
# DATA PRE-PROCESSING
# =============================================================================

# Importing the dataset
dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
""" 
      User ID  Gender  Age  EstimatedSalary  Purchased
0    15624510    Male   19            19000          0
1    15810944    Male   35            20000          0
2    15668575  Female   26            43000          0
... and so on
"""

# Selecting features (independent variables) and target (dependent variable)
X = dataset.iloc[:, [2, 3]].to_numpy()  # Features: 'Age' and 'EstimatedSalary'
Y = dataset.iloc[:, 4].to_numpy()  # Target: 'Purchased'

# Splitting the dataset into the Training set and Test set
# NOTE: We use 25% of data for testing to ensure a good evaluation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
# NOTE: Feature scaling is important for logistic regression with regularization
# and ensures faster convergence of the optimization algorithm
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit on training data and transform it
X_test = sc.transform(X_test)  # Transform test data using same scaling

# Use Scatter plot to visualize the scaled features
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Training set')
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', label='Test set')
plt.title('Feature Scaling Visualization')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.legend()
plt.show()

# =============================================================================
# LOGISTIC REGRESSION MODEL TRAINING
# =============================================================================

# CRITICAL: We must fit only on training data to avoid data leakage
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# =============================================================================
# PREDICTION ON TEST SET
# =============================================================================

# Generate predictions on previously unseen test data
Y_pred = classifier.predict(X_test)

# =============================================================================
# EVALUATING THE PREDICTION
# =============================================================================

# Making the Confusion Matrix
cm = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
print("Confusion Matrix:\n", cm)

# EXAMPLE: The confusion matrix is interpreted as follows:
# cm[0,0]: True Negatives (TN) - Correctly predicted non-purchases
# cm[0,1]: False Positives (FP) - Incorrectly predicted purchases (Type I error)
# cm[1,0]: False Negatives (FN) - Incorrectly predicted non-purchases (Type II error)
# cm[1,1]: True Positives (TP) - Correctly predicted purchases

# =============================================================================
# VISUALIZATION OF RESULTS
# =============================================================================

# Import 3D plotting capabilities
from mpl_toolkits.mplot3d import Axes3D  # NOTE: Required for 3D projection

# Calculate performance metrics from confusion matrix values
tn, fp, fn, tp = cm.ravel()  # Flatten the matrix into a 1D array
accuracy = (tn + tp) / (tn + fp + fn + tp)  # Overall accuracy
precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision: when model predicts positive, how often is it correct
recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall: how many actual positives were identified correctly

# Create 3D visualization of confusion matrix
fig = plt.figure(figsize=(12, 9)) # MODIFIED: Increased figure size for better spacing
ax = fig.add_subplot(111, projection='3d')

# Set up positions for the bars in the 3D plot
x_pos = [0, 1]
y_pos = [0, 1]
x_pos, y_pos = np.meshgrid(x_pos, y_pos)  # Create all combinations of x,y positions
x_pos = x_pos.flatten()  # Convert to 1D arrays
y_pos = y_pos.flatten()
z_pos = np.zeros_like(x_pos)  # Base of bars at z=0
dx = dy = 0.75  # Width and depth of bars
dz = cm.flatten()  # Height of bars = confusion matrix values

# Color bars based on correctness: green for correct predictions, red for errors
colors = ['red' if (i == 0 and j == 1) or (i == 1 and j == 0) else 'green' 
          for i, j in zip(x_pos, y_pos)]

# Create the 3D bar plot
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, alpha=0.8)

# Add axis labels and title
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_zlabel('Count')
ax.set_title('3D Confusion Matrix')

# Set custom tick positions and labels
ax.set_xticks([0.4, 1.4])
ax.set_yticks([0.4, 1.4])
ax.set_xticklabels(['Not Purchased', 'Purchased'])
ax.set_yticklabels(['Not Purchased', 'Purchased'])

# Add text annotations for each bar in the 3D plot
for i, v in enumerate(cm.flatten()):
    ax.text(x_pos[i], y_pos[i], v + 0.1, f'{v}', 
            horizontalalignment='center', verticalalignment='bottom')

# Add a text box with performance metrics
ax.text2D(0.05, 0.95, f'Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}', 
          transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

fig.tight_layout() # MODIFIED: Called tight_layout on the figure object
plt.show()  # Display the 3D confusion matrix

# =============================================================================
# DECISION BOUNDARY VISUALIZATION
# =============================================================================

# Create a figure with two subplots for training and test data
plt.figure(figsize=(12, 6))

# Subplot 1: Visualizing the Training set results
plt.subplot(1, 2, 1)
X_set, Y_set = X_train, Y_train

# Create a meshgrid covering the feature space for decision boundary visualization
# NOTE: We extend 1 unit beyond the min/max to ensure full coverage
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

# Use the classifier to predict class labels across the entire mesh
# CRITICAL: This creates the colored background showing decision regions
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

# Set plot boundaries
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot the actual data points with colors representing their classes
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), label=j) # Changed c to color

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.legend()

# Subplot 2: Visualizing the Test set results
plt.subplot(1, 2, 2)
X_set, Y_set = X_test, Y_test

# Create meshgrid for test data
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

# Plot decision boundary for test data
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot the actual data points with colors representing their classes
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), label=j) # Changed c to color

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.legend()

# Prevent overlapping elements by adjusting layout
plt.tight_layout()
plt.show()