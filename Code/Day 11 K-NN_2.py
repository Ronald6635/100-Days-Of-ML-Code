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
# Extracting features (Gender, Age, EstimatedSalary)
X = dataset.iloc[:, 1:4].values
# Extracting the target variable (Purchased)
y = dataset.iloc[:, 4].values

### Encoding Categorical data
# The Gender column is categorical and needs to be converted into numerical values for the model.
from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
X[: , 0] = gender_encoder.fit_transform(X[ : , 0])

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
def visualize_knn_decision_boundary(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray,
    classifier: KNeighborsClassifier
) -> None:
    """
    Visualize K-NN classifier decision boundaries for training and test sets.
    
    Creates side-by-side visualizations of the K-NN decision boundaries using
    the age and salary features. For 3D data (with encoded gender), the gender
    feature is held at its median value for boundary visualization.
    
    Args:
        X_train: Training feature data (scaled), shape (n_samples, n_features)
        y_train: Training target labels, shape (n_samples,)
        X_test: Test feature data (scaled), shape (n_samples, n_features)  
        y_test: Test target labels, shape (n_samples,)
        classifier: Trained KNeighborsClassifier instance
        
    Note:
        For datasets with 3 features (Gender, Age, Salary), only Age and Salary
        are visualized in 2D plots. Gender is held at its median value for
        decision boundary generation.
    """
    from matplotlib.colors import ListedColormap
    
    # Determine feature indices and labels based on data dimensions
    if X_train.shape[1] == 3:  # With encoded gender feature
        feature_indices = [1, 2]  # Age and Salary indices (skip gender at index 0)
        xlabel, ylabel = 'Age (Scaled)', 'Estimated Salary (Scaled)'
        median_gender = np.median(X_train[:, 0])  # Median encoded gender value
    else:  # Original 2D data
        feature_indices = [0, 1]  # Age and Salary indices
        xlabel, ylabel = 'Age (Scaled)', 'Estimated Salary (Scaled)'
        median_gender = None
    
    # Create figure with two subplots for training and test data
    plt.figure(figsize=(12, 6))
    
    # Define colormap for consistent visualization across both subplots
    cmap = ListedColormap(('red', 'green'))
    
    # =============================================================================
    # SUBPLOT 1: TRAINING SET VISUALIZATION
    # =============================================================================
    plt.subplot(1, 2, 1)
    
    # Extract visualization features from training data
    X_vis_train = X_train[:, feature_indices]
    
    # Create meshgrid for decision boundary visualization
    # NOTE: Fine step size (0.01) provides smooth decision boundary visualization
    X1, X2 = np.meshgrid(
        np.arange(start=X_vis_train[:, 0].min() - 1, 
                  stop=X_vis_train[:, 0].max() + 1, 
                  step=0.01),
        np.arange(start=X_vis_train[:, 1].min() - 1, 
                  stop=X_vis_train[:, 1].max() + 1, 
                  step=0.01)
    )
    
    # Prepare meshgrid points for classifier prediction
    if X_train.shape[1] == 3:
        # For 3D data: include gender feature at median value
        # CRITICAL: Must match the classifier's expected input dimensionality
        mesh_points = np.column_stack([
            np.full(X1.ravel().shape, median_gender),  # Gender at median
            X1.ravel(),  # Age values from meshgrid
            X2.ravel()   # Salary values from meshgrid
        ])
    else:
        # For 2D data: use meshgrid directly
        mesh_points = np.array([X1.ravel(), X2.ravel()]).T
    
    # Generate decision boundary by predicting class for each meshgrid point
    decision_boundary = classifier.predict(mesh_points).reshape(X1.shape)
    
    # Plot decision boundary using filled contours
    plt.contourf(X1, X2, decision_boundary, alpha=0.75, cmap=cmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    # Plot actual training data points with class-based coloring
    for i, class_label in enumerate(np.unique(y_train)):
        class_mask = y_train == class_label
        plt.scatter(X_vis_train[class_mask, 0], X_vis_train[class_mask, 1],
                   color=cmap(i), label=f'Class {class_label}', 
                   edgecolors='black', s=50, alpha=0.8)
    
    plt.title('K-NN Decision Boundary (Training Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    # =============================================================================
    # SUBPLOT 2: TEST SET VISUALIZATION
    # =============================================================================
    plt.subplot(1, 2, 2)
    
    # Extract visualization features from test data
    X_vis_test = X_test[:, feature_indices]
    
    # Create meshgrid for test set (using test data range)
    X1_test, X2_test = np.meshgrid(
        np.arange(start=X_vis_test[:, 0].min() - 1, 
                  stop=X_vis_test[:, 0].max() + 1, 
                  step=0.01),
        np.arange(start=X_vis_test[:, 1].min() - 1, 
                  stop=X_vis_test[:, 1].max() + 1, 
                  step=0.01)
    )
    
    # Prepare test meshgrid points for prediction
    if X_test.shape[1] == 3:
        median_gender_test = np.median(X_test[:, 0])
        mesh_points_test = np.column_stack([
            np.full(X1_test.ravel().shape, median_gender_test),
            X1_test.ravel(),
            X2_test.ravel()
        ])
    else:
        mesh_points_test = np.array([X1_test.ravel(), X2_test.ravel()]).T
    
    # Generate decision boundary for test visualization
    decision_boundary_test = classifier.predict(mesh_points_test).reshape(X1_test.shape)
    
    # Plot test set decision boundary
    plt.contourf(X1_test, X2_test, decision_boundary_test, alpha=0.75, cmap=cmap)
    plt.xlim(X1_test.min(), X1_test.max())
    plt.ylim(X2_test.min(), X2_test.max())
    
    # Plot actual test data points
    for i, class_label in enumerate(np.unique(y_test)):
        class_mask = y_test == class_label
        plt.scatter(X_vis_test[class_mask, 0], X_vis_test[class_mask, 1],
                   color=cmap(i), label=f'Class {class_label}', 
                   edgecolors='black', s=50, alpha=0.8)
    
    plt.title('K-NN Decision Boundary (Test Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    # Improve layout and display
    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_knn_decision_boundary(X_train, y_train, X_test, y_test, classifier)

# =============================================================================
# DECISION BOUNDARY EXPLANATION
# =============================================================================
# The visualization demonstrates how the K-NN classifier creates decision boundaries:
#
# 1. Meshgrid Creation: A fine grid of points is created across the feature space
# 2. Prediction: Each grid point is classified by the trained K-NN model
# 3. Boundary Visualization: Points of the same predicted class are colored identically
# 4. Data Overlay: Actual training/test points are plotted with their true labels
#
# Key Insights:
# - Red regions: Areas where the classifier predicts class 0 (no purchase)
# - Green regions: Areas where the classifier predicts class 1 (purchase)  
# - Decision boundaries are typically smooth for K-NN with k > 1
# - Training and test visualizations should show similar boundary patterns
#
# NOTE: For 3D data (with Gender encoding), the gender feature is held constant
# at its median value during visualization. This provides a 2D slice through
# the 3D decision space, showing how the classifier behaves for an "average"
# gender encoding value.
#
# TECHNICAL DETAILS:
# - Meshgrid step size of 0.01 provides smooth boundary visualization
# - Feature scaling ensures proper distance calculations in K-NN
# - Edge colors on scatter points improve visibility against background

# =============================================================================
# END OF K-NN MODULE
# =============================================================================