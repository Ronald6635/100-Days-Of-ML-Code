# Multiple Linear Regression

## Step 1: Data Preprocessing

### Importing the libraries
import pandas as pd
import numpy as np

### Importing the dataset
dataset = pd.read_csv('../datasets/50_Startups.csv')
""" print("Original Dataset:\n", dataset)
    R&D Spend  Administration  Marketing Spend       State     Profit
0   165349.20       136897.80        471784.10    New York  192261.83
1   162597.70       151377.59        443898.53  California  191792.06
2   153441.51       101145.55        407934.54     Florida  191050.39
3   144372.41       118671.85        383199.62    New York  182901.99
4   142107.34        91391.77        366168.42     Florida  166187.94
5   131876.90        99814.71        362861.36    New York  156991.12 """
X = dataset.iloc[ : , :-1].values # Select columns including R&D Spend, Administration, Marketing Spend, and State
Y = dataset.iloc[ : ,  4 ].values # Select column including Profit


### Encoding Categorical data
# The State column is categorical and needs to be converted into numerical values for the regression model.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder()
state_encoded = onehotencoder.fit_transform(X[:, 3].reshape(-1, 1)).toarray()
X = np.concatenate([X[:, :3], state_encoded, X[:, 4:]], axis=1)

### Avoiding Dummy Variable Trap
X = X[: , 1:]
# The first column of the one-hot encoded matrix is dropped to avoid the dummy variable trap.
# The dummy variable trap occurs when one variable can be perfectly predicted from the others, leading to multicollinearity.

### Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

### Feature Scaling
# Feature scaling is not necessary for multiple linear regression as it is not sensitive to the scale of the features.

## Step 2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

## Step 3: Predicting the Test set results
Y_pred = regressor.predict(X_test)

print("Predicted values: ", Y_pred)
print("Actual values: ", Y_test)


# Use a bar chart to visualize the predicted vs actual values
import matplotlib.pyplot as plt 
import seaborn as sns # Seaborn is a data visualization library based on Matplotlib
# Set the style of seaborn
sns.set(style="whitegrid")

# Set width of bars
bar_width = 0.35

# Set positions of bars on X axis
r1 = np.arange(len(Y_test))
r2 = [x + bar_width for x in r1]

plt.figure(figsize=(10, 6))
# Create side-by-side bars
plt.bar(r1, Y_test, width=bar_width, color='red', label='Actual')
plt.bar(r2, Y_pred, width=bar_width, color='blue', label='Predicted')

# Add xticks in the middle of the group bars
plt.xlabel('Index')
plt.ylabel('Profit')
plt.title('Actual vs Predicted Values')
plt.xticks([r + bar_width/2 for r in range(len(Y_test))], np.arange(len(Y_test)))
plt.legend()
plt.show()