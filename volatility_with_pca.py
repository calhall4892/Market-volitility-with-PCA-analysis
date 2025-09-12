import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# loading the data
df = pd.read_csv("data\Download Data - INDEX_UK_FTSE UK_UKX.csv", thousands = ',')
print(df.head())

# creating the daily change in values (volatility)
df['daily volatility'] = (df['Close'] - df['Open']) / df['Open']

#PCA is set up the same as a regression model. Create a X containing all of your features and y with the dependant variable 
# Creating the X and y for PCA
X = df[['High', 'Low', 'Open', 'Close']]
y = df['daily volatility']

# splitting the data
split_point = int(len(X)* 0.7)
X_train = X[:split_point]
X_test = X[split_point:]
y_train = y[:split_point]
y_test = y[split_point:]

print('X train')
print(X_train.head())
print('X test')
print(X_test.head())

print('\n---')

print('y train')
print(y_train.head())
print('y test')
print(y_test.head())

# In order to understand how many n_components are needed, we must check the explained varience. This is used just after SVD in the PCA process and is therefore covered in more detail in the journal.
pca = PCA(n_components=None)
pca.fit(X_train)

# This is an explained varience metric build into the PCA
explained_varience = pca.explained_variance_ratio_

plt.plot(np.cumsum(explained_varience))
# np.cumsum stands for cumulative sum. so a, a+b, a+b+c etc
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained varience')
plt.title('Comparison of components against explained varience')
plt.grid(True)
plt.show()

# Now it can be seen that a reduction of n_components to 2 will still allow for valid results so it can be reapplied
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_transform = pca.transform(X_train)
X_test_transform = pca.transform(X_test)

# Now we instantiate the model
model = LinearRegression()

# fit
model.fit(X_train_transform, y_train)

# predict
y_pred = model.predict(X_test_transform)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# displaying the results
print(f'The MSE is :{mse:.4f}')
print(f'The RMSE is :{rmse:.4f}')
