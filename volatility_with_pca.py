import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

# loading the data
df = pd.read_csv("data\Download Data - INDEX_UK_FTSE UK_UKX.csv", thousands = ',')
print(df.head())

# creating the daily change in values (volatility)
df['daily volatility'] = (df['Close'] - df['Open']) / df['Open']

#PCA is set up the same as a regression model. Create a X containing all of your features and y with the dependant variable 
# Creating the X and y for PCA
X = df['High', 'Low', 'Open', 'Close']
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
explained_varience = pca.explained_varience_ratio_

plt.plot(np.cumsum(explained_varience)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained varience')
plt.title('Comparison of components against explained varience')
plt.grid(True)
plt.show()

