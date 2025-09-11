import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# loading the data
df = pd.read_csv("data\Download Data - INDEX_UK_FTSE UK_UKX.csv", thousands = ',')
print(df.head())

# creating the daily change in values (volatility)
df['daily volatility'] = (df['Close'] - df['Open']) / df['Open']

#PCA is set up the same as a regression model. Create a X containing all of your features and y with the dependant variable 
# Creating the X and y for PCA
X = df['High', 'Low', 'Open', 'Close']
y = df['daily volatility']

split_data = int(len(X)* 0.7)
