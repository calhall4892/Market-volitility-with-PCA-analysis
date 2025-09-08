import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# loading the data
df = pd.read_csv("data\Download Data - INDEX_UK_FTSE UK_UKX.csv", thousands = ',')
print(df.head())

# creating the daily change in values (volatility)
df['daily volatility'] = (df['Close'] - df['Open']) / df['Open']
df['target'] = (df['daily volatility'] > 0).astype(int)