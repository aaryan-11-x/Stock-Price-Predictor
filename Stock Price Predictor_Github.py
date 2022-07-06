# Importing The Modules
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

plt.style.use('bmh')
style.use('ggplot')

# Enter Your File Name Here
df = pd.read_csv('WIPRO.NS.csv')

x = df[['High', 'Open', 'Low', 'Volume']].values        # Independent Variables
y = df['Close'].values                                  # Dependent Variables

plt.figure(figsize=(16, 8))
plt.title('Adani Greens')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['Close'])
plt.show()

# Get The Close Price
df = df[['Close']]

# Create A Variable To Predict 'X' Days Into The Future
f_days = 25
df['Prediction'] = df[['Close']].shift(-f_days)
print(df.tail(4))  # Used To Get The Last 'n' Rows
