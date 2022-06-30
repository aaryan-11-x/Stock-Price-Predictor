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

x = df[['High', 'Open', 'Low', 'Volume']].values
y = df['Close'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rg = LinearRegression()
rg.fit(x_train, y_train)
y_pred = rg.predict(x_test)
result = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

# Get The Close Price
df = df[['Close']]

# Create A Variable To Predict 'X' Days Into The Future
f_days = 25
df['Prediction'] = df[['Close']].shift(-f_days)
print(df.tail(4))  # Used To Get The Last 'n' Rows
