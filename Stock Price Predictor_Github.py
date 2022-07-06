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
df = pd.read_csv('ADANIGREEN.NS.csv')

x = df[['High', 'Open', 'Low', 'Volume']].values        # Independent Variables
y = df['Close'].values                                  # Dependent Variables

# Split The Data Set Into 80% Training & 20% Testing (For Linear Regression)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rg = LinearRegression()
rg.fit(x_train, y_train)
y_pred = rg.predict(x_test)
result = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

plt.figure(figsize=(16, 8))
plt.title('Adani Greens')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['Close'])
plt.show()
