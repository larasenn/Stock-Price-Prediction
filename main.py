from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import csv

data = yf.download('GLD', '1985-2-4', '2020-8-17', auto_adjust=True)
data = data[['Close']]

data = data.dropna()
data.Close.plot(figsize=(10, 7),color='b')
plt.ylabel("Gold Prices")
plt.show()


data['Average_3'] = data['Close'].rolling(window=3).mean()
data['Average_9'] = data['Close'].rolling(window=20).mean()
data['next_day_price'] = data['Close'].shift(-1)

data =data.dropna()

X = data[['Average_3', 'Average_9']]


y = data['next_day_price']

t = .8
t = int(t*len(data))


# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test=[]
y_test = y[t:]
# Create a linear regression
linear = LinearRegression().fit(X_train, y_train)

predicted_price = linear.predict(X_test)
p=[]
p=predicted_price
for i in range(np.size(p)):
    print(p[i]," ", y_test[i] )

import csv
with open('results.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    for i in range(np.size(p)):
        print(p[i], " ", y_test[i])
        spamwriter.writerow([p[i], " ", y_test[i]])


predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 7))
print(predicted_price)
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Gold ETF Price")
plt.show()
resemblance = linear.score(X[t:], y[t:])*100
float("{0:.2f}".format(resemblance))
print(resemblance)




