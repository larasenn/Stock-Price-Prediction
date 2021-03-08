import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import csv


df=pd.read_csv("GLD (3).csv")

training_set = df.iloc[:800, 1:2].values
test_set = df.iloc[800:, 1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []

for i in range(60, 800):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model = Sequential()

#Add the first LSTM layer and Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units = 1))
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Compile the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_train = df.iloc[:800, 1:2]
dataset_test = df.iloc[800:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(60, 519):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)


predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(df.loc[800:, "Date"],dataset_test.values, color = "red", label = "Real GLD Stock Price")
plt.plot(df.loc[800:, "Date"],predicted_stock_price, color = "blue", label = "Predicted GLD Stock Price")

plt.xticks(np.arange(0,459,50))
plt.title('GLD Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('GLD Stock Price')
plt.legend()
plt.show()

def moving_average(df, n):

    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    df = df.join(round(MA,3))



    return df


def wma_calc(w):
    def g(x):
        return sum(w*x)/sum(w)
    return g


def weighted_moving_average(df, n):

    weights =list(reversed([(n-p)*n for p in range(n)]))
    WMA = pd.Series(df['Close'].rolling(window=n).apply(wma_calc(weights), raw=True), name='WMA_' + str(n))
    df = df.join(round(WMA,3))

    return df


def exponential_moving_average(df, n):

    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    df = df.join(round(EMA,3))

    return df

def bollinger_bands(df, n):

    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df['Close'].rolling(n, min_periods=n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)

    return df

def relative_strength_index(df, n):

    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <=df.index[-1]:
        UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df = df.join(RSI)


    return df

def commodity_channel_index(df, n):

    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()/ PP.rolling(n, min_periods=n).std()),
                    name='CCI_' + str(n))
    df = df.join(CCI)

    return df


def stochastic_oscillator_d(df, n):

    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SO%d_' + str(n))
    df = df.join(SOd)



    return df

def macd(df, n_fast, n_slow):

    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    with open("data_2.csv", "w",newline='') as nfile:
        writer = csv.writer(nfile)
        for row in round(MACD,3):
            writer.writerow([row])
    with open("data_3.csv", "w",newline='') as nfile:
        writer = csv.writer(nfile)
        for row in round(MACDsign,3):
            writer.writerow([row])
    with open("GLD.csv", "w",newline='') as nfile:
        writer = csv.writer(nfile)
        for row in round(MACDdiff,3):
            writer.writerow([row])
    return df


print("SMA")
print(moving_average(df, 7))
print("WMA")
print(weighted_moving_average(df, 7))
print("EMA")
print(exponential_moving_average(df, 7))
print("BOLLIN BAND")
print(bollinger_bands(df, 20))
print("RSI")
print(relative_strength_index(df, 7))
print("CCI")
print(commodity_channel_index(df, 14))
print("STOCHASTIC")
print(stochastic_oscillator_d(df, 5))
print("MACD")
print(macd(df, 12, 26))