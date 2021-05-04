from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import time
# setting figure size
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

# for normalizing data
from sklearn.preprocessing import MinMaxScaler


def getTrainData(symbol, interval):
    market_client = MarketClient(init_log=True)
    list_obj = market_client.get_candlestick(symbol, interval, 2000)
    Id = []
    High = []
    Low = []
    Open = []
    Close = []
    Count = []
    Amount = []
    Volume = []
    if list_obj and len(list_obj):
        for obj in list_obj:
            Id.append(obj.id)
            High.append(obj.high)
            Low.append(obj.low)
            Open.append(obj.open)
            Close.append(obj.close)
            Count.append(obj.count)
            Amount.append(obj.amount)
            Volume.append(obj.vol)
        # print(obj.id)
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame(
        {'Id': Id, 'High': High, 'Low': Low, 'Open': Open, 'Close': Close, 'Count': Count, 'Amount': Amount,
         'Volume': Volume})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(interval + symbol + ".csv", index=False, sep=',')


def trainModel(x_train, y_train, units, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 7)))
    model.add(LSTM(units=units))
    model.add(Dense(7))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    # print(model.summary())
    return model


def predictval(model, original_data, leng, scaler):
    all_closing_price = []
    original_data_cp = original_data
    for i in range(0, leng):
        # print(i)
        X_test = []
        inputs = scaler.transform(original_data_cp)
        for j in range(0, i + 1):
            X_test.append(inputs[j:(j + 60), ])
        X_test = np.array(X_test)
        closing_price = model.predict(X_test)
        tmp = scaler.inverse_transform(closing_price)
        all_closing_price = tmp
        original_data_cp = np.append(original_data, tmp)
        original_data_cp = original_data_cp.reshape(-1, 7)

    return all_closing_price


# 初始化参数
global contr
global lastPrice
global totalprofit

contr = False
lastPrice = 0
totalprofit = 0


def buy_or_sell(closeList, whenTime):
    market_client = MarketClient()
    nowshould = closeList[0] < closeList[len(closeList) - 1]
    global contr
    global lastPrice
    global totalprofit
    # 买入true
    if contr and nowshould:
        now_price = market_client.get_market_trade(symbol="btcusdt")
        now_price = now_price[0].price
        lastPrice = now_price
        print("buy Price:", str(lastPrice) + "," + str(whenTime))
        contr = False

    # 卖出 false
    if (bool(1 - contr)) and (bool(1 - nowshould)):
        if lastPrice == 0:
            contr = True
        else:
            now_price = market_client.get_market_trade(symbol="btcusdt")
            now_price = now_price[0].price
            if (now_price - lastPrice - now_price * 0.002 - lastPrice * 0.002) > 0:
                print("sell Price:", str(now_price) + "," + str(whenTime))
                totalprofit = totalprofit + now_price - lastPrice - now_price * 0.002 - lastPrice * 0.002
                print("totalprofit:", str(totalprofit) + "," + str(whenTime))
                contr = True


tf.random.set_seed(54294)
while True:
    scaler = MinMaxScaler(feature_range=(0, 1))
    interval = CandlestickInterval.MIN5
    symbol = "btcusdt"
    getTrainData(symbol, interval)
    # read the file
    df = pd.read_csv(interval + symbol + ".csv")
    # df.head()
    filename = str(df['Id'][0]) + interval + symbol
    whenTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(df['Id'][1]))
    maxTime = df['Id'][0]
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(df['Id'][0])))
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(df['Id'][1])))
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>----------OK--------------------")
    df.index = (df['Id'] - df['Id'][len(df) - 1]) / (60 * 5)
    # print("index",df.index)
    data = df.sort_index(ascending=True, axis=0)
    data.drop('Id', axis=1, inplace=True)
    dataset = data.values
    # print("dataset", dataset.shape)
    train = dataset[0:2000, :]
    # print("train", len(train))
    trainLen = len(train)
    valid = dataset[len(train):, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, ])
        y_train.append(scaled_data[i,])

    x_train, y_train = np.array(x_train), np.array(y_train)
    # print(train.shape[0])

    inputs = dataset[(train.shape[0] - 60):, :]
    original_data = inputs[0:60]

    leng = dataset.shape[0] - train.shape[0]
    leng = 100
    # print("leng", leng)
    inputs = scaler.transform(inputs)
    # print("inputs", inputs.shape)
    X_test = []
    # print("inputs :", inputs.shape)
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, ])
    X_test.append(inputs[0:60, ])
    X_test = np.array(X_test)

    # create and fit the LSTM network
    model = trainModel(x_train, y_train, 120, 9, 12)
    # print("X_test", len(X_test))
    closing_price = model.predict(X_test)
    closing_price = np.array(closing_price)
    closing_price = scaler.inverse_transform(closing_price)
    my_closing_price = predictval(model, original_data, leng, scaler)

    my_closing_price = np.array(my_closing_price)
    buy_or_sell(my_closing_price[:, 3], whenTime)
    # train = data[0:trainLen]['Close']
    # # print("trainLen",trainLen)
    # # print("data",data[1800:2000])
    # valid = data[trainLen:trainLen + len(closing_price)]
    # # print("valid",len(valid))
    # # print("closing_price", len(closing_price))
    # # valid['Predictions'] = closing_price[:, 3]
    # # valid['MyPredictions'] = my_closing_price[:, 3]
    # plt.plot(train, label='train_Close')
    #
    # # plt.plot(valid['Close'], label='Close')
    # # plt.plot(valid['Predictions'], label='Predictions')
    # # plt.plot(valid['MyPredictions'], label='MyPredictions')
    # new_data = pd.DataFrame(my_closing_price[:, 3])
    # new_data.index = new_data.index + 2000
    # plt.plot(new_data, label='MyPredictions')
    # plt.legend(loc='best')
    # plt.savefig(str(filename) + ".png")
    # # plt.show()
    # plt.clf()
    # time.sleep(60 * 5)
    nowTime = int(time.time())
    if (nowTime - maxTime) < 300:
        print("time sleep:", (300 - (nowTime - maxTime)))
        time.sleep(300 - (nowTime - maxTime))
