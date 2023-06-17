import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Activation






#############################################
# Data preprocessing
#############################################

def read_data():
    '''
    download data from yahoo finance
    '''
    df = yf.download(tickers = '^RUI', start = '2012-03-11',end = '2022-07-10')
    return df



def add_indicators(df):
    '''
    Adding indicators to dataframe
    '''
    df['RSI']=ta.rsi(df.Close, length=15)
    df['EMAF']=ta.ema(df.Close, length=20)
    df['EMAM']=ta.ema(df.Close, length=100)
    df['EMAS']=ta.ema(df.Close, length=150)

    df['Target'] = df['Adj Close']-df.Open
    df['Target'] = df['Target'].shift(-1)

    df['TargetClass'] = [1 if df.Target[i]>0 else 0 for i in range(len(df))]
    df['TargetNextClose'] = df['Adj Close'].shift(-1)

    df.dropna(inplace=True)
    df.reset_index(inplace = True)
    df.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

    return df



def generate_dataset(df):
    '''
    generate dataset for LSTM
    '''
    dataset = df.iloc[:, 0:11]#.values
    sc = MinMaxScaler(feature_range=(0,1))
    dataset_scaled = sc.fit_transform(dataset)
    print('The shape of the scaled dataset is:', dataset_scaled.shape)

    return dataset_scaled



def xy_split(backcandles, dataset_scaled):
    '''
    Generate X and y for supervised learning
    '''
    # multiple feature from data provided to the model
    X = []
    #print(data_set_scaled[0].size)
    #data_set_scaled=data_set.values
    print(dataset_scaled.shape[0])
    for j in range(8):#data_set_scaled[0].size):#2 columns are target not X
        X.append([])
        for i in range(backcandles, dataset_scaled.shape[0]):#backcandles+2
            X[j].append(dataset_scaled[i-backcandles:i, j])
    print(len(X))        # 8
    print(len(X[0]))     # 2418
    print(len(X[0][0]))  # 30

    #move axis from 0 to position 2
    X = np.moveaxis(X, [0], [2])   # 2418 30 8

    #Erase first elements of y because of backcandles to match X length
    #del(yi[0:backcandles])
    #X, yi = np.array(X), np.array(yi)
    # Choose -1 for last column, classification else -2...
    X, yi = np.array(X), np.array(dataset_scaled[backcandles:,-1])
    y = np.reshape(yi,(len(yi),1))
    #y=sc.fit_transform(yi)
    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # print(X)
    print(X.shape)
    # print(y)
    print(y.shape)

    return X, y



def tt_split(X, y):
    '''
    train test split
    '''
    splitlimit = int(len(X)*0.8)
    print(splitlimit)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(y_train)

    return X_train, X_test, y_train, y_test



def lstm_model(X_train, X_test, y_train, y_test, backcandles):
    '''
    use LSTM model supported by tensorflow
    '''
    np.random.seed(10)
    # construct the model, i.e., the network layers
    lstm_input = Input(shape=(backcandles, 8), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    # optimizer and loss
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')
    # training
    model.fit(x=X_train, y=y_train, batch_size=15, 
              epochs=30, shuffle=True, validation_split = 0.1)
    # prediction
    y_pred = model.predict(X_test)
    #y_pred=np.where(y_pred > 0.43, 1,0)
    for i in range(10):
        print(y_pred[i], y_test[i])

    plt.figure(figsize=(16,8))
    plt.plot(y_test, color = 'black', label = 'Test')
    plt.plot(y_pred, color = 'green', label = 'pred')
    plt.legend()
    plt.show()





if __name__ == '__main__':
    # data reading and preprocessing
    df = read_data()
    df = add_indicators(df)
    dataset_scaled = generate_dataset(df)

    # generate data for supervised learning
    backcandles = 30
    X, y = xy_split(backcandles, dataset_scaled)
    X_train, X_test, y_train, y_test = tt_split(X, y)
    lstm_model(X_train, X_test, y_train, y_test, backcandles)
