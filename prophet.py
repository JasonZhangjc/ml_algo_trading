'''
Use facebook prophet to predict the future prices
In order to ues facebook prophet, need to install RTools and RStan
'''

import pandas as pd
import pandas_ta as ta
from prophet import Prophet
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
from neuralprophet import NeuralProphet







def read_data():
    '''
    read data from .csv files
    '''
    df = pd.read_csv("EURUSD_Candlestick_1_Hour_BID_04.05.2003-15.04.2023.csv")
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)
    df.isna().sum()
    df=df[1000:2000]

    df = df.rename(columns={"Gmt time": "ds"})
    df['y'] = df['close']

    df['ds'] = df['ds'].str[:-4] #hourly timeframe
    df['ds']=pd.to_datetime(df['ds'], format = '%d.%m.%Y %H:%M:%S')

    return df


    
def prophet_signal(df, l, backcandles, frontpredictions, diff_limit, signal):
    '''
    Use facebook prophet to do the prediction
    '''
    dfsplit = copy.deepcopy(df[l-backcandles:l+1])
    model = Prophet()
    model.fit(dfsplit) 
    #the algo runs at the closing time of the current candle 
    # which is included in the fit
    future = model.make_future_dataframe(periods=frontpredictions, 
                                         freq='H', include_history=False)
    forecast = model.predict(future)
    if signal:
        if(forecast.yhat.mean()-dfsplit.close.iat[-1]<diff_limit):
            return 1
        elif(forecast.yhat.mean()-dfsplit.close.iat[-1]>diff_limit):
            return 2
        else:
            return 0
    else:
        forecast["y_target"] = df['y'].iloc[l+1:l+frontpredictions+1]
        forecast = forecast[['yhat', 'yhat_lower', 'yhat_upper']].values[0]
        return forecast[0],forecast[1],forecast[2]



def prophet_predict(df, backcandles=100, frontpredictions=1):
    '''
    Generate yhat, yhathigh, yhatlow for df
    '''
    yhatlow = [0 for i in range(len(df))]
    yhat = [0 for i in range(len(df))]
    yhathigh = [0 for i in range(len(df))]

    for row in tqdm(range(backcandles, len(df)-frontpredictions)):
        prophet_pred = prophet_signal(df, row, backcandles, 
            frontpredictions=1, diff_limit=0.005, signal=False)
        yhat[row] = prophet_pred[0]
        yhatlow[row] = prophet_pred[1]
        yhathigh[row] = prophet_pred[2]

    df["yhat"] = yhat
    df["yhatlow"] = yhatlow
    df["yhathigh"] = yhathigh

    df['yhatlow'] = df['yhatlow'].shift(+1)
    df['yhathigh'] = df['yhathigh'].shift(+1)
    df['yhat'] = df['yhat'].shift(+1)

    return df



def plot_x(df, x1 = 200, x2 = 250):
    '''
    Plot yhat and y from x1 to x2
    '''
    plt.figure(figsize=(18,6))
    plt.plot(df['ds'].iloc[x1:x2],df['yhatlow'].iloc[x1:x2], label="Prediction", marker='o')
    plt.fill_between(df['ds'].iloc[x1:x2], df['yhat'].iloc[x1:x2], df['yhathigh'].iloc[x1:x2], color='b', alpha=.1)
    plt.plot(df['ds'].iloc[x1:x2], df['y'].iloc[x1:x2], label = "Market", marker='o')
    plt.legend(loc="upper left")



def add_prophet_signal(df, backcandles=100, frontpredictions=4):
    '''
    Add prophet signals to df
    '''
    prophetsignal = [0 for i in range(len(df))]
    for row in tqdm(range(backcandles, len(df)-frontpredictions)):
        prophetsignal[row] = prophet_signal(df, row, backcandles, 
                                            frontpredictions=1, diff_limit=0.005)
    df["prophet_signal"] = prophetsignal
    df.reset_index(inplace=True, drop=True)

    return df



def generate_pointpos(df):
    '''
    generate pointpos for trading
    '''
    def pointpos(x):
        if x["prophet_signal"]==1:
            return x['high']+1e-4
        elif x["prophet_signal"]==2:
            return x['low']-1e-4
        else:
            return np.nan

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

    return df



def plot_df(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'])])

    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
        paper_bgcolor='black',
        plot_bgcolor='black')
    fig.update_xaxes(gridcolor='black')
    fig.update_yaxes(gridcolor='black')
    fig.add_scatter(x=df.index, y=df['pointpos'], mode="markers",
                    marker=dict(size=8, color="MediumPurple"),
                    name="Signal")
    fig.show()



def neural_prophet_signal(df, l, backcandles, frontpredictions, diff_limit):
    dfsplit = copy.deepcopy(df[l-backcandles:l+1][["ds", "y"]])
    model=NeuralProphet()
    model.fit(dfsplit) #the algo runs at the closing time of the current candle which is included in the fit
    future = model.make_future_dataframe(df=dfsplit, periods=frontpredictions)
    print(future)
    forecast = model.predict(future)
    #forecast = model.predict(df.iloc[l+1:l+frontpredictions,:-1]) #prediction start from next candle
    model.set_plotting_backend("plotly")
    #fig = model.plot(forecast)
    if(forecast.yhat1.mean()-df[l-backcandles:l+1].close.iat[-1]<diff_limit):
        return 1
    elif(forecast.yhat1.mean()-df[l-backcandles:l+1].close.iat[-1]>diff_limit):
        return 2
    else:
        return 0
    




if __name__ == '__main__':
    df = read_data()

    backcandles=100
    frontpredictions=1
    df = prophet_predict(df, backcandles, frontpredictions)

    x1 = 200
    x2 = 250
    plot_x(df, x1, x2)

    backcandles = 100
    frontpredictions = 4
    add_prophet_signal(df, backcandles, frontpredictions)
    df = generate_pointpos(df)
    plot_df(df)

    neural_prophet_signal(df, 200, 100, frontpredictions=5, diff_limit=0.002)



