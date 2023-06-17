import pandas as pd
import pandas_ta as pa
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel






#############################################
# Data preprocessing
#############################################

def read_data():
    '''
    read data from .csv file
    '''
    df = pd.read_csv("EURUSD_Candlestick_1_D_ASK_05.05.2003-30.06.2021.csv")
    #Check if NA values are in data
    df=df[df['volume']!=0]
    df.reset_index(drop=True, inplace=True)
    
    return df



def generate_indicators(df):
    '''
    Generate many different indicators
    '''
    df["RSI"] = pa.rsi(df.close, length=16)
    df["CCI"] = pa.cci(df.high, df.low, df.close, length=16)
    df["AO"] = pa.ao(df.high, df.low)
    df["MOM"] = pa.mom(df.close, length=16)
    a = pa.macd(df.close)
    df = df.join(a)
    df["ATR"] = pa.atr(df.high, df.low, df.close, length=16)
    df["BOP"] = pa.bop(df.open, df.high, df.low, df.close, length=16)
    df["RVI"] = pa.rvi(df.close)
    a = pa.dm(df.high, df.low, length=16)
    df = df.join(a)
    a = pa.stoch(df.high, df.low, df.close)
    df = df.join(a)
    a = pa.stochrsi(df.close, length=16)
    df = df.join(a)
    df["WPR"] = pa.willr(df.high, df.low, df.close, length=16)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df



def get_target(df, barsupfront=20, pipdiff=200*1e-4, SLTPRatio=2):
    '''
    Generate target as labels (y) for supervised learning 
    '''
    #Target flexible way
    def mytarget(barsupfront, df1):
        length = len(df1)
        high = list(df1['high'])
        low = list(df1['low'])
        close = list(df1['close'])
        open = list(df1['open'])
        trendcat = [None] * length
        for line in range (0,length-barsupfront-2):
            valueOpenLow = 0
            valueOpenHigh = 0
            for i in range(1,barsupfront+2):
                value1 = open[line+1]-low[line+i]
                value2 = open[line+1]-high[line+i]
                valueOpenLow = max(value1, valueOpenLow)
                valueOpenHigh = min(value2, valueOpenHigh)
            #if ( (valueOpenLow >= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= (pipdiff/SLTPRatio)) ):
            #    trendcat[line] = 2 # bth limits exceeded
            #elif ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
            #    trendcat[line] = 3 #-1 downtrend
            #elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
            #    trendcat[line] = 1 # uptrend
            #elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
            #    trendcat[line] = 0 # no trend
            #elif ( (valueOpenLow >= (pipdiff/SLTPRatio)) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
            #    trendcat[line] = 5 # light trend down
            #elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= (pipdiff/SLTPRatio)) ):
            #    trendcat[line] = 4 # light trend up
                if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
                    trendcat[line] = 1 #-1 downtrend
                    break
                elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
                    trendcat[line] = 2 # uptrend
                    break
                else:
                    trendcat[line] = 0 # no clear trend

        return trendcat

    df['Target'] = mytarget(barsupfront, df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df



##########################################
# Machine learning 
##########################################

def xgboost_with_indicators(df):
    attributes = ['RSI', 'CCI', 'AO', 'MOM', 'MACD_12_26_9', 'MACDh_12_26_9', 
                  'MACDs_12_26_9', 'ATR', 'BOP', 'RVI', 'DMP_16', 'DMN_16', 
                  'STOCHk_14_3_3', 'STOCHd_14_3_3',
                  'STOCHRSIk_16_14_3_3', 'STOCHRSId_16_14_3_3', 'WPR']

    attributes = ['MACDs_12_26_9', 'ATR', 'DMP_16']

    X = df[attributes]
    y = df['Target']

    train_pct_index = int(0.7 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]

    model = XGBClassifier()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
    print('****Train Results****')
    print("Accuracy: {:.4%}".format(acc_train))
    print('****Test Results****')
    print("Accuracy: {:.4%}".format(acc_test))

    matrix_train = confusion_matrix(y_train, pred_train)
    matrix_test = confusion_matrix(y_test, pred_test)

    print(matrix_train)
    print(matrix_test)

    report_train = classification_report(y_train, pred_train)
    report_test = classification_report(y_test, pred_test)

    print(report_train)
    print(report_test)
    #choices = [2, 0, -1, +1]
    ##choices = [2, 0, 3, +1]
    print(model.get_booster().feature_names)

    plot_importance(model)
    pyplot.show()
    print(model.get_booster().feature_names)














if __name__ == '__main__':
    # data reading and preprocessing
    df = read_data()

    # generate indicators
    df = generate_indicators(df)

    # generate target as labels (y) for machine learning
    barsupfront=20
    pipdiff=200*1e-4
    SLTPRatio=2
    df = get_target(df, barsupfront, pipdiff, SLTPRatio)

    # xgboost model prediction
    xgboost_with_indicators(df)