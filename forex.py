import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance






####################################################
# Data preprocessing functions
####################################################

def read_data():
    df = pd.read_csv("USDJPY_Candlestick_1_D_ASK_05.05.2003-19.10.2019.csv")
    #Check if any zero volumes are available
    indexZeros = df[ df['volume'] == 0 ].index
    df.drop(indexZeros , inplace=True)

    return df



def preprocessing(df):
    df['ATR'] = df.ta.atr(length=20)
    df['RSI'] = df.ta.rsi()
    df['Average'] = df.ta.midprice(length=1) #midprice
    df['MA40'] = df.ta.sma(length=40)
    df['MA80'] = df.ta.sma(length=80)
    df['MA160'] = df.ta.sma(length=160)

    return df



def get_slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)

    return slope



def back_roll(df, backrollingN=6):
    df['slopeMA40'] = df['MA40'].rolling(window=backrollingN).apply(get_slope, raw=True)
    df['slopeMA80'] = df['MA80'].rolling(window=backrollingN).apply(get_slope, raw=True)
    df['slopeMA160'] = df['MA160'].rolling(window=backrollingN).apply(get_slope, raw=True)
    df['AverageSlope'] = df['Average'].rolling(window=backrollingN).apply(get_slope, raw=True)
    df['RSISlope'] = df['RSI'].rolling(window=backrollingN).apply(get_slope, raw=True)

    return df



def create_mytarget(df, barsupfront, pipdiff = 500*1e-3, SLTPRatio = 2):
    # target is the class label for 'y' in ML
    # tuning pipdiff can best change the distribution

    def mytarget(df, barsupfront):
        length = len(df)
        high = list(df['high'])
        low = list(df['low'])
        close = list(df['close'])
        open = list(df['open'])
        trendcat = [None] * length

        for line in range (0,length-barsupfront-2):
            valueOpenLow = 0
            valueOpenHigh = 0
            for i in range(1,barsupfront+2):
                value1 = open[line+1]-low[line+i]
                value2 = open[line+1]-high[line+i]
                valueOpenLow = max(value1, valueOpenLow)
                valueOpenHigh = min(value2, valueOpenHigh)

                if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
                    trendcat[line] = 1 #-1 downtrend
                    break
                elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
                    trendcat[line] = 2 # uptrend
                    break
                else:
                    trendcat[line] = 0 # no clear trend

        return trendcat

    df['mytarget'] = mytarget(df, barsupfront)

    return df



def plot_data(df):
    fig = plt.figure(figsize = (15,20))
    ax = fig.gca()
    df_model= df[['volume', 'ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope', 'mytarget']]
    df_model.hist(ax = ax)
    plt.show()



def plot_distribution(df):
    df_up=df.RSI[ df['mytarget'] == 2 ]
    df_down=df.RSI[ df['mytarget'] == 1 ]
    df_unclear=df.RSI[ df['mytarget'] == 0 ]
    pyplot.hist(df_unclear, bins=100, alpha=0.5, label='unclear')
    pyplot.hist(df_down, bins=100, alpha=0.5, label='down')
    pyplot.hist(df_up, bins=100, alpha=0.5, label='up')

    pyplot.legend(loc='upper right')
    pyplot.show()



####################################################
# ML functions
####################################################

def xy_split(df):
    df=df.dropna()
    attributes=['ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope']
    X = df[attributes]
    y = df["mytarget"]

    return df, X, y



def tt_split(X, y):
    # thge train_test_split seems correct
    # but it is actually wrong
    # time-series data, you cannot do such split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # former as train
    # latter as test
    train_index = int(0.8 * len(X))
    X_train, X_test = X[:train_index], X[train_index:]
    y_train, y_test = y[:train_index], y[train_index:]
    
    return X_train, X_test, y_train, y_test



def training(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=200, weights='uniform', 
                                 algorithm='kd_tree', leaf_size=30, p=1, 
                                 metric='minkowski', metric_params=None, n_jobs=1)
    model.fit(X_train, y_train)
    
    return model



def predict(model, X_train, X_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return y_pred_train, y_pred_test



def evaluate(df, y_pred_train, y_pred_test, y_train, y_test):
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("Accuracy train: %.2f%%" % (accuracy_train * 100.0))
    print("Accuracy test: %.2f%%" % (accuracy_test * 100.0))

    #------------------------------------------------------------------
    #--- How should I compare my accuracy ?
    print(df['mytarget'].value_counts()*100/df['mytarget'].count())

    # Random Model, gambler?
    pred_test = np.random.choice([0, 1, 2], len(y_pred_test))
    accuracy_test = accuracy_score(y_test, pred_test)
    print("Accuracy Gambler: %.2f%%" % (accuracy_test * 100.0))



def predict_xgboost(X_train, y_train, X_test, y_test):
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
    return model



def plot_xgboost(model):
    # plot feature importance
    plot_importance(model)
    pyplot.show()



if __name__ == "__main__":
    print("The program works!")

    # data preprocessing
    df = read_data()
    df = preprocessing(df)
    backrollingN=6
    df = back_roll(df, backrollingN)
    #Target flexible way
    pipdiff = 500*1e-3 #for TP
    SLTPRatio = 2 #pipdiff/Ratio gives SL
    barsupfront = 16
    df = create_mytarget(df, barsupfront, pipdiff, SLTPRatio)
    # plot_data(df)
    # plot_distribution(df)

    ################################################################
    # As a result, the df now has a new column called mytarget
    # we will use this column as the label for supervised learning
    ################################################################

    # machine learning with KNN
    df, X, y = xy_split(df)
    X_train, X_test, y_train, y_test = tt_split(X, y)
    model_knn = training(X_train, y_train)
    y_pred_train, y_pred_test = predict(model_knn, X_train, X_test)
    evaluate(df, y_pred_train, y_pred_test, y_train, y_test)
    # the results show that the accuracy is around 40% for 
    # train and test data
    # poor performance but highly likely not overfitting
    # our ML results are better than random gamble!

    # use XGBoost to solve the problem
    model_xgboost = predict_xgboost(X_train, y_train, X_test, y_test)
    plot_xgboost(model_xgboost)




