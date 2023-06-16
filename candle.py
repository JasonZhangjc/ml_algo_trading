import pandas as pd
from backtesting import Strategy, Backtest
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



def support(df, l, n1, n2): #n1 n2 before and after candle l
    '''
    candle l is the <local max of low> from n1 to n2
    '''
    for i in range(l-n1+1, l+1):
        if(df.low[i]>df.low[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df.low[i]<df.low[i-1]):
            return 0
    return 1



def resistance(df, l, n1, n2): #n1 n2 before and after candle l
    '''
    candle l is the <local min of high> from n1 to n2
    '''
    for i in range(l-n1+1, l+1):
        if(df.high[i]<df.high[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df.high[i]>df.high[i-1]):
            return 0
    return 1



def generate_signal(df, n1=2, n2=2, backCandles=30):
    '''
    process the candlestick and generate signals for trading operations
    '''
    length = len(df)
    high = list(df['high'])
    low = list(df['low'])
    close = list(df['close'])
    open = list(df['open'])

    # bodydiff is the difference between close and open prices
    bodydiff = [0] * length
    highdiff = [0] * length
    lowdiff = [0] * length
    ratio1 = [0] * length
    ratio2 = [0] * length

    def isEngulfing(l):
        '''
        evaluate whether l is an Engulfing candle
        two types:
            uptrend = 2
            downtrend = 1
        '''
        row=l
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<0.000001:
            bodydiff[row]=0.000001

        bodydiffmin = 0.002                       # the threshold of trivial diff

        if (bodydiff[row]>bodydiffmin and         # current diff is not trivial
            bodydiff[row-1]>bodydiffmin and       # previous diff is not trivial
            open[row-1]<close[row-1] and          # previous uptrend
            open[row]>close[row] and              # current downtrend 
            (open[row]-close[row-1])>=-0e-5 and   # current open >= previous close
            close[row]<open[row-1]): #+0e-5 -5e-5 # current close < previous open
            return 1                              # turns out a downtrend Engulfing
        
        elif(bodydiff[row]>bodydiffmin and        # current diff is not trivial
            bodydiff[row-1]>bodydiffmin and       # previous diff is not trivial
            open[row-1]>close[row-1] and          # previous downtrend
            open[row]<close[row] and              # current uptrend
            (open[row]-close[row-1])<=+0e-5 and   # current open <= previousclose
            close[row]>open[row-1]):#-0e-5 +5e-5  # current close > previous open
            return 2                              # turns out a uptrend Engulfing
        else:
            return 0                              # undetermined

    def isStar(l):
        '''
        Evaluate whether candle l is a shooting star candle
        '''
        bodydiffmin = 0.0020     # threshold to avoid trivial diff
        row=l
        highdiff[row] = high[row]-max(open[row],close[row])
        lowdiff[row] = min(open[row],close[row])-low[row]
        bodydiff[row] = abs(open[row]-close[row])
        if bodydiff[row]<0.000001:
            bodydiff[row]=0.000001
        ratio1[row] = highdiff[row]/bodydiff[row]
        ratio2[row] = lowdiff[row]/bodydiff[row]

        if (ratio1[row]>1 and                    # highdiff is wide
            lowdiff[row]<0.2*highdiff[row] and   # lowdiff is very narrow
            bodydiff[row]>bodydiffmin):          # bodydiff is not trivial
            # and open[row]>close[row]):
            return 1                             # turns out an uptrend star
        
        elif(ratio2[row]>1 and                   # lowdiff is wide
            highdiff[row]<0.2*lowdiff[row] and   # highdiff is very narrow
            bodydiff[row]>bodydiffmin):          # bodydiff is not trivial
            # and open[row]<close[row]):
            return 2                             # turns out a downtrend star
        else:
            return 0                             # undetermined

    def closeResistance(l,levels,lim):
        '''
        evaluate whether candle l is close to resistance line
        levels: what are they?
        '''
        if len(levels)==0:
            return 0
        # c1: diff between <high, min(levels)> is small
        c1 = abs(df.high[l] - 
                 min(levels, key=lambda x:abs(x-df.high[l])))<=lim
        # c2: diff between <max(open,close), min(levels)> is small
        c2 = abs(max(df.open[l],df.close[l]) - 
                 min(levels, key=lambda x:abs(x-df.high[l])))<=lim
        # c3: min(open,close) < min(levels)
        c3 = min(df.open[l],df.close[l]) < \
             min(levels, key=lambda x:abs(x-df.high[l]))
        # c4: low < min(levels)
        c4 = df.low[l] < min(levels, key=lambda x:abs(x-df.high[l]))
        if( (c1 or c2) and c3 and c4 ):
            return 1     # turns out to be close to resistance
        else:
            return 0     # not close to resistance

    def closeSupport(l,levels,lim):
        '''
        evaluate whether candle l is close to support line
        '''
        if len(levels)==0:
            return 0
        # c1: diff between <low, min(levels)> is small
        c1 = abs(df.low[l] - 
                 min(levels, key=lambda x:abs(x-df.low[l])))<=lim
        # c2: diff between <min(open,close), min(levels)> is small
        c2 = abs(min(df.open[l],df.close[l]) - 
                 min(levels, key=lambda x:abs(x-df.low[l])))<=lim
        # c3: max(open,close) > min(levels) 
        c3 = max(df.open[l],df.close[l]) > \
             min(levels, key=lambda x:abs(x-df.low[l]))
        # c4: high > min(levels)
        c4 = df.high[l] > min(levels, key=lambda x:abs(x-df.low[l]))
        if( (c1 or c2) and c3 and c4 ):
            return 1      # turns out close to support
        else:
            return 0      # not close to support
    
    # start to generate signals
    signal = [0] * length

    for row in range(backCandles, len(df)-n2):
        ss = []
        rr = []
        for subrow in range(row-backCandles+n1, row+1):
            if support(df, subrow, n1, n2):
                ss.append(df.low[subrow])
            if resistance(df, subrow, n1, n2):
                rr.append(df.high[subrow])
        # parameters
        # if downtrend according to Engulfing or Star and
        # if candle is close to resistance
        # we expect the price to bounce down
        # sell signal!
        if ((isEngulfing(row)==1 or isStar(row)==1) and 
            closeResistance(row, rr, 150e-5) ):#and df.RSI[row]<30
            signal[row] = 1
        # if uptrend according to Engulfing or Star and
        # if candle is close to support
        # we expect the price to bounce up
        # buy signal!
        elif((isEngulfing(row)==2 or isStar(row)==2) and 
             closeSupport(row, ss, 150e-5)):#and df.RSI[row]>70
            signal[row] = 2
        else:
            signal[row] = 0

    df['signal']=signal
    df.columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume', 'signal']

    return df



###########################################
# Backtesting
###########################################

def backtest(df):
    def SIGNAL():
        return df.signal
    
    class MyCandlesStrat(Strategy):
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            if self.signal1==2:
                sl1 = self.data.Close[-1] - 600e-4
                tp1 = self.data.Close[-1] + 450e-4
                self.buy(sl=sl1, tp=tp1)
            elif self.signal1==1:
                sl1 = self.data.Close[-1] + 600e-4
                tp1 = self.data.Close[-1] - 450e-4
                self.sell(sl=sl1, tp=tp1)
    
    bt = Backtest(df, MyCandlesStrat, cash=10_000, commission=.00)
    stat = bt.run()
    print('The results of the backtesting are:\n', stat)
    bt.plot()

    return stat



##########################################
# Machine learning 
##########################################

def get_target(df, barsupfront=30, pipdiff=250*1e-4, SLTPRatio=1):
    '''
    Target flexible way
    pipdiff is for TP
    SLTPRatio is pipdiff/Ratio gives SL
    '''
    def mytarget(barsupfront, df1):
        length = len(df1)
        high = list(df1['High'])
        low = list(df1['Low'])
        close = list(df1['Close'])
        open = list(df1['Open'])
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
                if ( (valueOpenLow >= pipdiff) and 
                    (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
                    trendcat[line] = 1 # downtrend
                    break
                elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and 
                      (-valueOpenHigh >= pipdiff) ):
                    trendcat[line] = 2 # uptrend
                    break
                else:
                    trendcat[line] = 0 # no clear trend

        return trendcat

    df['Target'] = mytarget(barsupfront, df)
    df['Target'].hist()

    return df



def target_processing(df):
    '''
    Further processing the data such that 
    it is suitable for ML fitting
    '''
    df["RSI"] = pa.rsi(df.Close, length=16)
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    print('The new df is:\n', df.describe())

    attributes = ['RSI', 'signal', 'Target']
    df = df[attributes].copy()
    df['signal'] = pd.Categorical(df['signal'])
    dfDummies = pd.get_dummies(df['signal'], prefix = 'signalcategory')
    df = df.drop(['signal'], axis=1)
    df = pd.concat([df, dfDummies], axis=1)
    
    return df



def tt_split(df):
    '''
    generate/split X_train, y_train, X_test, y_test from df
    '''
    attributes = ['RSI', 'signalcategory_0', 'signalcategory_1', 'signalcategory_2']
    X = df[attributes]
    y = df['Target']
    train_pct_index = int(0.7 * len(X))
    X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]

    return X_train, X_test, y_train, y_test



def xgboost_model(X_train, X_test, y_train, y_test):
    '''
    Train the XGBoost model with X_train and y_train
    Evaluate accuracy with X_test and y_test
    '''
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

    return model, pred_train, pred_test



def get_metrics(model, X_train, X_test, y_train, y_test, pred_train, pred_test):
    '''
    Get metrics like precision, recall, F1-score
    '''
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



def plot_feature_importance(model):
    '''
    plot the importance of features
    '''
    plot_importance(model)
    pyplot.show()
    print(model.get_booster().feature_names)








if __name__ == '__main__':
    # data reading and preprocessing
    df = read_data()
    n1=2
    n2=2
    backCandles=30
    df = generate_signal(df, n1, n2, backCandles)

    # backtesting
    stat = backtest(df)

    # generate target as labels (y) for machine learning
    barsupfront=30
    pipdiff=250*1e-4
    SLTPRatio=1
    df = get_target(df, barsupfront, pipdiff, SLTPRatio)
    df = target_processing(df)
    # data preparation for ML models
    X_train, X_test, y_train, y_test = tt_split(df)
    # use XGBoost model
    model, pred_train, pred_test = xgboost_model(X_train, X_test, y_train, y_test)
    # get metrics
    get_metrics(model, X_train, X_test, y_train, y_test, pred_train, pred_test)
    # plot feature importance for the XGBoost model
    plot_feature_importance(model)






