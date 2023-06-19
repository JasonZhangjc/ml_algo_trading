'''
Use random forest with trendlines (resistance and support)
to predict future long or short decisions
'''


import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.ensemble import RandomForestClassifier



def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err



def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])



def fit_upper_trendline(data: np.array):
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (data - line_points).argmax() 
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)
    return resist_coefs 



def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared) 
    # coefs[0] = slope,  coefs[1] = intercept 
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
   
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 



def trendline_breakout_dataset(
        ohlcv: pd.DataFrame, lookback: int, 
        hold_period:int=12, tp_mult: float=3.0, sl_mult: float=3.0, 
        atr_lookback: int=168
):
    assert(atr_lookback >= lookback)

    close = np.log(ohlcv['close'].to_numpy())
   
    # ATR for normalizing, setting stop loss take profit
    atr = ta.atr(np.log(ohlcv['high']), np.log(ohlcv['low']), np.log(ohlcv['close']), atr_lookback)
    atr_arr = atr.to_numpy()
   
    # Normalized volume
    vol_arr = (ohlcv['volume'] / ohlcv['volume'].rolling(atr_lookback).median()).to_numpy() 
    adx = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], lookback)
    adx_arr = adx['ADX_' + str(lookback)].to_numpy()

    trades = pd.DataFrame()
    trade_i = 0

    in_trade = False
    tp_price = None
    sl_price = None
    hp_i = None
    for i in range(atr_lookback, len(ohlcv)):
        # NOTE window does NOT include the current candle
        window = close[i - lookback: i]

        s_coefs, r_coefs = fit_trendlines_single(window)

        # Find current value of line
        r_val = r_coefs[1] + lookback * r_coefs[0]

        # Entry
        if not in_trade and close[i] > r_val:
            
            tp_price = close[i] + atr_arr[i] * tp_mult
            sl_price = close[i] - atr_arr[i] * sl_mult
            hp_i = i + hold_period
            in_trade = True

            trades.loc[trade_i, 'entry_i'] = i
            trades.loc[trade_i, 'entry_p'] = close[i]
            trades.loc[trade_i, 'atr'] = atr_arr[i]
            trades.loc[trade_i, 'sl'] = sl_price 
            trades.loc[trade_i, 'tp'] = tp_price 
            trades.loc[trade_i, 'hp_i'] = i + hold_period
            
            trades.loc[trade_i, 'slope'] = r_coefs[0]
            trades.loc[trade_i, 'intercept'] = r_coefs[1]


            # Trendline features
            # Resist slope
            trades.loc[trade_i, 'resist_s'] = r_coefs[0] / atr_arr[i] 
       
            # Resist erorr
            line_vals = (r_coefs[1] + np.arange(lookback) * r_coefs[0])
            err = np.sum(line_vals  - window ) / lookback
            err /= atr_arr[i]
            trades.loc[trade_i, 'tl_err'] = err

            # Max distance from resist
            diff = line_vals - window
            trades.loc[trade_i, 'max_dist'] = diff.max() / atr_arr[i]

            # Volume on breakout
            trades.loc[trade_i, 'vol'] = vol_arr[i]

            # ADX
            trades.loc[trade_i, 'adx'] = adx_arr[i]


        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                trades.loc[trade_i, 'exit_i'] = i
                trades.loc[trade_i, 'exit_p'] = close[i]
                
                in_trade = False
                trade_i += 1

    trades['return'] = trades['exit_p'] - trades['entry_p']
    
    # Features
    data_x = trades[['resist_s', 'tl_err', 'vol', 'max_dist', 'adx']]
    # Label
    data_y = pd.Series(0, index=trades.index)
    data_y.loc[trades['return'] > 0] = 1

    return trades, data_x, data_y



def walkforward_model(
        close: np.array, trades:pd.DataFrame, 
        data_x: pd.DataFrame, data_y: pd.Series, 
        train_size: int, step_size: int
):
    
    signal = np.zeros(len(close))
    prob_signal = np.zeros(len(close))

    next_train = train_size
    trade_i = 0

    in_trade = False
    tp_price = None
    sl_price = None
    hp_i = None
      

    model = None 
    for i in range( len(close)):
        if i == next_train:
            start_i = i - train_size

            train_indices = trades[(trades['entry_i'] > start_i) & (trades['exit_i'] < i)].index

            x_train = data_x.loc[train_indices]
            y_train = data_y.loc[train_indices]
            print('training', i, 'N cases', len(train_indices))
            model = RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=69420)
            model.fit(x_train.to_numpy(), y_train.to_numpy())

            next_train += step_size
        
        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                signal[i] = 0
                prob_signal[i] = 0
                in_trade = False
            else:
                signal[i] = signal[i - 1]
                prob_signal[i] = prob_signal[i - 1]


        if  trade_i < len(trades) and i == trades['entry_i'].iloc[trade_i]:
            
            if model is not None:
                prob = model.predict_proba(data_x.iloc[trade_i].to_numpy().reshape(1, -1))[0][1]
                prob_signal[i] = prob

                trades.loc[trade_i, 'model_prob'] = prob

                if prob > 0.5: # greater than 50%, take trade
                    signal[i] = 1
                
                in_trade = True
                trade = trades.iloc[trade_i]
                tp_price = trade['tp'] 
                sl_price = trade['sl'] 
                hp_i = trade['hp_i'] 

            trade_i += 1

    return signal, prob_signal 





if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    trades, data_x, data_y = trendline_breakout_dataset(data, 72)
    signal, prob = walkforward_model(
            np.log(data['close']).to_numpy(), 
            trades, data_x, data_y, 
            365 * 24 * 2, 365 * 24
    )
    
    data['sig'] = signal

    # dumb_sig takes every trade, no ML filter
    data['dumb_sig'] = prob
    data.loc[data['dumb_sig'] > 0, 'dumb_sig'] = 1
   
    data = data[data.index > '2020-01-01']
    data['r'] = np.log(data['close']).diff().shift(-1)


    # Compute trade stats for all trades vs. model's selected trades
    trades = trades.dropna() 
    all_r = trades['return']    
    mod_r = trades[trades['model_prob'] > 0.5]['return']

    no_filter_rets = data['r'] * data['dumb_sig']
    filter_rets = data['r'] * data['sig']

    def prof_factor(rets):
        return rets[rets>0].sum() / rets[rets<0].abs().sum()
    
    
    print("All Trades PF", prof_factor(no_filter_rets))
    print("All Trades Avg", all_r.mean())
    print("All Trades Win Rate", len(all_r[all_r > 0]) / len(all_r) )
    print("All Trades Time In Market", len(data[data['dumb_sig'] > 0]) / len(data) )
    
    print("Meta-Label Trades PF", prof_factor(filter_rets)) 
    print("Meta-Label Trades Avg", mod_r.mean())
    print("Meta-Label Trades Win Rate", len(mod_r[mod_r > 0]) / len(mod_r))
    print("Meta-Label Time In Market", len(data[data['sig'] > 0]) / len(data))
    
    plt.style.use('dark_background') 
    (data['r'] * data['sig']).cumsum().plot(label='Meta-Labeled')
    (data['r'] * data['dumb_sig']).cumsum().plot(label='All Trades')
    (data['r']).cumsum().plot(label='Buy Hold')
    plt.legend()
    plt.show()
    





