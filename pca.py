'''
RSI indicator + PCA for algo trading
23 different RSI periods
Use PCA to reduce the dimension from 23 to 4
Use least square to predict future prices with the 4 PC
'''




import pandas as pd
import pandas_ta as ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import linalg as la




def read_data():
    '''
    Read data from .csv file
    '''
    # hourly bitcoin data
    df = pd.read_csv('BTCUSDT3600.csv')
    df['date'] = df['date'].astype('datetime64[s]')
    df = df.set_index('date')

    # people usually use small RSI periods
    # we select period=2 to period=25, 23 different periods in total
    # 23 dimensions data - needs PCA for dimension reduction
    rsi_periods = list(range(2, 25))
    rsis = pd.DataFrame()
    for p in rsi_periods:
        rsis[p] = ta.rsi(df['close'], p)

    # # plot RSI hitogram
    # plt.style.use('dark_background')
    # rsis.hist(bins=100)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
    #                     wspace=None, hspace=0.5)
    # plt.show()
    # rsis.plot()
    # plt.show()
    # # plot correlation between the 23 periods
    # sns.heatmap(rsis.corr(), annot=True)
    # plt.xlabel("RSI Period")
    # plt.ylabel("RSI Period")

    return df, rsis



def pca(rsis):
    '''
    Use PCA to reduce the dimensionality of RSIs
    '''
    # normalization
    rsi_means = rsis.mean()
    rsis -= rsi_means
    rsis = rsis.dropna()

    # Find covariance and compute eigen vectors
    cov = np.cov(rsis, rowvar=False)
    evals , evecs = la.eigh(cov)
    # Sort eigenvectors by size of eigenvalue
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    # reduce the dimension to 4
    n_components = 4
    rsi_pca = pd.DataFrame()
    for j in range(n_components):
        rsi_pca['PC' + str(j)] = pd.Series(np.dot(rsis, evecs[j]) , 
                                           index=rsis.index)

    plt.style.use('dark_background')
    rsi_periods = list(range(2, 25))
    for j in range(n_components):
        pd.Series(evecs[j], index=rsi_periods).plot(label='PC' + str(j+1))
    plt.xlabel("RSI Period")
    plt.ylabel("Eigenvector Value")
    plt.legend()
    plt.show()

    return rsi_pca



def pca_linear_model(x: pd.DataFrame, y: pd.Series, 
                     n_components: int, thresh: float= 0.01):
    '''
    use least square algo to fit the data for a model
    '''
    # Center data at 0
    means = x.mean()
    x -= means
    x = x.dropna()

    # Find covariance and compute eigen vectors
    cov = np.cov(x, rowvar=False)
    evals , evecs = la.eigh(cov)
    # Sort eigenvectors by size of eigenvalue
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    # Create data set for model
    model_data = pd.DataFrame()
    for j in range(n_components):
         model_data['PC' + str(j)] = \
            pd.Series( np.dot(x, evecs[j]) , index=x.index)
    
    cols = list(model_data.columns)
    model_data['target'] = y
    model_coefs = la.lstsq(model_data[cols], y)[0]
    model_data['pred'] = np.dot( model_data[cols], model_coefs)

    l_thresh = model_data['pred'].quantile(0.99)
    s_thresh = model_data['pred'].quantile(0.01)

    return model_coefs, means, l_thresh, s_thresh, model_data



def pca_and_ml_(df, rsis):
    '''
    Use PCA to process data and use ML algo to fit the data
    '''
    target = np.log(df['close']).diff(6).shift(-6)

    # Drop nans
    rsis['tar'] = target
    rsis = rsis.dropna()
    target = rsis['tar']
    rsis = rsis.drop('tar',axis=1)
    coefs, means, l_thresh, s_thresh, model_data = \
        pca_linear_model(rsis, target, 3)

    plt.style.use('dark_background')
    model_data.plot.scatter('pred', 'target')
    plt.axhline(0.0, color='white')
    plt.show()




if __name__ == '__main__':
    df, rsis = read_data()
    rsi_pca = pca(rsis)
    pca_and_ml_(df, rsis)