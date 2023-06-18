# ml_algo_trading
 Machine Learning meets Algorithmic Trading
 
Techniques used in this project:
- KNN in sklearn
- XGBoost
- MLP in sklearn
- Feature selection
- LSTM in tensorflow
- Facebook Prophet
- PCA

### Machine learning for trend prediction ###

Three takeaways:

1. Do not predict specific price, predict trend instead

    Price prediction is often very inaccurate.
    Trend prediction can be inaccurate, too.
    But it can serve as an indicator for traders to consider.

2. Do not sample randomly for train_test_split

    Random sampling leads to overfitting and does not generalize to unseen data.
    Instead, we should simply use the early 80% time-series data, 
    as training set and later 20% time-series data as test set.

3. Lower your standard for results evluation

    The recall can be low, as we cannot seize every trading opportunity
    to take profit.
    However, we need a high precision to make sure it is a correct opportunity.
    How high? Above 50%

The key observation is that ML is not as suitable as imaginations for algo trading.