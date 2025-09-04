import backtrader as bt
import numpy as np
from scipy.stats import norm, multivariate_normal
import joblib
import pandas as pd
from utils import price_dict_to_df, fetch_binance_data
from numba import njit
import matplotlib.pyplot as plt
# plt.style.use('dark_background')


def cum_returns_price(prices):
    logp = np.log(prices)
    rets = logp[1:] - logp[:-1]
    cum_ret = np.cumsum(rets)
    return cum_ret

def compute_cum_returns(prices):
    logp = np.log(prices)
    cum_ret = logp - logp[0]
    std = np.std(cum_ret, ddof=1)
    return cum_ret / std

def empirical_cdf(x):
    ranks = np.argsort(np.argsort(x))
    return (ranks + 1) / (len(x) + 1.0)

def to_gaussian(u, eps=1e-6):
    u = np.clip(u, eps, 1 - eps)
    return norm.ppf(u)

def mvn_pdf(obs, cov):
    return multivariate_normal.pdf(obs, mean=np.zeros(len(obs)), cov=cov)

def compute_spread_zscore(prices1, prices2):
    spread = np.log(prices1) - np.log(prices2)
    return (spread[-1] - np.mean(spread)) / (np.std(spread, ddof=1) + 1e-6)

@njit
def compute_beta(prices1, prices2):
    n = len(prices1)
    
    # true log returns
    rets1 = np.empty(n-1)
    rets2 = np.empty(n-1)
    for i in range(1, n):
        rets1[i-1] = np.log(prices1[i] / prices1[i-1])
        rets2[i-1] = np.log(prices2[i] / prices2[i-1])
    
    mean1 = np.mean(rets1)
    mean2 = np.mean(rets2)
    
    cov = np.sum((rets1 - mean1) * (rets2 - mean2)) / (n - 2)
    var2 = np.sum((rets2 - mean2) ** 2) / (n - 2)
    
    return cov / var2
    
class CopulaPairStrategy(bt.Strategy):
    params = dict(window=300, 
                  prob_threshold=0.05, 
                  entry_z=2, 
                  exit_z=0.0,
                  reverse=False)

    def __init__(self):
        self.dataclose0 = self.datas[0].close
        self.dataclose1 = self.datas[1].close
        
        self.logprice0 = cum_returns_price(self.dataclose0)
        self.logprice1 = cum_returns_price(self.dataclose1)
        
        self.probs = []
        self.zscores = []
        self.cumrets0 = []
        self.cumrets1 = []
        self.spreads = []
        self.equity = []
        self.curr_entry = None
        self.last_zscore = None

    def next(self):
        def append_nan():
            self.probs.append(np.nan)
            self.zscores.append(np.nan)
            self.cumrets0.append(np.nan)
            self.cumrets1.append(np.nan)
            self.spreads.append(np.nan)

        self.equity.append(self.broker.getvalue())

        # Not enough data
        if len(self.dataclose0) < self.p.window:
            append_nan()
            return

        window1 = np.array(self.dataclose0.get(size=self.p.window))
        window2 = np.array(self.dataclose1.get(size=self.p.window))

        # Invalid prices
        if np.any(np.isnan(window1)) or np.any(np.isnan(window2)) or np.any(window1 <= 0) or np.any(window2 <= 0):
            append_nan()
            return

        beta = compute_beta(window1, window2)
        cum_ret1, cum_ret2 = compute_cum_returns(window1), compute_cum_returns(window2)
        u1, u2 = empirical_cdf(cum_ret1), empirical_cdf(cum_ret2)
        z1, z2 = to_gaussian(u1), to_gaussian(u2)

        if len(z1) < 2 or len(z2) < 2:
            append_nan() 
            return

        cov = np.cov(np.column_stack([z1, z2]).T)
        pdf_val = mvn_pdf([z1[-1], z2[-1]], cov)
        zscore = compute_spread_zscore(window1, window2)
        spread = self.logprice0[len(self.dataclose0)-1] - self.logprice1[len(self.dataclose1)-1]

        # Store metrics
        self.probs.append(pdf_val)
        self.zscores.append(zscore)
        self.cumrets0.append(cum_ret1[-1] if cum_ret1.size else 0)
        self.cumrets1.append(cum_ret2[-1] if cum_ret2.size else 0)
        self.spreads.append(spread)
        
        frac = 0.1
        if pdf_val < self.p.prob_threshold:
            value = self.broker.get_value()
            px0, px1 = self.datas[0].close[0], self.datas[1].close[0]
            size0 = int((value * frac * abs(beta)) / px0)
            size1 = int((value * frac) / px1)

            if self.last_zscore:
                crossed_zero = (self.last_zscore > self.p.exit_z and zscore <= self.p.exit_z) or \
                            (self.last_zscore < self.p.exit_z and zscore >= self.p.exit_z)

                if crossed_zero:
                    if self.getposition(self.datas[0]).size:
                        self.close(self.datas[0])
                    if self.getposition(self.datas[1]).size:
                        self.close(self.datas[1])

            if zscore > self.p.entry_z and self.curr_entry != 1:
                if not self.p.reverse:
                    self.buy(self.datas[0], size=size0)
                    self.sell(self.datas[1], size=size1)
                else: 
                    self.sell(self.datas[0], size=size0)
                    self.buy(self.datas[1], size=size1)
                self.curr_entry = 1

            elif zscore < -self.p.entry_z and self.curr_entry != 2:
                if not self.p.reverse:
                    self.sell(self.datas[0], size=size0)
                    self.buy(self.datas[1], size=size1)
                else:
                    self.buy(self.datas[0], size=size0)
                    self.sell(self.datas[1], size=size1)
                self.curr_entry = 2
            
            self.last_zscore = zscore

            
datas = joblib.load('/Users/leoinv/Documents/CODE/BACKTEST/data/1h.joblib')
data1 = datas['BTCUSDT']
data2 = datas['ETHUSDT']

# Set datetime index
data1.set_index('open_time', inplace=True)
data2.set_index('open_time', inplace=True)

def load_data(symbol, timeframe):
    df = fetch_binance_data(symbol, timeframe, 0.3)
    df['open_time'] = pd.to_datetime(df.index)
    df.set_index('open_time', inplace=True)
    return df

# data1 = load_data('DOGEUSDT', '5m')
# data2 = load_data('SHIBUSDT', '5m')

# Align indices
idx = data1.index.intersection(data2.index)
data1 = data1.loc[idx]
data2 = data2.loc[idx]

# Initialize Cerebro
cerebro = bt.Cerebro()

# Add data feeds
feed1 = bt.feeds.PandasData(dataname=data1)
feed2 = bt.feeds.PandasData(dataname=data2)

cerebro.adddata(feed1)
cerebro.adddata(feed2)

params = dict(window=300, 
                prob_threshold=0.05, 
                entry_z=0.1, 
                exit_z=0.0,
                reverse=False)

cerebro.addstrategy(CopulaPairStrategy, **params)
cerebro.broker.set_cash(1000000000)
cerebro.broker.setcommission(0.0005)
strats = cerebro.run()
strat = strats[0]
cerebro.plot(width=20, height=20)

fig, ax = plt.subplots(5, 1, figsize=(12,12), sharex=True)

# 1. Cumulative Returns
ax[0].plot(strat.cumrets0, label='CumRet Asset1')
ax[0].plot(strat.cumrets1, label='CumRet Asset2')
ax[0].set_ylabel('Cumulative Returns')
ax[0].legend()

# 2. Copula PDF
ax[1].plot(strat.probs, label='Copula PDF')
ax[1].set_ylabel('PDF')
ax[1].legend()

# 3. Spread Z-score
ax[2].plot(strat.zscores, label='Spread Z-score', color='orange')
ax[2].set_ylabel('Z-score')
ax[2].legend()

# 4. Log Prices + Spread (twin axis)
lns1 = ax[3].plot(strat.logprice0, label='Log Price1', color='blue')
lns2 = ax[3].plot(strat.logprice1, label='Log Price2', color='red')
ax[3].set_ylabel('Log Prices')

ax3_twin = ax[3].twinx()
lns3 = ax3_twin.plot(strat.spreads, label='Spread Price', color='green')
ax3_twin.set_ylabel('Spread')

# merge legends
lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]
ax[3].legend(lns, labs, loc='best')

# 5. Portfolio Equity
ax[4].plot(strat.equity, label='Equity', color='red')
ax[4].set_ylabel('Portfolio Value')
ax[4].set_xlabel('Bar Index')
ax[4].legend()

plt.tight_layout()
plt.show()
