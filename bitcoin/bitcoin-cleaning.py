import pandas as pd
import numpy as np

## cleaning steps from:
## https://www.kaggle.com/smitad/bitcoin-trading-strategy-simulation

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

data_loc = "/media/sree/mars/data/bitcoin/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv"
data = pd.read_csv(data_loc)



## Rows with NULL open prices

print('Total null open prices: %s' % data['Open'].isnull().sum())

## ffill - propogate the previous valid value in place of na
## until the next valid observation in the series
##
## bfill - propogate the next valid value in place of na
## until the next valid observation in the series


## Trading Signals

signal_lookback = 60 * 24 * 60 # days * hours * minutes
data['Buy'] = np.zeros(len(data))
data['Sell'] = np.zeros(len(data))


## Rolling window calculations in pandas
data['RollingMax'] = data['Close'].shift(1).rolling(signal_lookback, min_periods=signal_lookback).max()
data['RollingMin'] = data['Close'].shift(1).rolling(signal_lookback, min_periods=signal_lookback).min()

## buy if the amount at close is greater than previous 60 days - stocks are likely to increase
## sell if the amount at close is less than min of previous 60 days - stocks are likely to go down
data.loc[data['RollingMax'] < data['Close'], 'Buy'] = 1
data.loc[data['RollingMin'] > data['Close'], 'Sell'] = -1

