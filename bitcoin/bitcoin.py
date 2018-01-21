import pandas as pd
import numpy as np

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

data_loc = "/media/sree/mars/data/bitcoin/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv"
data = pd.read_csv(data_loc)

import datetime
data["year_month"] = data["Timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))


data.head()

data.groupby(['year_month']).size()

years = np.unique(data['year_month'].dt.year)
years


avg_open = []
avg_volume = []
avg_close = []
avg_high = []
avg_low = []
avg_BTC = []
avg_average = []

for year in years:
    avg_volume.append(data[data['year_month'].dt.year == year]['Volume_(Currency)'].mean())
    avg_BTC.append(data[data['year_month'].dt.year == year]['Volume_(BTC)'].mean())
    avg_open.append(data[data['year_month'].dt.year == year]['Open'].mean())
    avg_close.append(data[data['year_month'].dt.year == year]['Close'].mean())
    avg_high.append(data[data['year_month'].dt.year == year]['High'].mean())
    avg_low.append(data[data['year_month'].dt.year == year]['Low'].mean())
    avg_average.append(data[data['year_month'].dt.year == year]['Weighted_Price'].mean())


monthly_avg_open = []
monthly_avg_volume = []
monthly_avg_close = []
monthly_avg_high = []
monthly_avg_low = []
monthly_avg_BTC = []
monthly_avg_average = []

months = list(range(1,13,1))

for year in years:
    for month in months:
        avg_volume.append(data[(data['year_month'].dt.year == year) & (data['year_month'].dt.month == month)]['Volume_(Currency)'].mean())        

        

        
