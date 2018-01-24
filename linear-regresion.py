import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


data_loc = "/media/sree/mars/data/bitcoin/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv"
data = pd.read_csv(data_loc)

cols = data.columns.values

## feature scaling
## weighted price is the response variable
scaler = MinMaxScaler()
data[cols] = scaler.fit_transform(data[cols])
data.head()

X = data.iloc[:,[1,2,3,4,5,6]]
Y = data.iloc[:,[7]]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)


print ('Residual sum of squares Train: %.2f' % np.mean((model.predict(X_train) - y_train) ** 2))
print ('Residual sum of squares Test: %.2f' % np.mean((model.predict(X_test) - y_test) ** 2))




