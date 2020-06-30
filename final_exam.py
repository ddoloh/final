import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as stats

kospi = 'kospi.csv' # kospi csv file name from 2010 to 2019
kospi200 = 'kospi200.csv' # kospi200 csv file from 2010 to 2019

# for standardrization process.
first_close = 1696.140015
first_close_kospi200 = 223.49

# data pre-processing, kospi
kospi_df = pd.read_csv(kospi)
data_ = kospi_df.drop('Adj Close', axis= 1)

# data pre-processing, kospi200
kospi200_df = pd.read_csv(kospi)
data_200 = kospi200_df.drop('Adj Close', axis= 1)

# data types casting
data_['Date'] = pd.to_datetime(data_['Date'], format='%Y-%m-%d', errors='raise')
data_.astype({'Date':'datetime64', 'Open':'float64', 'High':'float64',
             'Low':'float64', 'Close':'float64', 'Volume':'int32'})
data = data_.set_index("Date")

data_200['Date'] = pd.to_datetime(data_200['Date'], format='%Y-%m-%d', errors='raise')
data_200.astype({'Date':'datetime64', 'Open':'float64', 'High':'float64',
             'Low':'float64', 'Close':'float64', 'Volume':'int32'})
data200 = data_200.set_index("Date")

# check
print(data.head())
print(data.index.dtype)
print(data.dtypes)

# Standardization
data['Close'] = data['Close'] / first_close
data['High'] = data['High'] / first_close
data['Low'] = data['Low'] / first_close
data['Open'] = data['Open'] / first_close

data200['Close'] = data200['Close'] / first_close_kospi200
data200['High'] = data200['High'] / first_close_kospi200
data200['Low'] = data200['Low'] / first_close_kospi200
data200['Open'] = data200['Open'] / first_close_kospi200



# check the result of standardrization Close price
print(data)

# moving average lines
ma21 = data['Close'].rolling(window=21).mean()
data.insert(len(data.columns), "MA21", ma21)

# plotting
plt.plot(data.index, data['MA21'], label='Moving average via 21 days')
plt.plot(data.index, data['Close'], label='KOSPI')
plt.legend(loc="best")
plt.grid()
plt.show()

# correlation
print("corr between MA21 and KOSPI :", data['Close'].corr(data['MA21']))


# plotting KOSPI and KOSPI200
plt.plot(data.index, data['Close'], label='KOSPI')
plt.plot(data.index, data200['Close'], label='KOSPI 200')
plt.legend(loc="best")
plt.grid()
plt.show()

# corrleation betwwen kospi and kospi200
print("corr between MA21 and KOSPI: ", data['Close'].corr(data200['Close']))

# lienar logistic regression analysis
model = stats.regression.linear_model.OLS(data['Close'], data200['Close'])
result = model.fit()
print(result.summary())

# problem 2, a
S0 = 1  # initial value
r = 0.02  # constant short rate
sigma = 0.285  # constant volatility
T = 1.0  # in years
I = 100000  # number of random draws
M = 400 # divided gap
dt = T / M
S = np.zeros((M + 1, I))
S[0] = S0

for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
                             + sigma * np.sqrt(dt) * np.random.standard_normal(I))

print("Fair value: %f", S)

# problem 2, b
S0 = 0.25  # initial value
r = 0.02  # constant short rate
sigma = 0.285  # constant volatility
T = 1.0  # in years
I = 100000  # number of random draws
M = 400 # divided gap
dt = T / M
S = np.zeros((M + 1, I))
S[0] = S0

total_value = 0.0

list_of_range = [0, 0.25, 0.5, 0.75, 1.0]
for i in list_of_range:
    S[0] = S0+i
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
                                 + sigma * np.sqrt(dt) * np.random.standard_normal(I))
    total_value +=S

print("total Fair value: %f", total_value)