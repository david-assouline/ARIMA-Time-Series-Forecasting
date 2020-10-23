import pandas as pd
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import datetime as datetime


# Load Data
df = pd.read_csv("AC.TO.M.csv")

# Optimize Dataframe
df.drop(["Open","High","Low","Close","Volume"],axis=1,inplace=True)
df.columns = ["Date","Adj Close"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date",inplace=True)
time_series = df["Adj Close"]


# Plot 30 Day Mean/Std Dev
# time_series.rolling(30).mean().plot(label="30 Day Rolling Mean")
# time_series.rolling(30).std().plot(label="30 Day Rolling Std Dev")
# time_series.plot()
# plt.legend()
# plt.show()

# Initialize model with P,D,Q = 1,0,0 and seasonality = 1,1,1,12
model = ARIMA(time_series, order=(1, 0, 0), seasonal_order=(1,1,1,12))
results = model.fit()

# backtest model with previous data
# df["Prediction"] = results.predict(start=datetime.datetime(2018, 1, 1), end=datetime.datetime(2020,10,22))
# df[["Adj Close","Prediction"]].plot(figsize=(10,6))
# plt.show()

# new dataframe with predictions
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1,12)]
future_df = pd.DataFrame(index=future_dates,columns=df.columns)
final = pd.concat([df,future_df])
final["Prediction"] = results.predict(start=datetime.datetime(2020,10,1), end=datetime.datetime(2021,9,1))
final[["Adj Close","Prediction"]].plot(figsize=(10,6))

# plot graph
plt.title("12 Month Predicted AC price")
plt.xlabel("Date")
plt.ylabel("Adj Close Price")
plt.show()






