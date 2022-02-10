

import pandas as pd
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

days = 365


tick = yf.Ticker("TCS.NS")
stock_hist = tick.history("max")
df = stock_hist.Close
df = df.to_frame().reset_index()
df_for_training = df[["Close"]]


# Creating dataset
train_X = []
#train_Y = []
#n_past = 200
n_past = 14
n_future = 1

df_for_training = df_for_training.values


#creating data for prediction
for i in range(len(df_for_training)-n_past-1):
		a = df_for_training[i:(i+n_past), 0]   ###i=0, 0,1,2,3-----99   100 
		train_X.append(a)
		


train_X = np.array(train_X)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], n_future)
# train_X.shape
# train_X[-1:].shape
# train_X[-1].reshape(-1, 1).shape
temp = train_X[-1].reshape(-1, 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
temp = sc.fit_transform(temp)

li = []


# loading model from tensorflow import keras 
# model using 14steps
# creating for loop for n number of prediction
def pred_and_load(days, temp_for):
    from tensorflow import keras
    timestamp = n_past
    model = keras.models.load_model("de_lstm_14days.h5")
    li = []
    for i in range(days):
        x_input_for = temp_for.reshape(1, temp_for.shape[0],1)
        pred_for = model.predict(x_input_for).flatten()
        li.append(pred_for)
        next_for = np.append(temp_for,pred_for)
        temp_for = next_for[-timestamp:]
    return li






li = pred_and_load(days, temp)
arr = np.array(li)
y_pred_future = sc.inverse_transform(arr)
y_pred_future = y_pred_future.flatten()


#creating dates for forecasted days
training_dates = pd.to_datetime(df["Date"])
forecast_period_date = pd.date_range(list(training_dates)[-1], periods=days, freq="1d").tolist()
forecast_dates = []
for time_i in forecast_period_date:
   forecast_dates.append(time_i.date())


#creating dataset for future prediction
df_future = pd.DataFrame({"Date":np.array(forecast_dates), "Close":y_pred_future})

original = stock_hist["Close"]
original = original.to_frame().reset_index()


#ploting predicted price
plt.title(f"Predicted Stock Price for {days} Days")
plt.plot(forecast_dates,  y_pred_future)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

#creating filter for dates
from datetime import datetime, timedelta
if days < 365:
    back_days = 730
else:
    back_days = 1460

past_date = datetime.now() - timedelta(back_days)
filter_date = datetime.strftime(past_date.date(), "%Y-%m-%d")
shorted = original[original['Date'] > filter_date]


#ploting actual vs predicte
plt.title("Actual Price Vs Predicted Price")
plt.plot(shorted['Date'], shorted['Close'], label = 'Actual')
plt.plot(df_future["Date"], df_future["Close"], label="Predicted")
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()



df_future

