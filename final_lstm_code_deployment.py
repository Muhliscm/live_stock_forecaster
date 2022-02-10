

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
import pandas as pd
from tensorflow import keras




tick = yf.Ticker("MSFT")



hist = tick.history("max")


df = hist.Close


# ### Preprocessing


df = df.to_frame()
df = df.reset_index()



#df.head()




training_dates = pd.to_datetime(df["Date"])

df_for_training = df[["Close"]]



#df_for_training.head()



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(df_for_training)



# Creating dataset
train_X = []
train_Y = []
n_past = 14
n_future = 1


df_for_training = df_for_training.values


for i in range(len(df_for_training)-n_past-1):
		a = df_for_training[i:(i+n_past), 0]    
		train_X.append(a)
		train_Y.append(df_for_training[i + n_past, 0])




# converting to array
train_X = np.array(train_X)
train_Y = np.array(train_Y)


#print(train_X.shape, train_Y.shape)

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], n_future)
#train_X.shape



train_Y = train_Y.reshape(train_Y.shape[0], n_future)
#train_Y.shape

# loading model from tensorflow import keras

model1 = keras.models.load_model("stock_prediction_lstm.h5")

n_future = 90
forecast_period_date = pd.date_range(list(training_dates)[-1], periods=n_future, freq="1d").tolist()


#forecast_period_date


forecast = model1.predict(train_X[-n_future:])
forecast

forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis = 1)
y_pred_future = forecast_copies


forecast_dates = []
for time_i in forecast_period_date:
   forecast_dates.append(time_i.date())


y_pred_future = y_pred_future.flatten()


df_future = pd.DataFrame({"Date":np.array(forecast_dates), "Close":y_pred_future})




df_future["Date"] = pd.to_datetime(df_future["Date"])



original = hist["Close"]
original = original.to_frame().reset_index()



plt.title(f"Predicted Stock Price for {n_future} Days")
plt.plot(forecast_dates,  y_pred_future)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()



plt.title("Actual Price Vs Predicted Price")
plt.plot(original["Date"], original["Close"], label="Actual Price")
plt.plot(df_future["Date"], df_future["Close"], label="Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


#df_future

#dumping model
# import pickle
# filename = "trained_model.sav"
# pickle.dump(model1, open(filename, 'wb'))
#loaded_model = pickle.load(open("stock_prediction_lstm.h5", 'rb'))


