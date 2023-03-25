
# import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
import pandas as pd
# from tensorflow import keras
from yahooquery import Ticker
# from sklearn.preprocessing import StandardScaler
# from components import extData


def search_stocks(symbol):
    # symbol = symbol.lower()

    try:
        tik = Ticker(symbol)
        name = tik.quote_type.get(symbol, {}).get('shortName', "")
        # st.write(name)

    except Exception as ex:
        tik, name = "", ""
        st.warning(">>>>>>>>>>>>> check your input")
        # st.write(
        #     "exception occurred when searching stock symbol", ex)

    return (tik, name)


def plot_candlestick(tik, name):
    df = tik.history(period="max").reset_index()
    st.write(df.head())

    if any(df):
        fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                             open=df['open'],
                                             high=df['high'],
                                             low=df['low'],
                                             close=df['close'])],
                        )
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(title=name, yaxis_title='Price',
                          xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
        # fig.show()


def prepare_training_data(df_for_training, n_past=14, n_future=1):
    train_X = []

    # creating data for prediction
    for i in range(len(df_for_training)-n_past-1):
        a = df_for_training[i:(i+n_past), 0]  # i=0, 0,1,2,3-----99   100
        train_X.append(a)

    train_X = np.array(train_X)
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], n_future)
    prepared_data = train_X[-1].reshape(-1, 1)
    return prepared_data


# @st.cache_resource
# def pred_and_load(days, temp_for, n_past=14):
#     from tensorflow import keras
#     timestamp = n_past
#     model = keras.models.load_model("de_lstm_14days.h5")
#     li = []
#     for i in range(days):
#         x_input_for = temp_for.reshape(1, temp_for.shape[0], 1)
#         pred_for = model.predict(x_input_for).flatten()
#         li.append(pred_for)
#         next_for = np.append(temp_for, pred_for)
#         temp_for = next_for[-timestamp:]
#     return li


def line_plot(x, y, x_label="Date", y_label="Price", title="Title"):
    fig, ax = plt.subplots()
    plt.suptitle(name, y=1.05, fontsize=20)
    ax.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.title(title)
    st.pyplot(fig)


st.set_page_config(layout="wide")

sBar = st.sidebar
rad = sBar.radio("Navigation", ["Home", "Compare Stock", "Forecast Stock"])

st.title("Stock Price Prediction")

if rad == "Home":

    st.image("images//stock_market.jpg")
    st.header("Stock Visualization")
    symbol = st.text_input("Enter Stock Symbol (eg: AAPL for Apple)")
    st.write("Add .ns at end if you want indian stocks Eg: TATAMOTORS.NS")
    if not symbol:
        st.warning('Please Enter a symbol.')
        st.stop()

    if st.button("Search"):
        st.success('Thank you for input.')

        tik, name = search_stocks(symbol)

        if tik and name:
            plot_candlestick(tik, name)
        else:
            st.error(">>>>>>>> error in input")

if rad == "Compare Stock":
    st.header("Compare Stocks")
    st.write("Add .ns at end if you want indian stocks Eg: TATAMOTORS.NS")
    st1, st2 = st.columns(2)
    st1 = st1.text_input("Enter 1st stock symbol")
    st2 = st2.text_input("Enter 2nd stock symbol")

    if not st1 or not st2:
        st.warning('Please Enter a symbols.')
        st.stop()

    comp = st.button("Compare")

    if st1 and st2 and comp:
        st.success('Thank you for input.')

        tik1, name1 = search_stocks(st1)
        tik2, name2 = search_stocks(st2)
        if tik1 and tik1:
            plot_candlestick(tik1, name1)
            plot_candlestick(tik2, name2)
        else:
            st.waring(">>>>>>>>>>> check your input")

if rad == "Forecast Stock":

    st.header("Stock Price Prediction")
    st.write("For education purpose only")
    symbol = st.text_input("Enter Stock symbol (eg: AAPL for Apple)")
    st.write("Add .ns at end if you want indian stocks (Eg:Tatamotors.ns)")

    days = st.selectbox("Number of days", [
        90, 180, 270, 360, 450, 540, 630, 720])

    if not symbol:
        st.warning('Please Enter a symbol.')
        st.stop()

    search = st.button("Predict")

    if search and days:
        st.success('Thank you for input.')
        tik, name = search_stocks(symbol)
        if tik and name:
            df = tik.history(period='max').reset_index()
            # print(df.head())

            # fig, ax = plt.subplots()
            plt.suptitle(name, y=1.05, fontsize=20)
            line_plot(df['date'], df["close"], title="Current stock price")

        #     df_for_training = df[["close"]].values
        #     prepared_data = prepare_training_data(df_for_training)
        #     # st.write(prepared_data)

        #     if not prepared_data.any():
        #         st.warning(">>>>>>>>>>> prepared data is empty")
        #     else:

        #         sc = StandardScaler()
        #         prepared_data = sc.fit_transform(prepared_data)
        #         li = pred_and_load(days, prepared_data)

        #         arr = np.array(li)
        #         y_pred_future = sc.inverse_transform(arr)
        #         y_pred_future = y_pred_future.flatten()

        #         # creating dates for forecasted days
        #         training_dates = df["date"]
        #         forecast_period_date = pd.date_range(
        #             list(training_dates)[-1], periods=days, freq="1d").tolist()
        #         forecast_dates = []
        #         for time_i in forecast_period_date:
        #             forecast_dates.append(time_i.date())

        #         # creating dataset for future prediction
        #         df_future = pd.DataFrame(
        #             {"Date": np.array(forecast_dates), "Close": y_pred_future})

        #         title = f"Predicted Stock Price for {days} Days"
        #         line_plot(forecast_dates,  y_pred_future, title=title)

        #         # creating filter for dates
        #         from datetime import datetime, timedelta
        #         if days < 365:
        #             back_days = 730
        #         else:
        #             back_days = 1460

        #         original = df[["date", "close"]]
        #         past_date = datetime.now() - timedelta(back_days)
        #         # filter_date = datetime.strftime(past_date.date(), "%Y-%m-%d")
        #         original['date'] = pd.to_datetime(original['date'], utc=True)
        #         shorted = original[original['date'].dt.date > past_date.date()]
        #         # shorted = original[original['date'] > past_date.date()]

        #         # ploting actual vs predicted
        #         fig, ax = plt.subplots()
        #         plt.title("Actual Price Vs Predicted Price")
        #         ax.plot(shorted['date'], shorted['close'], label='Actual')
        #         plt.plot(df_future["Date"],
        #                  df_future["Close"], label="Predicted")
        #         plt.xticks(rotation=45)
        #         plt.xlabel("Date")
        #         plt.ylabel("Price")
        #         plt.legend()
        #         st.pyplot(fig)

        #         # printing dataframe
        #         st.header("Prediction data")
        #         st.dataframe(df_future)
        # else:
        #     st.warning(">>>>>>>>>>>>>> check your input")
