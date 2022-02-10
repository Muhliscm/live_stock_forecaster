
import yfinance as yf
import streamlit as st
import sys
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
import pandas as pd
from tensorflow import keras
#from components import extData


sBar = st.sidebar
rad = sBar.radio("Navigation", ["Home", "Compare Stock", "Forecast Stock"])

st.title("Stock Price Prediction")



if rad=="Home":
    
    st.image("images\\stock_market.jpg")
    st.header("Stock Visualization")
    symb = st.text_input("Enter Stock Symbol (eg: AAPL for Apple)")
    st.write("Add .ns at end if you want indian stocks (Eg:tatamotors.ns)")
    symb = symb.upper()

    if st.button("Search"):

        try:
            tik = yf.Ticker(symb)
            info = tik.info
            st.image(info['logo_url'])
            name = info['shortName']
        
        except:
            st.error("Check your input")
            sys.exit()

        df = tik.history(period='max')

        # df = extData(symb)[0]
        # name = extData(symb)[1]

        fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])],
                )
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(title=name,yaxis_title='Price', xaxis_title="Date")  
        st.plotly_chart(fig)
        
    

        # if st.checkbox("for more information"):
        #     tik = yf.Ticker(symb)
        #     info = tik.info
        #     st.write(info)


if rad=="Compare Stock":
    st.header("Compare Stocks")
    st.write("Add .ns at end if you want indian stocks (Eg:tatamotors.ns)")
    st1, st2 = st.columns(2)
    st1 = st1.text_input("Enter 1st stock symbol")
    st1 = st1.upper()
    st2 = st2.text_input("Enter 2nd stock symbol")
    st2 = st2.upper()

    comp = st.button("Compare")

    if st1 and st2 and comp:
        # st.write(st1, st2)

        try:
            tik1 = yf.Ticker(st1)
            tik2 = yf.Ticker(st2)
        
        except:
            st.error("Check your input")
            sys.exit()

        info1 = tik1.info
        #st.write(info1)
        info2 = tik2.info

        img1, img2 = st.columns(2)
        img1.image(info1['logo_url'])
        name1 = info1['longName']
        img1.write(name1)
        
        #st.write(name1)

        img2.image(info2['logo_url'])
        name2 = info2['longName']
        img2.write(name2)
        #st.write(name2)

        df1 = tik1.history(period='max')
        fig1 = go.Figure(data=[go.Candlestick(x=df1.index,
                open=df1['Open'],
                high=df1['High'],
                low=df1['Low'],
                close=df1['Close'])],
                )
        fig1.update_layout(xaxis_rangeslider_visible=False)
        fig1.update_layout(title=name1,yaxis_title='Price', xaxis_title="Date")  
        st.plotly_chart(fig1)


        df2 = tik2.history(period='max')
        fig2 = go.Figure(data=[go.Candlestick(x=df2.index,
                open=df2['Open'],
                high=df2['High'],
                low=df2['Low'],
                close=df2['Close'])],
                )
        fig2.update_layout(xaxis_rangeslider_visible=False)
        fig2.update_layout(title=name2,yaxis_title='Price', xaxis_title="Date")  
        st.plotly_chart(fig2)


if rad=="Forecast Stock":
    
    

    st.header("Stock Price Prediction")
    symb = st.text_input("Enter Stock Symbol (eg: AAPL for Apple)")
    st.write("Add .ns at end if you want indian stocks (Eg:tatamotors.ns)")
    symb = symb.upper()

    
    days = st.selectbox("Number of days", [90, 180, 270, 360, 450, 540, 630, 720])
    search = st.button("Predict")
    st.write("For education purpose only")

    if search and days:

        try:
            tick = yf.Ticker(symb)
            info = tick.info
            st.image(info['logo_url'])
            name = info['shortName']
        
        except:
            st.error("Check your input")
            sys.exit()

        stock_hist = tick.history("max")

        fig, ax = plt.subplots()
        plt.suptitle(name, y=1.05, fontsize=20)
        ax.plot(stock_hist["Close"])
        plt.xlabel("Year")
        plt.ylabel("Price")
        plt.title("Current stock price")
        st.pyplot(fig)

        #getting number of days from selection box
        days = days


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
        @st.cache
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
        fig, ax = plt.subplots()
        plt.title(f"Predicted Stock Price for {days} Days")
        ax.plot(forecast_dates,  y_pred_future)
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot(fig)

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
        fig, ax = plt.subplots()
        plt.title("Actual Price Vs Predicted Price")
        ax.plot(shorted['Date'], shorted['Close'], label = 'Actual')
        plt.plot(df_future["Date"], df_future["Close"], label="Predicted")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig)

        #printing dataframe
        st.header("Prediction data")
        st.dataframe(df_future)

            





            
                
            

            
