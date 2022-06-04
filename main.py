# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
# import plotly.graph_objs as go
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from PIL import Image

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# image = Image.open('photo.jpg')
# st.image(image, caption='Forecasting stock price')

st.title('Forecasting Stock Price')

st.header("info")
st.markdown(
    'Forecasting stock price under development this program shows the average of high and low of that particular '
    'day or time there will be no direct prices shown on the reult because thats not how stock market works use '
    'your own sense and knowledge before using the app it will show the prediction upto 4 years with high '
    'accurracy there are so so so many factors that can affect the stock prices stock prices changes every second '
    'and if you are thinking it will show perfect number then its not prediction .')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
default_stock = 'AAPL'
# selected_stock = st.selectbox('Select dataset for prediction', stocks)

selected_stock = st.text_input('Enter the symbol', default_stock)

# n_years = st.slider('Years of prediction:', 1, 4)
default_years = 1
n_years = st.text_input('prediction for next ', default_years)
period = int(n_years) * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


# col1, col2, col3, col4 = st.columns(4)
# col1.metric("OPEN", "71", "1.2 °F")
# col2.metric("HIGH", "9 mph", "-8%")
# col3.metric("LOW", "86%", "4%")
# col4.metric("LOW", "86%", "4%")

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

st.subheader(f'summary of {selected_stock} stock from 2015 - 2022')
st.write(data.describe())


# Plot raw data
def plot_raw_data():
    # st.subheader('Raw data')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

st.info('This is a purely informational message')


# st.write(data.info())


def test_features():
    import plotly.graph_objects as go

    live_data = yf.download(tickers=selected_stock, period='1d', interval='1m')
    # st.dataframe(live_data)

    # Candle sticks

    # figure

    st.subheader('Candle Sticks')

    fig4 = go.Figure()
    fig4.add_trace(go.Candlestick(x=live_data.index,
                                  open=live_data['Open'],
                                  high=live_data['High'],
                                  low=live_data['Low'],
                                  close=live_data['Close'],
                                  name='market data'))
    # titles
    fig4.update_layout(
        title='live stock price',
        yaxis_title='stock price usd per share',
        # height=600
    )

    # X- axis
    fig4.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list((
                dict(count=15, label='15m', step="minute", stepmode="backward"),
                dict(count=45, label='45m', step="minute", stepmode="backward"),
                dict(count=1, label='HTD', step="hour", stepmode="todate"),
                dict(count=2, label='2h', step="hour", stepmode="backward"),
                dict(step='all')
            ))
        )
    )

    st.write(fig4)


test_features()

st.caption("© 2021 - 2022 Forecasting Stock Price - All Rights Reserved.")


# Test features
# Hiding non-buisness hours in candle sticks

def candlesticks_predicted():
    pass


candlesticks_predicted()


def download_historical_data():
    # stock_data = yf.download(tickers=selected_stock, period='1d', interval='1m')
    stock_data = yf.Ticker(selected_stock)

    st.sidebar.subheader("""Display Additional Information""")
    # checkbox to display stock actions for the searched ticker
    actions = st.sidebar.checkbox("Stock Actions")
    if actions:
        st.subheader("""Stock **actions** for """ + selected_stock)
        display_action = stock_data.actions
        if display_action.empty:
            st.write("No data available at the moment")
        else:
            st.write(display_action)

    # checkbox to display quarterly financials for the searched ticker
    financials = st.sidebar.checkbox("Quarterly Financials")
    if financials:
        st.subheader("""**Quarterly financials** for """ + selected_stock)
        display_financials = stock_data.quarterly_financials
        if display_financials.empty:
            st.write("No data available at the moment")
        else:
            st.write(display_financials)

    # checkbox to display list of institutional shareholders for searched ticker
    major_shareholders = st.sidebar.checkbox("Institutional Shareholders")
    if major_shareholders:
        st.subheader("""**Institutional investors** for """ + selected_stock)
        display_shareholders = stock_data.institutional_holders
        if display_shareholders.empty:
            st.write("No data available at the moment")
        else:
            st.write(display_shareholders)

    # checkbox to display quarterly balance sheet for searched ticker
    balance_sheet = st.sidebar.checkbox("Quarterly Balance Sheet")
    if balance_sheet:
        st.subheader("""**Quarterly balance sheet** for """ + selected_stock)
        display_balancesheet = stock_data.quarterly_balance_sheet
        if display_balancesheet.empty:
            st.write("No data available at the moment")
        else:
            st.write(display_balancesheet)

    # checkbox to display quarterly cashflow for searched ticker
    cashflow = st.sidebar.checkbox("Quarterly Cashflow")
    if cashflow:
        st.subheader("""**Quarterly cashflow** for """ + selected_stock)
        display_cashflow = stock_data.quarterly_cashflow
        if display_cashflow.empty:
            st.write("No data available at the moment")
        else:
            st.write(display_cashflow)

    # checkbox to display quarterly earnings for searched ticker
    earnings = st.sidebar.checkbox("Quarterly Earnings")
    if earnings:
        st.subheader("""**Quarterly earnings** for """ + selected_stock)
        display_earnings = stock_data.quarterly_earnings
        if display_earnings.empty:
            st.write("No data available at the moment")
        else:
            st.write(display_earnings)

    # checkbox to display list of analysts recommendation for searched ticker
    analyst_recommendation = st.sidebar.checkbox("Analysts Recommendation")
    if analyst_recommendation:
        st.subheader("""**Analysts recommendation** for """ + selected_stock)
        display_analyst_rec = stock_data.recommendations
        if display_analyst_rec.empty:
            st.write("No data available at the moment")
        else:
            st.write(display_analyst_rec)


download_historical_data()

st.text(" ")
st.text(" ")


def stock_info():
    tickerSymbol = selected_stock  # Select ticker symbol
    tickerData = yf.Ticker(tickerSymbol)  # Get ticker data

    # # Ticker information
    string_logo = '<img src=%s>' % tickerData.info['logo_url']
    st.markdown(string_logo, unsafe_allow_html=True)

    string_name = tickerData.info['longName']
    st.header('**%s**' % string_name)

    string_summary = tickerData.info['longBusinessSummary']
    st.info(string_summary)


stock_info()
