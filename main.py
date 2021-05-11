import streamlit as st
import time
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# 1. Build UI Layout
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "COIN", "MSFT", "DIS", "BABA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# st.write("selected", selected_stock)
# st.write("period", period)

# 2. Load the stock data
# NOTE We can cache the data using decorator
@st.cache
def load_data(ticker):
    df = yf.download(ticker, START, TODAY)
    df.reset_index(inplace=True)
    return df


# Add a loading spinner
with st.spinner("Loading data..."):
    data = load_data(selected_stock)
    time.sleep(1)
    # st.success("Done!")
    # st.balloons()


# 3. Display chart and table
# Create the plotly chart from downloaded data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Display to the dashboard
st.subheader(f"{selected_stock} | {n_years} years | {data.shape[0]} entries")
st.dataframe(data.tail())
plot_raw_data()

# 4. Forecasting with Facebook Prophet
# Create training dataframe
df_train = data[["Date", "Close"]]
# Rename the columns per fbprophet docs
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# st.dataframe(df_train)

# Create the model
m = Prophet()
m.fit(df_train)

# Create a future dataframe for the forecast
future = m.make_future_dataframe(periods=period)

# Create a forecast DataFrame based on our future DataFrame
forecast = m.predict(future)

# 5. Output the predictions/results.
# NOTE Using built-in fbprophet.plot.plot_plotly
st.subheader(f"Forecast Data & Components: {selected_stock}")
st.write(forecast.tail())

st.write("Forecast Data")
forecast_fig = plot_plotly(m, forecast)
st.plotly_chart(forecast_fig)

# NOTE Using built-in fbprophet.plot_components() (not plotly graph)
st.write("Forecast Components")
forecast_components_fig = m.plot_components(forecast)
# Display using write() not plotly_chart()
st.write(forecast_components_fig)
