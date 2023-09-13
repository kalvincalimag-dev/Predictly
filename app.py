from io import BytesIO
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import base64
import datetime
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Constants
PAGE_CONFIG = {
    "page_title": "Predictly - Stock Analysis",
    "page_icon": "📈",
    "layout": "centered",
    "initial_sidebar_state": "expanded",
}

SIDEBAR_MENU = ["Last Trade Prices", "Automated Crossover", "Prediction vs Actual"]
SIDEBAR_ICONS = ["align-end", "clipboard2-data", "graph-up-arrow"]

# Function for Streamlit Configuration
def set_page_config():
    st.set_page_config(**PAGE_CONFIG)

# Function for Logo & Title 
def display_sidebar_title_and_logo():
    st.sidebar.markdown(
        f"""<style>{open("styles.css").read()}</style>""", unsafe_allow_html=True)

    img = Image.open('StLogo.png')

    def image_to_base64(img):
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        return img_str

    st.sidebar.markdown(
        f'''<div class="sidebar-title">
                <div class="logo-container"> <img src="data:image/png;base64,{image_to_base64(img)}" alt="Logo"> </div>
                <h1>Predictly</h1>
            </div"> 
        ''', unsafe_allow_html=True
    )

# Function for Page Selection
def display_selected_page(selected_page):
    if selected_page == "Last Trade Prices":
        display_last_trade_prices()
    elif selected_page == "Automated Crossover":
        display_automated_crossover()
    elif selected_page == "Prediction vs Actual":
        display_prediction_vs_actual()

# Sidebar  
def main():
    set_page_config()
    display_sidebar_title_and_logo()
    with st.sidebar:
        selected_page = option_menu(menu_title=None, options=SIDEBAR_MENU, icons=SIDEBAR_ICONS)
    display_selected_page(selected_page)

    # Bottom Nav Items
    st.markdown(f"""<style>{open("styles.css").read()}</style>""", unsafe_allow_html=True)
    st.sidebar.markdown("<p class='first-sidebar-link'><a href='mailto:calimag.kalvin.d@gmail.com'>Support</a></p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p class='sidebar-link'><a href='#documentation'>Documentation</a></p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p class='sidebar-link'><a href='https://t.me/+WlhCyGZCqBo5OTJl'>Telegram</a></p>", unsafe_allow_html=True)

# Display Functions for each Nav Item
def display_last_trade_prices():
    st.title("Dashboard - Stock Trend Predictor")

    # User Input
    user_input = st.text_input('Enter Stock Ticker Symbol', 'AAPL', help="Enter a Company's Stock Ticker: https://finance.yahoo.com/trending-tickers")

    # Date Range
    start_date = '2010-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    # Download Stock Data
    stockdataframe = yf.download(user_input, start=start_date, end=end_date)

    # Ticker Info
    ticker_info = yf.Ticker(user_input)
    company_name = ticker_info.info['longName']

    # Display Raw Data Summary
    st.subheader(f'Raw Data Summary: {company_name}')
    st.write(stockdataframe.describe().style.set_table_attributes('style="width: 8000px;"'), unsafe_allow_html=True)
    st.write("- The dataset illustrated above contains various historical stock data over the specified time window.")
    
    # Display Closing Prices
    display_closing_prices(stockdataframe)

def display_closing_prices(stockdataframe):
    st.subheader('Last Trade Prices')
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('#F6F7F8')
    plt.plot(stockdataframe.Close, 'b')
    plt.gca().set_facecolor('#F6F7F8')
    st.pyplot(fig)

def display_automated_crossover():
    st.title("Automated Crossover Analytics")

    user_input = st.text_input('Stock Ticker Symbol', 'NFLX')
    start_date = '2010-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    stockdataframe = yf.download(user_input, start=start_date, end=end_date)

    display_crossover_analytics(stockdataframe) 
    
def display_crossover_analytics(stockdataframe):
    st.write("- If SMA100 > SMA200 = Golden Cross, Possible Bullish")
    st.write("- If Closing Price > SMA200 > SMA100 = Death Cross, Possible Bearish")
    simple_moving_avg_100 = stockdataframe.Close.rolling(100).mean()
    simple_moving_avg_200 = stockdataframe.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.gca().set_facecolor('#F6F7F8')
    plt.plot(stockdataframe.Close, 'b')
    plt.plot(simple_moving_avg_100, 'r')
    plt.plot(simple_moving_avg_200, 'g')
    st.pyplot(fig)

def display_prediction_vs_actual():
    st.title("Prediction vs Actual")

    user_input = st.text_input('Stock Ticker Symbol', 'NFLX')
    start_date = '2010-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    stockdataframe = yf.download(user_input, start=start_date, end=end_date)

    # Data Pre-processing Phase 1 - Splitting into Training & Testing
    training_70 = pd.DataFrame(stockdataframe['Close'][0:int(len(stockdataframe) * 0.70)])
    testing_30 = pd.DataFrame(stockdataframe['Close'][int(len(stockdataframe) * 0.70): int(len(stockdataframe))])

    # Data Pre-processing Phase 2 - Scaling/Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))  
    data_training_array = scaler.fit_transform(training_70)

    # Model Integration
    model = load_model('keras_model.keras')  

    # Testing Part
    past_100_days = training_70.tail(100)
    final_dataframe = pd.concat([past_100_days, testing_30], ignore_index=True)
    input_data = scaler.fit_transform(final_dataframe)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    # Reverse Scaling
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final Graph
    dates = stockdataframe.index[int(len(stockdataframe) * 0.70):]
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test, 'b', label='Original Price')
    plt.plot(dates, y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
