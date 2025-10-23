import streamlit as st
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# --- App Configuration ---
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered"
)

# --- Model and Constants ---
MODEL_PATH = "DL_MINI_PROJECT.h5"
TIMESTEPS = 100

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_keras_model():
    """Loads the pre-trained Keras model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_keras_model()

# --- App Title and Description ---
st.title("📈 The Stock Outlook")
st.write(
    "Enter a stock ticker symbol to get the predicted next-day closing price "
    "using a pre-trained LSTM model."
)

# --- User Input ---
ticker = st.text_input("**Enter Stock Ticker Symbol** (e.g., AAPL, GOOGL, MSFT)", "AAPL").upper()
predict_button = st.button("Predict Next Day's Price")

# --- Prediction Logic ---
if predict_button and model is not None:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching data and making prediction for {ticker}..."):
            try:
                # 1. Fetch data and company info
                stock = yf.Ticker(ticker)
                company_name = stock.info.get('longName', ticker) # <-- NEW: Get company name
                df = stock.history(period="200d", interval="1d") # <-- MODIFIED: Use the Ticker object

                if df.empty or len(df) < TIMESTEPS:
                    st.error(f"Not enough data for ticker '{ticker}'. Please try another one.")
                else:
                    # 2. Prepare data for the model
                    close_prices = df['Close'].values.reshape(-1, 1)

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_prices = scaler.fit_transform(close_prices)

                    last_sequence = scaled_prices[-TIMESTEPS:]
                    X_pred = np.array([last_sequence])

                    # 3. Make prediction
                    predicted_price_scaled = model.predict(X_pred)

                    # 4. Inverse transform the prediction
                    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
                    last_actual_price = close_prices[-1][0]

                    # --- Display Results ---
                    # <-- MODIFIED: Display the company name prominently
                    st.subheader(f"Prediction for {company_name} ({ticker})")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"Last Actual Close Price",
                            value=f"${last_actual_price:,.2f}"
                        )
                    with col2:
                        st.metric(
                            label=f"Predicted Next Day's Close Price",
                            value=f"${predicted_price:,.2f}",
                            delta=f"{predicted_price - last_actual_price:,.2f}"
                        )

                    st.subheader("Recent Closing Prices")
                    chart_data = df['Close'].tail(TIMESTEPS)
                    st.line_chart(chart_data)

            except Exception as e:
                st.error(f"An error occurred. The ticker '{ticker}' may be invalid. Please check and try again.")
                st.error(f"Details: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Created for demonstration purposes.")