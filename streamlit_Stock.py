import streamlit as st

# Data Handling and Processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import seaborn as sns

# Financial Data
import yfinance as yf

# Time Series Analysis
from arch import arch_model

# Statistical Tests and Metrics
from scipy.stats import norm  # for probability calculations
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")  # Ignore convergence warnings


# Lightweight Charts (Web-Based Visualization)
from plotly.subplots import make_subplots


#=======================================================================================================================
# Start of the code
#=======================================================================================================================

def combine_data(*dataframes_with_symbols):
    """
    Combine multiple stock datasets based on their index. Each dataset must have a 'Date' column
    or a DatetimeIndex. The stock symbols are passed explicitly.
    """
    combined_data = None

    for df, stock_symbol in dataframes_with_symbols:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Each input should be a pandas DataFrame.")

        # Ensure the DataFrame has a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            else:
                raise ValueError("DataFrame must have a 'Date' column or a DatetimeIndex.")

        # Optionally drop 'Date' column if it exists as a regular column
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])

        # Rename columns to include the stock symbol
        df.columns = [f"{col}_{stock_symbol}" for col in df.columns]

        # Concatenate along the index using an inner join to keep only common dates
        if combined_data is None:
            combined_data = df
        else:
            combined_data = pd.concat([combined_data, df], axis=1, join='inner')

    return combined_data

# =======================================================================================================================
# Streamlit title
# =======================================================================================================================

st.title("Stock Analysis Dashboard")
st.subheader('----------------------------------------------------------')

# =======================================================================================================================
# Main Stock Input boton
# =======================================================================================================================

# Initialize a list in session state to store tickers if it doesn't exist
if "stock_list" not in st.session_state:
    st.session_state.stock_list = []

# Input field for stock ticker
ticker = st.text_input("Enter a stock ticker for analysis:", value="SPY")

# Button to add ticker to the list
if st.button("Add Stock"):
    if ticker:
        if ticker.upper() not in st.session_state.stock_list:
            st.session_state.stock_list.append(ticker.upper())
            st.success(f"Added {ticker.upper()} to the analysis list!")
        else:
            st.warning(f"{ticker.upper()} is already in the list.")
    else:
        st.error("Please enter a valid ticker.")

# Display tickers along with a delete button for each
if st.session_state.stock_list:
    st.write("Tickers for analysis:")
    for i, t in enumerate(st.session_state.stock_list):
        col1, col2 = st.columns([3, 1])
        col1.write(t)
        if col2.button("Delete", key=f"delete_{i}"):
            st.session_state.stock_list.pop(i)
            try:
                st.experimental_rerun()
            except st.runtime.scriptrunner.script_run_context.RerunException:
                pass

# Create a list to store processed data for each ticker
processed_data_list = []
information = []

# Process each ticker for analysis
if st.session_state.stock_list:
    for t in st.session_state.stock_list:
        try:
            # Download all available historical data
            stock_data = yf.download(t, period="max")
            if stock_data.empty:
                st.error(f"Error: {t} has been delisted or was not found. Please try a different ticker.")
                continue
            else:
                st.success(f"Stock data for {t} fetched successfully!")

                # Clean the data
                stock_data.columns = [col[0] for col in stock_data.columns]  # Flatten multi-level columns if any
                stock_data.reset_index(inplace=True)
                stock_data['Date'] = pd.to_datetime(stock_data['Date'].apply(lambda x: x.date()))
                stock_data['Volume'] = stock_data['Volume'].astype(float)


                stock_data['Return'] = stock_data['Close'].pct_change()
                stock_data.dropna(subset=['Return'], inplace=True)
                stock_data['Cumulative Return'] = (1 + stock_data['Return']).cumprod() - 1

                numerical_stats = stock_data.drop(columns=['Date','Cumulative Return']).agg(
                    ['mean', 'median', 'count', 'std', 'max', 'min', 'var','kurt','skew']
                ).round(2)

                st.header(f"Statistics for {t}")
                st.write(numerical_stats)

#=======================================================================================================================
                #plotting candletick chart
#=======================================================================================================================

                st.title(f"Stock Chart for {t}")

                # Set title using ticker_name if provided, else fallback to DataFrame attribute
                title = t if t else stock_data.attrs.get("name", "Stock")

                # Determine the current price as the last closing price in the dataset
                current_price = stock_data['Close'].iloc[-1]

                # Create subplots: 2 rows, one for candlestick, one for volume
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03,
                                    row_heights=[0.7, 0.3],
                                    subplot_titles=(f'Candlestick Chart of {title}', 'Volume'))

                # Add candlestick trace
                fig.add_trace(
                    go.Candlestick(
                        x=stock_data['Date'],
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        name='Price'
                    ),
                    row=1, col=1
                )

                # Add volume as a bar chart
                fig.add_trace(
                    go.Bar(
                        x=stock_data['Date'],
                        y=stock_data['Volume'],
                        marker_color='blue',
                        name='Volume'
                    ),
                    row=2, col=1
                )

                # Add a horizontal line for the current price on the candlestick chart
                fig.add_shape(
                    type="line",
                    x0=stock_data['Date'].min(),
                    y0=current_price,
                    x1=stock_data['Date'].max(),
                    y1=current_price,
                    line=dict(color="RoyalBlue", width=2, dash="dot"),
                    row=1, col=1
                )

                # Annotate the current price on the chart
                fig.add_annotation(
                    x=stock_data['Date'].iloc[-1],
                    y=current_price,
                    text=f"Current Price: {current_price:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30,
                    row=1, col=1
                )

                # Update layout
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False,  # Hide range slider
                    template='plotly_white'
                )

                # Plot the chart in Streamlit
                st.plotly_chart(fig)

# =======================================================================================================================
#               plotting returns
# =======================================================================================================================

                st.title("Plotting Returns")

                if t:
                    stock_data = yf.download(t, period="max")
                    if stock_data.empty:
                        st.error(f"Could not fetch data for {t}.")
                    else:
                        # Reset index and ensure the Date column is datetime, then set as index
                        stock_data.reset_index(inplace=True)
                        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                        stock_data = stock_data.set_index('Date').sort_index()

                        # Calculate returns and add as a new column
                        stock_data['Return'] = stock_data['Close'].pct_change()
                        returns = stock_data['Return']

                        # ------------------------- Graph 1: Histogram of Returns -------------------------
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        ax2.hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
                        ax2.set_title(f'Distribution of Returns for {t}')
                        ax2.set_xlabel('Return')
                        ax2.set_ylabel('Frequency')
                        ax2.grid(True)
                        st.pyplot(fig2)


                        # ------------------------- Graph 2: Drawdown -------------------------
                        cumulative_returns = (1 + stock_data['Close'].pct_change()).cumprod()
                        rolling_max = cumulative_returns.cummax()
                        drawdown = (cumulative_returns - rolling_max) / rolling_max

                        fig3, ax4 = plt.subplots(figsize=(12, 6))
                        ax4.plot(drawdown.index, drawdown, label='Drawdown', color='purple', alpha=0.7)
                        ax4.scatter(drawdown.idxmin(), drawdown.min(), color='red', label='Max Drawdown', zorder=3)
                        ax4.set_title(f'Drawdown for {t}')
                        ax4.set_xlabel("Date")
                        ax4.set_ylabel('Drawdown')
                        ax4.legend()
                        ax4.grid(True)
                        # Format the x-axis as dates
                        ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.xticks(rotation=45)
                        st.pyplot(fig3)

# =======================================================================================================================
#                       Fundamentals (Industry/Earnings)
# =======================================================================================================================

                st.title("Fundamental Information")

                ticker_input = t
                if ticker_input:
                    ticker_obj = yf.Ticker(ticker_input)
                    info = ticker_obj.info

                    st.subheader(f"Data for {ticker_input}")
                    st.write(info)  # Optionally display raw info

#=======================================================================================================================
#               Buy/Sell Recommendation (EMA Strat)
#=======================================================================================================================

                st.title("Buy/Sell Reccomendation (not financial advice)")

                conf_int = 0.95

                # Before running the EMA strategy, ensure the DataFrame has a "Date" column
                if 'Date' not in stock_data.columns:
                    stock_data = stock_data.reset_index()  # Bring the Date index back as a column

                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)

                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data.set_index('Date', inplace=True)
                stock_data.sort_index(inplace=True)

                # Drop rows with NaN values resulting from the shift
                stock_data.dropna(subset=['Close'], inplace=True)

                # Calculate the EMA and EWM Std Dev
                ema = stock_data['Close'].ewm(span=30, adjust=False).mean()
                ewm_std = stock_data['Close'].ewm(span=30, adjust=False).std()

                # Calculate Z-scores
                z_scores = (stock_data['Close'] - ema) / ewm_std

                # Calculate probability of an upward move
                stock_data['Probability Up'] = 1 - norm.cdf(z_scores)

                # Define dynamic thresholds based on probabilities
                z_threshold = norm.ppf(1 - conf_int)
                upper_boundary = ema + (z_threshold * ewm_std)
                lower_boundary = ema - (z_threshold * ewm_std)


                # Calculate the Volume-weighted RSI
                def calculate_rsi(stock_data, period=14):
                    delta = stock_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    volume_weighted_gain = (gain * stock_data['Volume']).rolling(window=period).mean()
                    volume_weighted_loss = (loss * stock_data['Volume']).rolling(window=period).mean()
                    rs = volume_weighted_gain / volume_weighted_loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi


                stock_data['RSI'] = calculate_rsi(stock_data, period=14)

                # Define trend based on short-term and long-term EMAs
                stock_data['short_term_ema'] = stock_data['Close'].ewm(span=30, adjust=False).mean()
                stock_data['long_term_ema'] = stock_data['Close'].ewm(span=200, adjust=False).mean()
                stock_data['trend'] = np.where(stock_data['short_term_ema'] > stock_data['long_term_ema'], 'uptrend',
                                               'downtrend')

                # Apply weighted Z-score based on trend direction
                stock_data['z_score_weighted'] = np.where(stock_data['trend'] == 'downtrend', z_scores * 1.5,
                                                          z_scores * 0.5)


                # Calculate MACD with Volume weighting
                def calculate_macd(stock_data, fast_period=12, slow_period=26, signal_period=9):
                    fast_ema = stock_data['Close'].ewm(span=fast_period, adjust=False).mean()
                    slow_ema = stock_data['Close'].ewm(span=slow_period, adjust=False).mean()
                    macd = fast_ema - slow_ema
                    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
                    weighted_signal = signal_line * (1 + stock_data['Volume'] / stock_data['Volume'].mean())
                    return macd, weighted_signal


                stock_data['macd'], stock_data['macd_signal'] = calculate_macd(stock_data)


                # Calculate GARCH-based volatility prediction (with scaling)
                def calculate_garch_volatility(stock_data, scaling_factor=100):
                    returns_series = stock_data['Close'].pct_change().dropna()
                    returns_rescaled = returns_series * scaling_factor
                    garch_model = arch_model(returns_rescaled, vol='Garch', p=1, q=1)
                    garch_fit = garch_model.fit(disp="off")
                    predicted_volatility_scaled = garch_fit.conditional_volatility
                    predicted_volatility = predicted_volatility_scaled / scaling_factor
                    return predicted_volatility


                stock_data['predicted_volatility'] = calculate_garch_volatility(stock_data, scaling_factor=100)


                # Calculate On-Balance Volume (OBV)
                def calculate_obv(stock_data):
                    obv = [0] * len(stock_data)
                    for i in range(1, len(stock_data)):
                        if stock_data['Close'].iloc[i] > stock_data['Close'].iloc[i - 1]:
                            obv[i] = obv[i - 1] + stock_data['Volume'].iloc[i]
                        elif stock_data['Close'].iloc[i] < stock_data['Close'].iloc[i - 1]:
                            obv[i] = obv[i - 1] - stock_data['Volume'].iloc[i]
                        else:
                            obv[i] = obv[i - 1]
                    stock_data['OBV'] = obv
                    return stock_data


                stock_data = calculate_obv(stock_data)

                # Define Buy and Sell signals based on indicators
                buy_signals = stock_data[
                    (stock_data['Close'] < lower_boundary) &
                    (stock_data['RSI'] < 30) &
                    (stock_data['macd'] > stock_data['macd_signal']) &
                    (stock_data['predicted_volatility'] < stock_data['predicted_volatility'].quantile(0.75))
                    ]

                sell_signals = stock_data[
                    (stock_data['Close'] > upper_boundary) &
                    (stock_data['RSI'] > 70) &
                    (stock_data['macd'] < stock_data['macd_signal'] * 1.5) &
                    (stock_data['predicted_volatility'] < stock_data['predicted_volatility'].quantile(0.75))
                    ]

                # Apply seasonal adjustment as an example
                stock_data['Month'] = stock_data.index.month
                stock_data['Day of Week'] = stock_data.index.dayofweek
                seasonal_adjustment = stock_data['Month'].apply(lambda x: 1.2 if x == 12 else 1)
                buy_signals = buy_signals.copy()
                sell_signals = sell_signals.copy()
                buy_signals.loc[:, 'Adjusted'] = buy_signals['Close'] * seasonal_adjustment.loc[buy_signals.index]
                sell_signals.loc[:, 'Adjusted'] = sell_signals['Close'] * seasonal_adjustment.loc[sell_signals.index]

                # Create the Plotly figure for the EMA strategy
                fig4 = go.Figure()

                # Add Close Price Line
                fig4.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
                                          mode='lines', name='Close Price',
                                          line=dict(color='blue', width=2), opacity=0.6))

                # Add 30-Day EMA
                fig4.add_trace(go.Scatter(x=stock_data.index, y=ema,
                                          mode='lines', name='30-Day EMA',
                                          line=dict(color='red', dash='dash'), opacity=0.8))

                # Add Upper and Lower Boundaries
                fig4.add_trace(go.Scatter(x=stock_data.index, y=upper_boundary,
                                          mode='lines', name=f'+{int(conf_int * 100)}% Threshold',
                                          line=dict(color='green', dash='dash'), opacity=0.8))

                fig4.add_trace(go.Scatter(x=stock_data.index, y=lower_boundary,
                                          mode='lines', name=f'-{int(conf_int * 100)}% Threshold',
                                          line=dict(color='orange', dash='dash'), opacity=0.8))

                # Add MACD Histogram
                fig4.add_trace(go.Bar(x=stock_data.index, y=stock_data['macd'] - stock_data['macd_signal'],
                                      name='MACD Histogram', marker=dict(color='purple', opacity=0.3)))

                # Add Buy Signals
                fig4.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Adjusted'],
                                          mode='markers', name='Buy Signal (RSI < 30)',
                                          marker=dict(color='green', size=8, symbol='circle')))

                # Add Sell Signals
                fig4.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Adjusted'],
                                          mode='markers', name='Sell Signal (RSI > 70)',
                                          marker=dict(color='red', size=8, symbol='circle')))

                # Highlight and annotate the last price
                last_price = stock_data['Close'].iloc[-1]
                last_date = stock_data.index[-1]
                fig4.add_trace(go.Scatter(x=[last_date], y=[last_price],
                                          mode='markers+text', name='Last Price',
                                          marker=dict(color='black', size=10),
                                          text=[f'{last_price:.2f}'], textposition="bottom right"))

                # Layout Customization
                fig4.update_layout(
                    title=f'EMA-Based Mean Reversion Strategy with RSI, MACD, and Volatility for {t}',
                    xaxis_title='Date',
                    yaxis_title='Close Price',
                    legend=dict(x=0, y=1),
                    template='plotly_white'
                )

                # Show the interactive Plotly chart in Streamlit
                st.plotly_chart(fig4)

#=======================================================================================================================
#               Yahoo finance Analyst Recommendation
#=======================================================================================================================

                st.title(f"Yahoo Finance Analyst Recommendations for {t}")

                if t:
                    ticker_obj = yf.Ticker(t)
                    recs = ticker_obj.recommendations

                    if recs is None or recs.empty:
                        st.error(f"No analyst recommendations available for {t}.")
                    else:
                        st.subheader(f"Raw Recommendations for {t}")
                        st.dataframe(recs)

                        # The DataFrame likely has columns like: [period, strongBuy, buy, hold, sell, strongSell]
                        # We need to unpivot these columns into a single "Action" column
                        # so we can create a stacked bar chart.
                        # Check if 'period' is in the columns:

                        if 'period' in recs.columns:
                            # Melt (unpivot) the DataFrame
                            recs_melt = recs.melt(
                                id_vars='period',
                                var_name='Action',
                                value_name='Count'
                            )

                            # Create a stacked bar chart
                            fig5 = px.bar(
                                recs_melt,
                                x='period',
                                y='Count',
                                color='Action',
                                barmode='stack',
                                title=f"Analyst Recommendations for {t} Over Periods",
                                labels={"period": "Period", "Action": "Recommendation Type", "Count": "Number"}
                            )
                            st.plotly_chart(fig5)
                        else:
                            st.error("No 'period' column found in the DataFrame. Unable to create stacked bar chart.")

#=======================================================================================================================
#               Arima Forecast
#=======================================================================================================================

                st.title(f"ARIMA Forecast {t}")

                forecast_steps = 365  # Number of days to forecast
                forecast_results = {}  # Dictionary to store each ticker's forecast DataFrame

                series = stock_data['Close']
                # Ensure the index has frequency (assume business days).
                if series.index.freq is None:
                    series = series.asfreq('B')

                try:
                    # Fit a SARIMAX model with seasonality.
                    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
                    model_fit = model.fit(disp=False)

                    # Get forecast for the specified number of steps.
                    forecast_result = model_fit.get_forecast(steps=forecast_steps)
                    forecast_mean = forecast_result.predicted_mean
                    conf_int = forecast_result.conf_int()

                    # Standard error of the forecast (used for probability calculations).
                    std_err = forecast_result.se_mean  # 1D array-like of standard errors
                except Exception as e:
                    print(f"SARIMAX forecast failed for ticker {ticker}: {e}")
                    continue

                # Create forecast index starting one day after the last observed date.
                last_date = series.index[-1]
                forecast_index = pd.date_range(
                    last_date + pd.Timedelta(days=1),
                    periods=forecast_steps,
                    freq='B'
                )

                # Build a forecast DataFrame.
                forecast_df = pd.DataFrame({
                    'Forecast': forecast_mean,
                    'Lower CI': conf_int.iloc[:, 0],
                    'Upper CI': conf_int.iloc[:, 1]
                }, index=forecast_index)

                # Rename columns to include the ticker symbol (e.g., "spy_forecast").
                forecast_df.columns = [
                    f"{ticker.lower()}_{col.replace(' ', '_').lower()}"
                    for col in forecast_df.columns
                ]

                # Example probability: Probability forecast will be above the last historical close
                current_price = series.iloc[-1]
                forecast_col = f"{ticker.lower()}_forecast"
                lower_ci_col = f"{ticker.lower()}_lower_ci"
                upper_ci_col = f"{ticker.lower()}_upper_ci"

                # Compute probability that the forecast is above current_price on each day
                # P(X > threshold) = 1 - CDF(threshold)
                prob_col = f"{ticker.lower()}_prob_above_current"
                forecast_df[prob_col] = 1 - norm.cdf(
                    x=current_price,
                    loc=forecast_df[forecast_col],
                    scale=std_err
                )

                # Store the forecast results in a dictionary
                forecast_results[ticker] = forecast_df

                # --------------------- Plotly Plot --------------------- #
                fig6 = go.Figure()

                # Plot historical data
                fig6.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series,
                        mode='lines',
                        name='Historical Close'
                    )
                )

                # Plot forecast
                fig6.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[forecast_col],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red')
                    )
                )

                # Fill between lower and upper confidence intervals
                fig6.add_trace(
                    go.Scatter(
                        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                        y=list(forecast_df[upper_ci_col]) + list(forecast_df[lower_ci_col][::-1]),
                        fill='toself',
                        fillcolor='rgba(255, 192, 203, 0.3)',  # pinkish
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name='95% CI'
                    )
                )

                # Probability line on a secondary y-axis
                fig6.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[prob_col],
                        mode='lines',
                        name='Prob(Above Current)',
                        line=dict(color='blue'),
                        yaxis='y2'
                    )
                )

                # --------------------- Annotations --------------------- #
                # Current (last historical) price
                fig6.add_annotation(
                    x=series.index[-1],
                    y=current_price,
                    text=f"Current: {current_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xanchor='left',
                    yanchor='bottom'
                )

                # Last forecasted price
                forecast_last_price = forecast_df[forecast_col].iloc[-1]
                forecast_last_date = forecast_df.index[-1]
                fig6.add_annotation(
                    x=forecast_last_date,
                    y=forecast_last_price,
                    text=f"Forecast End: {forecast_last_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xanchor='left',
                    yanchor='bottom'
                )

                # Top price in the upper CI
                top_price = forecast_df[upper_ci_col].max()
                top_date = forecast_df[upper_ci_col].idxmax()
                fig6.add_annotation(
                    x=top_date,
                    y=top_price,
                    text=f"Top: {top_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xanchor='left',
                    yanchor='bottom'
                )

                # Bottom price in the lower CI
                bottom_price = forecast_df[lower_ci_col].min()
                bottom_date = forecast_df[lower_ci_col].idxmin()
                fig6.add_annotation(
                    x=bottom_date,
                    y=bottom_price,
                    text=f"Bottom: {bottom_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xanchor='left',
                    yanchor='top'
                )

                # --------------------- Figure Layout --------------------- #
                fig6.update_layout(
                    title=f"SARIMAX Forecast for {ticker}",
                    xaxis_title="Date",
                    yaxis_title="Close Price",
                    template="plotly_white",
                    legend=dict(x=0, y=1.05, orientation='h'),
                    yaxis2=dict(
                        title="Probability",
                        overlaying='y',
                        side='right',
                        range=[0, 1]  # Probability range
                    )
                )

                st.plotly_chart(fig6)

                stock_data['Cumulative Return'] = (1 + stock_data['Return']).cumprod() - 1

                # Append the processed data and its symbol to the list
                processed_data_list.append((stock_data, t))

        except Exception as e:
            st.error(f"An error occurred for {t}: {e}")

# Combine the data from all tickers if available
if processed_data_list:
    combined_data = combine_data(*processed_data_list)
    st.header("Combined Data")
    st.write(combined_data)
else:
    st.write("No data found")

# ========================================================================================================================
#               Comparative Performance
# ========================================================================================================================

# Define default values based on combined_data if not defined
if 'combined_data' in globals() and not combined_data.empty:
    # Use "max" as default period if not defined
    if 'period' not in globals():
        period = "max"
    # Set start_date and end_date from the combined_data index if not defined
    if 'start_date' not in globals() or start_date is None:
        start_date = combined_data.index.min().strftime('%Y-%m-%d')
    if 'end_date' not in globals() or end_date is None:
        end_date = combined_data.index.max().strftime('%Y-%m-%d')

    # Create a figure with two subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 14), sharex=True)

    # ------------------------- Subplot 1: Adjusted Close Prices -------------------------
    for tick in st.session_state.stock_list:
        close_col = f"Close_{tick}"
        if close_col in combined_data.columns:
            last_price = combined_data[close_col].iloc[-1]
            last_date = combined_data.index[-1]
            last_price_formatted = f"{last_price:,.2f}"
            ax1.plot(combined_data.index, combined_data[close_col],
                     label=f"{tick} Close = {last_price:,.2f}")
            ax1.scatter(last_date, last_price, color='red', zorder=5)
            ax1.text(last_date, last_price, f" {last_price_formatted}",
                     fontsize=9, color='black',
                     horizontalalignment='left', verticalalignment='bottom',
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

    # Set title for Subplot 1 based on provided date range or period
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        years_diff = round((end_dt - start_dt).days / 365.25, 0)
        ax1.set_title(f"Adjusted Close Prices for All Tickers over a {years_diff}-year Period")
    except Exception:
        ax1.set_title(f"Adjusted Close Prices for All Tickers over the {period} Period")

    ax1.set_ylabel("Adjusted Close")
    ax1.legend(fontsize=14)
    ax1.grid(True)

    # ------------------------- Subplot 2: Cumulative Returns -------------------------
    for tick in st.session_state.stock_list:
        cum = f"Cumulative Return_{tick}"
        if cum in combined_data.columns:
            final_return = combined_data[cum].iloc[-1] * 100  # Convert to percentage
            final_date = combined_data.index[-1]
            final_return_formatted = f"{final_return:,.2f}"
            ax2.plot(combined_data.index, combined_data[cum] * 100,
                     label=f"{tick} Cumulative Return = {final_return:,.2f}%")
            ax2.scatter(final_date, final_return, color='red', zorder=5)
            ax2.text(final_date, final_return, f" {final_return_formatted}%",
                     fontsize=9, color='black',
                     horizontalalignment='left', verticalalignment='bottom',
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

    try:
        ax2.set_title(f"Cumulative Returns Over Time over a {years_diff}-year Period")
    except Exception:
        ax2.set_title(f"Cumulative Returns Over Time over the {period} Period")

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.legend(fontsize=14)
    ax2.grid(True)

    # Format the x-axis to display dates nicely
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("No combined data available.")

# ========================================================================================================================
#               Correlation Matrix
# ========================================================================================================================

# Ensure combined_data exists and is not empty
if 'combined_data' in globals() and not combined_data.empty:
    # Extract only the columns that represent adjusted close prices.
    # This assumes that the column names for close prices start with "Close_"
    close_columns = [col for col in combined_data.columns if col.startswith("Close_")]
    adj_close_data = combined_data[close_columns]

    # Drop columns that are entirely NaN to avoid errors during correlation calculation
    adj_close_data = adj_close_data.dropna(axis=1, how="all")

    # Compute the correlation matrix
    correlation_matrix = adj_close_data.corr()

    # Create a figure for the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="icefire", fmt=".2f", square=True,
                annot_kws={'size': 10}, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Adjusted Close Prices")
    plt.tight_layout()
    st.pyplot(plt.gcf())
else:
    st.write("No combined data available for correlation matrix.")