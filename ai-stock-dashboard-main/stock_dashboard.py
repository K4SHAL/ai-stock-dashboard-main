import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def _mapping_number(values, key):
    """Return a finite numeric field from a mapping-like provider response."""
    if values is None:
        return None
    try:
        value = values.get(key)
    except AttributeError:
        try:
            value = values[key]
        except (KeyError, TypeError, IndexError):
            return None
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _format_price(value):
    number = value if isinstance(value, (int, float, np.number)) else None
    if number is None or not np.isfinite(number) or number <= 0:
        return "N/A"
    return f"${number:,.2f}"


def market_snapshot_values(fast_info, info, quote_history):
    """Combine Yahoo Finance quote fields with historical quote fallbacks."""
    if quote_history is None or quote_history.empty:
        return {}

    latest_bar = quote_history.iloc[-1]
    latest_close = _mapping_number({"value": latest_bar.get("Close")}, "value")
    last_price = _mapping_number(fast_info, "last_price")
    if last_price is None:
        last_price = latest_close

    previous_close = _mapping_number(fast_info, "previous_close")
    if previous_close is None:
        previous_close = _mapping_number(info, "previousClose")
    if previous_close is None and len(quote_history) > 1:
        previous_close = _mapping_number({"value": quote_history["Close"].iloc[-2]}, "value")

    open_price = _mapping_number(fast_info, "open")
    day_high = _mapping_number(fast_info, "day_high")
    day_low = _mapping_number(fast_info, "day_low")
    if open_price is None:
        open_price = _mapping_number({"value": latest_bar.get("Open")}, "value")
    if day_high is None:
        day_high = _mapping_number({"value": latest_bar.get("High")}, "value")
    if day_low is None:
        day_low = _mapping_number({"value": latest_bar.get("Low")}, "value")

    session_date = quote_history.index[-1]
    try:
        session_date = session_date.strftime("%b %d, %Y")
    except AttributeError:
        session_date = str(session_date)

    return {
        "last_price": last_price,
        "previous_close": previous_close,
        "open": open_price,
        "high": day_high,
        "low": day_low,
        "session_date": session_date,
    }


def count_reported_institutional_holders(holders):
    """Count named holders present in Yahoo Finance's returned holder table."""
    if holders is None or not isinstance(holders, pd.DataFrame) or holders.empty:
        return None
    if "Holder" not in holders.columns:
        return None
    names = holders["Holder"].dropna().astype(str).str.strip()
    names = names[names.ne("")]
    return int(names.nunique()) if not names.empty else None


def next_earnings_date(calendar, info, today=None):
    """Format the next future earnings date supplied by Yahoo Finance, if any."""
    today = today or datetime.now().date()
    if isinstance(calendar, dict):
        values = calendar.get("Earnings Date")
    elif isinstance(calendar, pd.DataFrame):
        if "Earnings Date" in calendar.index:
            values = calendar.loc["Earnings Date"]
        elif "Earnings Date" in calendar.columns:
            values = calendar["Earnings Date"]
        else:
            values = None
    else:
        values = None
    if values is None and isinstance(info, dict):
        values = info.get("earningsDate")
        if values is None:
            values = info.get("earningsTimestampStart") or info.get("earningsTimestamp")

    if isinstance(values, (list, tuple, pd.Series, np.ndarray)):
        candidates = list(values)
    elif values is None:
        candidates = []
    else:
        candidates = [values]

    dates = []
    for value in candidates:
        if value is None:
            continue
        try:
            if isinstance(value, (int, float, np.integer, np.floating)):
                if not np.isfinite(value):
                    continue
                parsed = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
            else:
                parsed = pd.to_datetime(value, utc=True, errors="coerce")
            if not pd.isna(parsed):
                dates.append(parsed.date())
        except (TypeError, ValueError, OverflowError):
            continue

    future_dates = [date_value for date_value in dates if date_value >= today]
    return min(future_dates).strftime("%b %d, %Y") if future_dates else None


def _set_selected_stock(symbol):
    st.session_state["stock_selector"] = symbol


def _toggle_favorite(symbol):
    favorites = list(st.session_state.get("favorites", []))
    if symbol in favorites:
        favorites.remove(symbol)
    else:
        favorites.append(symbol)
    st.session_state["favorites"] = favorites


def apply_orange_white_theme():
    """Apply a consistent white canvas, orange accents, and accessible motion."""
    st.markdown(
        """
        <style>
        :root { --dashboard-orange: #f97316; --dashboard-orange-dark: #c2410c; }
        [data-testid="stAppViewContainer"] { background: #fff; color: #172033; }
        [data-testid="stHeader"] { background: rgba(255, 255, 255, .92); }
        [data-testid="stSidebar"] { background: #fff7ed; }
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 { color: #172033; }
        div.stButton > button, div.stDownloadButton > button {
            border: 1px solid #fdba74; border-radius: 10px; background: #fff;
            color: #9a3412; font-weight: 600;
            transition: transform .16s ease, background-color .16s ease,
                        color .16s ease, box-shadow .16s ease, border-color .16s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            border-color: var(--dashboard-orange); background: var(--dashboard-orange);
            color: #fff; transform: translateY(-1px);
            box-shadow: 0 5px 14px rgba(249, 115, 22, .20);
        }
        div.stButton > button:active, div.stDownloadButton > button:active {
            transform: scale(.98);
        }
        div.stButton > button:focus-visible, div.stDownloadButton > button:focus-visible {
            outline: 3px solid rgba(249, 115, 22, .32); outline-offset: 2px;
        }
        div.stButton > button[kind="primary"] {
            border-color: var(--dashboard-orange); background: var(--dashboard-orange);
            color: #fff;
        }
        div.stButton > button[kind="primary"]:hover {
            border-color: var(--dashboard-orange-dark); background: var(--dashboard-orange-dark);
        }
        [data-testid="stMetric"] {
            border: 1px solid #fed7aa; border-radius: 12px; background: #fff;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: #7c2d12; }
        [data-testid="stMetricValue"] { color: #172033; }
        button[role="tab"][aria-selected="true"] {
            color: var(--dashboard-orange-dark); border-bottom-color: var(--dashboard-orange);
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Configure Streamlit page
st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch quotes and fundamentals from Yahoo Finance through yfinance."""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, auto_adjust=False)
            quote_history = data

            # Short selections such as 1d may contain no prior session for the
            # previous-close fallback, so fetch a small real quote history.
            if data is None or len(data) < 2:
                try:
                    quote_history = stock.history(period="5d", interval="1d", auto_adjust=False)
                except Exception:
                    quote_history = data

            try:
                info = stock.info or {}
            except Exception:
                info = {}

            fast_info = {}
            try:
                raw_fast_info = stock.fast_info
                for key in ("last_price", "previous_close", "open", "day_high", "day_low", "year_high", "year_low"):
                    value = _mapping_number(raw_fast_info, key)
                    if value is not None:
                        fast_info[key] = value
            except Exception:
                pass

            try:
                earnings_calendar = stock.calendar or {}
            except Exception:
                earnings_calendar = {}

            try:
                major_holders = stock.major_holders
            except Exception:
                major_holders = None

            try:
                institutional_holders = stock.institutional_holders
            except Exception:
                institutional_holders = None

            try:
                mutualfund_holders = stock.mutualfund_holders
            except Exception:
                mutualfund_holders = None

            year_high = _mapping_number(fast_info, "year_high")
            year_low = _mapping_number(fast_info, "year_low")
            if year_high is None:
                year_high = _mapping_number(info, "fiftyTwoWeekHigh")
            if year_low is None:
                year_low = _mapping_number(info, "fiftyTwoWeekLow")
            if year_high is None or year_low is None:
                try:
                    year_data = stock.history(period="1y", interval="1d", auto_adjust=False)
                    if year_data is not None and not year_data.empty:
                        if year_high is None and "High" in year_data:
                            year_high = _mapping_number({"value": year_data["High"].max()}, "value")
                        if year_low is None and "Low" in year_data:
                            year_low = _mapping_number({"value": year_data["Low"].min()}, "value")
                except Exception:
                    pass

            return (
                data, info, major_holders, institutional_holders, mutualfund_holders,
                fast_info, earnings_calendar, quote_history, year_high, year_low,
            )
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, {}, None, None, None, {}, {}, None, None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate moving averages and volume metrics from historical quotes."""
        df = data.copy()
        
        # Exponential Moving Averages
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price-based indicators
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        return df
    
    def prepare_ml_features(self, data):
        """Prepare features for machine learning"""
        df = data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d'] = df['Close'].pct_change(10)
        
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        
        for window in [5, 10, 20, 50]:
            df[f'Close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'High_mean_{window}'] = df['High'].rolling(window).mean()
            df[f'Low_mean_{window}'] = df['Low'].rolling(window).mean()
        
        df['Price_vs_EMA20'] = (df['Close'] - df['EMA_20']) / df['EMA_20'] * 100
        df['Price_vs_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50'] * 100
        
        df['Price_volatility_10d'] = df['Returns'].rolling(10).std()
        df['Price_volatility_20d'] = df['Returns'].rolling(20).std()
        
        return df
    
    def train_prediction_model(self, data):
        """Train ML model for price prediction"""
        df = self.prepare_ml_features(data)
        df = df.dropna()
        
        if len(df) < 5:
            return None
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                       'Returns', 'Returns_5d', 'Returns_10d']
        feature_cols = [col for col in df.columns if not any(exc in col for exc in exclude_cols)]
        feature_cols = [col for col in feature_cols if 'lag' in col or 'mean' in col or 
                       'std' in col or col in ['Price_vs_EMA20', 'Price_vs_EMA50', 
                                              'Price_volatility_10d', 'Price_volatility_20d']]
        
        if len(feature_cols) < 2:
            return None
        
        X = df[feature_cols].ffill().bfill()
        y = df['Close'].shift(-1)
        
        X = X[:-1]
        y = y[:-1]
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 3:
            return None
        
        test_size_val = 0.2 if len(X) > 10 else 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'last_features': X.iloc[-1:],
            'feature_cols': feature_cols
        }
    
    def predict_next_price(self, model_info):
        """Predict next trading day price"""
        if model_info is None:
            return None
        last_features_scaled = self.scaler.transform(model_info['last_features'])
        prediction = self.model.predict(last_features_scaled)[0]
        return prediction

def create_performance_metrics(data, symbol):
    """Create performance metrics visualization"""
    if len(data) < 2:
        st.info("Insufficient data points for cumulative returns graph under this timeframe.")
        return
        
    data['Daily_Returns'] = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    
    total_return = data['Cumulative_Returns'].iloc[-1] * 100 if not pd.isna(data['Cumulative_Returns'].iloc[-1]) else 0
    volatility = data['Daily_Returns'].std() * np.sqrt(252) * 100 if len(data) > 5 else 0
    sharpe_ratio = (data['Daily_Returns'].mean() * 252) / (data['Daily_Returns'].std() * np.sqrt(252)) if len(data) > 5 and data['Daily_Returns'].std() > 0 else 0
    max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100 if len(data) > 1 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Volatility (Ann.)", f"{volatility:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Cumulative_Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#f97316', width=2)
        )
    )
    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, width="stretch")

# Streamlit App
def main():
    apply_orange_white_theme()
    st.session_state.setdefault("favorites", [])
    st.session_state.setdefault("stock_selector", "AAPL")

    st.title("🚀 Professional AI Stock Market Dashboard")
    st.markdown("*Yahoo Finance market data, company fundamentals, and machine learning analysis*")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    popular_stocks = {
        'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL', 
        'Amazon': 'AMZN', 'Tesla': 'TSLA', 'NVIDIA': 'NVDA',
        'Meta': 'META', 'Netflix': 'NFLX', 'AMD': 'AMD', 'Intel': 'INTC'
    }
    
    ticker_names = {ticker: name for name, ticker in popular_stocks.items()}
    stock_options = list(dict.fromkeys(list(popular_stocks.values()) + list(st.session_state["favorites"])))
    stock_options.append("Custom")
    if st.session_state["stock_selector"] not in stock_options:
        st.session_state["stock_selector"] = "AAPL"

    selected_stock = st.sidebar.selectbox(
        "🏢 Select Stock:",
        options=stock_options,
        format_func=lambda ticker: "Enter a custom symbol" if ticker == "Custom" else f"{ticker_names[ticker]} ({ticker})" if ticker in ticker_names else ticker,
        key="stock_selector",
    )

    if selected_stock == 'Custom':
        symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL", max_chars=10).strip().upper()
    else:
        symbol = selected_stock
    
    period = st.sidebar.selectbox(
        "📅 Analysis Period:",
        options=['1d', '1wk', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=2
    )
    
    st.sidebar.markdown("---")
    
    show_prediction = st.sidebar.checkbox("🔮 ML Price Prediction", value=True)
    show_performance = st.sidebar.checkbox("📊 Performance Metrics", value=True)
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    if symbol:
        is_favorite = symbol in st.session_state["favorites"]
        st.button(
            "★ Remove from Favorites" if is_favorite else "☆ Add to Favorites",
            type="primary",
            key=f"favorite-toggle-{symbol}",
            on_click=_toggle_favorite,
            args=(symbol,),
        )

    st.subheader("⭐ Favorites")
    favorites = list(st.session_state["favorites"])
    if favorites:
        for offset in range(0, len(favorites), 5):
            favorite_row = favorites[offset:offset + 5]
            favorite_columns = st.columns(len(favorite_row))
            for column, favorite in zip(favorite_columns, favorite_row):
                with column:
                    label = f"✓ {favorite} · Selected" if favorite == symbol else f"★ {favorite}"
                    st.button(
                        label,
                        key=f"favorite-select-{favorite}",
                        on_click=_set_selected_stock,
                        args=(favorite,),
                    )
    else:
        st.caption("Add a stock to Favorites to keep it one click away during this session.")

    analyzer = StockAnalyzer()
    
    with st.spinner(f"📡 Fetching Yahoo Finance data for {symbol}..."):
        (
            data, info, major_holders, institutional_holders, mutualfund_holders,
            fast_info, earnings_calendar, quote_history, year_high, year_low,
        ) = analyzer.fetch_stock_data(symbol, period)
    
    if data is None or data.empty:
        st.error(f"❌ Could not fetch data for {symbol}. Please verify the symbol and try again.")
        st.info("💡 Try popular symbols like AAPL, MSFT, GOOGL, TSLA, etc.")
        return
    
    with st.spinner("⚙️ Calculating metrics..."):
        data = analyzer.calculate_technical_indicators(data)
    
    st.markdown("---")
    
    if quote_history is None or quote_history.empty:
        quote_history = data
    snapshot = market_snapshot_values(fast_info, info, quote_history)
    latest_price = snapshot.get("last_price")
    previous_close = snapshot.get("previous_close")
    open_price = snapshot.get("open")
    day_high = snapshot.get("high")
    day_low = snapshot.get("low")
    price_change = latest_price - previous_close if latest_price is not None and previous_close is not None else None
    price_change_pct = (price_change / previous_close) * 100 if price_change is not None and previous_close else None
    volume = _mapping_number({"value": data["Volume"].iloc[-1] if "Volume" in data else None}, "value")
    avg_volume = data["Volume"].rolling(20).mean().iloc[-1] if len(data) >= 20 else (data["Volume"].mean() if "Volume" in data else None)
    volume_change = ((volume - avg_volume) / avg_volume) * 100 if volume is not None and avg_volume and avg_volume > 0 else None
    market_cap = _mapping_number(info, "marketCap")
    if market_cap is None:
        market_cap = _mapping_number(fast_info, "market_cap")
    ema_20 = _mapping_number({"value": data["EMA_20"].iloc[-1]}, "value")

    st.subheader(f"📌 {symbol} · Market Snapshot")
    latest_session = snapshot.get("session_date", "N/A")
    st.caption(f"Latest available trading session: {latest_session}. Quotes are supplied by Yahoo Finance through yfinance and may be delayed.")
    market_columns = st.columns(4)
    with market_columns[0]:
        st.metric("💰 Last Price", _format_price(latest_price), delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)" if price_change is not None and price_change_pct is not None else None)
    with market_columns[1]:
        st.metric("Previous Close", _format_price(previous_close))
    with market_columns[2]:
        st.metric("Open", _format_price(open_price))
    with market_columns[3]:
        st.metric("Volume", f"{volume:,.0f}" if volume is not None and volume > 0 else "N/A", delta=f"{volume_change:+.1f}% vs avg" if volume_change is not None else None)

    price_columns = st.columns(4)
    with price_columns[0]:
        st.metric("High", _format_price(day_high))
    with price_columns[1]:
        st.metric("Low", _format_price(day_low))
    with price_columns[2]:
        if market_cap and market_cap > 0:
            cap_display = f"${market_cap/1e12:,.2f}T" if market_cap >= 1e12 else f"${market_cap/1e9:,.1f}B" if market_cap >= 1e9 else f"${market_cap/1e6:,.0f}M"
        else:
            cap_display = "N/A"
        st.metric("Market Cap", cap_display)
    with price_columns[3]:
        st.metric("EMA 20", _format_price(ema_20))

    trailing_pe = _mapping_number(info, "trailingPE")
    trailing_eps = _mapping_number(info, "trailingEps")
    earnings_date = next_earnings_date(earnings_calendar, info)
    st.subheader("📚 Company Fundamentals")
    fundamental_columns = st.columns(5)
    with fundamental_columns[0]:
        st.metric("P/E Ratio (trailing)", f"{trailing_pe:.2f}" if trailing_pe is not None else "N/A")
    with fundamental_columns[1]:
        st.metric("EPS (trailing)", f"${trailing_eps:,.2f}" if trailing_eps is not None else "N/A")
    with fundamental_columns[2]:
        st.metric("Next Earnings Date", earnings_date or "N/A")
    with fundamental_columns[3]:
        st.metric("52-Week High", _format_price(year_high))
    with fundamental_columns[4]:
        st.metric("52-Week Low", _format_price(year_low))
    st.caption("Fundamentals and calendar dates come from Yahoo Finance. Earnings dates can be estimates and may change; unsupported fields stay N/A rather than being inferred.")
    
    st.markdown("---")
    
    if show_performance:
        st.subheader("📊 Performance Analysis")
        create_performance_metrics(data, symbol)
    
    if show_prediction:
        st.subheader("🔮 Machine Learning Price Prediction")
        col_pred1, col_pred2 = st.columns([1, 1])
        
        with col_pred1:
            with st.spinner("🤖 Training AI prediction model..."):
                model_info = analyzer.train_prediction_model(data)
            
            if model_info:
                prediction = analyzer.predict_next_price(model_info)
                current_price = data['Close'].iloc[-1]
                predicted_change = ((prediction - current_price) / current_price) * 100
                
                st.success("✅ Model trained successfully!")
                
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    st.metric(
                        label="🎯 Next Day Prediction",
                        value=f"${prediction:,.2f}",
                        delta=f"{predicted_change:+.2f}%"
                    )
                
                with sub_col2:
                    st.metric(
                        label="🎲 Test R² Score",
                        value=f"{model_info['test_score']:.1%}",
                    )
                
                st.info(f"📈 **Training R²:** {model_info['train_score']:.1%} | **Test R²:** {model_info['test_score']:.1%}. R² measures model fit and is not a probability of a correct prediction.")
            else:
                st.warning("⚠️ Insufficient data points for reliable ML prediction under this timeframe.")
        
        with col_pred2:
            if model_info:
                importance_df = pd.DataFrame(
                    list(model_info['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="🔍 Top 10 Most Important Features",
                    color_discrete_sequence=["#f97316"],
                    template='plotly_white'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, width="stretch")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Company Info & Holders", "📊 Volume Comparison & Raw Data", "🔧 Technical Metrics"])
    
    with tab1:
        if info or institutional_holders is not None or mutualfund_holders is not None:
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("### 🏢 Company Details")
                emp_count = info.get('fullTimeEmployees')
                formatted_emp = f"{emp_count:,}" if emp_count and isinstance(emp_count, (int, float)) else 'N/A'
                
                company_info = {
                    "Company Name": info.get('longName', 'N/A'),
                    "Sector": info.get('sector', 'N/A'),
                    "Industry": info.get('industry', 'N/A'),
                    "Country": info.get('country', 'N/A'),
                    "Website": info.get('website', 'N/A'),
                    "Employees": formatted_emp
                }
                for key, value in company_info.items():
                    st.write(f"**{key}:** {value}")
            
            with c2:
                st.write("### 🏛️ Major Holders Breakdown")
                insider_pct = _mapping_number(info, 'heldPercentInsiders')
                inst_pct = _mapping_number(info, 'heldPercentInstitutions')
                reported_holder_count = count_reported_institutional_holders(institutional_holders)
                
                breakdown_data = {
                    "Value": [
                        f"{insider_pct*100:.2f}%" if insider_pct is not None else "N/A",
                        f"{inst_pct*100:.2f}%" if inst_pct is not None else "N/A",
                        f"{reported_holder_count:,}" if reported_holder_count is not None else "N/A",
                    ],
                    "Breakdown Metric": [
                        "% of Shares Held by All Insiders",
                        "% of Shares Held by Institutions",
                        "Number of Institutions Holding Shares"
                    ]
                }
                st.dataframe(pd.DataFrame(breakdown_data), width="stretch")
                st.caption("The institution count is the number of named entries in Yahoo Finance's available holder list; it may not include every institution holding shares.")
            
            st.markdown("---")
            st.write("### 🏛️ Top Institutional Holders & Session Comparison")
            
            if institutional_holders is not None and not institutional_holders.empty:
                formatted_inst = institutional_holders.copy()
                if 'Shares' in formatted_inst.columns:
                    formatted_inst['Shares'] = formatted_inst['Shares'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
                if 'Value' in formatted_inst.columns:
                    formatted_inst['Value'] = formatted_inst['Value'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else x)
                if 'pctHeld' in formatted_inst.columns:
                    formatted_inst['pctHeld'] = formatted_inst['pctHeld'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else x)
                if 'pctChange' in formatted_inst.columns:
                    formatted_inst['pctChange'] = formatted_inst['pctChange'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else x)

                if 'hist_holders' not in st.session_state:
                    st.session_state['hist_holders'] = {}
                
                current_raw_dict = institutional_holders.set_index('Holder')['Shares'].to_dict() if 'Holder' in institutional_holders.columns and 'Shares' in institutional_holders.columns else {}
                
                if symbol in st.session_state['hist_holders']:
                    old_dict = st.session_state['hist_holders'][symbol]
                    comparison_rows = []
                    for holder, shares in current_raw_dict.items():
                        old_shares = old_dict.get(holder, 0)
                        diff = shares - old_shares
                        comparison_rows.append({
                            "Holder": holder,
                            "Current Shares": f"{shares:,.0f}",
                            "Previous Tracked Shares": f"{old_shares:,.0f}",
                            "Change (+/-)": f"{diff:+,.0f}"
                        })
                    comp_df = pd.DataFrame(comparison_rows)
                    st.write("📊 **Comparison with Previously Cached Session Data:**")
                    st.dataframe(comp_df, width="stretch")
                else:
                    st.info("💡 First snapshot for this stock cached in session state. Refresh or update to see comparative deltas.")
                
                st.session_state['hist_holders'][symbol] = current_raw_dict
                
                st.write("📋 **Yahoo Finance Reported Institutional Holder Data:**")
                st.dataframe(formatted_inst, width="stretch")
            else:
                st.warning("Detailed institutional holders data currently unavailable via API for this ticker.")
            
            st.markdown("---")
            st.write("### 📈 Top Mutual Fund Holders")
            if mutualfund_holders is not None and not mutualfund_holders.empty:
                formatted_mf = mutualfund_holders.copy()
                if 'Shares' in formatted_mf.columns:
                    formatted_mf['Shares'] = formatted_mf['Shares'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else x)
                if 'Value' in formatted_mf.columns:
                    formatted_mf['Value'] = formatted_mf['Value'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else x)
                if 'pctHeld' in formatted_mf.columns:
                    formatted_mf['pctHeld'] = formatted_mf['pctHeld'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else x)
                
                st.dataframe(formatted_mf, width="stretch")
            else:
                st.warning("Mutual fund holders data currently unavailable via API for this ticker.")
        else:
            st.warning("Company information not available")
    
    with tab2:
        st.write("### 📊 Multi-Horizon Volume Comparison (1 Day, 1 Week, 1 Month, 3 Months)")
        
        vol_comparison_data = {}
        horizon_labels = {'1d': '1 Day', '1wk': '1 Week', '1mo': '1 Month', '3mo': '3 Months'}
        
        for h_key, h_label in horizon_labels.items():
            try:
                temp_hist = yf.Ticker(symbol).history(period=h_key, auto_adjust=False)
                if not temp_hist.empty:
                    average_volume = temp_hist['Volume'].mean()
                    vol_comparison_data[h_label] = float(average_volume) if pd.notna(average_volume) else np.nan
                else:
                    vol_comparison_data[h_label] = np.nan
            except Exception:
                vol_comparison_data[h_label] = np.nan
                
        vol_comp_df = pd.DataFrame(list(vol_comparison_data.items()), columns=['Time Horizon', 'Average Volume'])
        
        if vol_comp_df['Average Volume'].notna().any():
            fig_multi_vol = px.bar(
                vol_comp_df,
                x='Time Horizon',
                y='Average Volume',
                text_auto=',.2s',
                title=f"{symbol} - Average Volume Across Time Horizons",
                template='plotly_white',
                color='Average Volume',
                color_continuous_scale='Oranges'
            )
            fig_multi_vol.update_layout(height=400)
            st.plotly_chart(fig_multi_vol, width="stretch")
        else:
            st.info("Yahoo Finance did not return volume history for these comparison periods.")
        
        st.markdown("---")
        st.write("### 📊 Recent Price & Volume Table")
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Volume_SMA']].tail(20).copy()
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        
        for col in ['Open', 'High', 'Low', 'Close']:
            display_data[col] = display_data[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
        display_data['Volume_SMA'] = display_data['Volume_SMA'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
        
        st.dataframe(display_data, width="stretch")
        
        csv = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20).to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_stock_data.csv',
            mime='text/csv'
        )
    
    with tab3:
        st.write("### 🔧 Technical Indicators (EMA 20, EMA 50, EMA 200)")
        tech_columns = ['Close', 'EMA_20', 'EMA_50', 'EMA_200']
        available_columns = [col for col in tech_columns if col in data.columns]
        
        if available_columns:
            tech_data = data[available_columns].tail(10).copy()
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            
            for col in available_columns:
                tech_data[col] = tech_data[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
                
            st.dataframe(tech_data, width="stretch")
        else:
            st.warning("Technical indicators not available")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>AI Stock Dashboard</strong> - Professional financial analytics with machine learning</p>
            <p><em>⚠️ This is for educational purposes only. Not financial advice.</em></p>
            <p>Built with ❤️ by <a href='' target='_blank'>Kushal Dhakal</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
