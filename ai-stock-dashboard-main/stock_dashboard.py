import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
<style>
:root {
    --orange: #ff7900;
    --orange-dark: #e76600;
    --orange-soft: #fff4e8;
    --border: #e7e7e7;
    --text: #171717;
    --muted: #6b7280;
    --green: #159447;
    --red: #d33b3b;
}

.stApp {
    background: #ffffff;
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #eeeeee;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, #fff7ef 0%, #ffffff 72%);
    border: 1px solid #f0dcc8;
    border-radius: 22px;
    padding: 26px;
    margin: 18px 0 18px 0;
    box-shadow: 0 8px 30px rgba(0,0,0,.035);
}

.hero-symbol {
    font-size: 2.2rem;
    font-weight: 850;
    letter-spacing: -.045em;
}

.hero-name {
    color: var(--muted);
    font-size: .95rem;
    margin-top: 2px;
}

.hero-price {
    font-size: 2.8rem;
    font-weight: 850;
    letter-spacing: -.05em;
    margin-top: 13px;
}

.hero-up {
    color: var(--green);
    font-weight: 750;
}

.hero-down {
    color: var(--red);
    font-weight: 750;
}

.hero-flat {
    color: var(--muted);
    font-weight: 750;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 850;
    margin: 28px 0 13px 0;
    letter-spacing: -.02em;
}

.metric-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 17px;
    min-height: 102px;
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: #ffbc78;
    box-shadow: 0 10px 25px rgba(255,121,0,.10);
}

.metric-label {
    color: #737373;
    font-size: .78rem;
    font-weight: 650;
    margin-bottom: 8px;
}

.metric-value {
    color: #111111;
    font-size: 1.24rem;
    font-weight: 820;
    word-break: break-word;
}

.notice {
    background: #fff7ef;
    border: 1px solid #ffd2a6;
    border-radius: 14px;
    padding: 15px 17px;
    color: #6d3d00;
}

.favorite-box {
    background: #fffaf5;
    border: 1px solid #f4d8bc;
    border-radius: 15px;
    padding: 14px;
}

.small-muted {
    color: #8a8a8a;
    font-size: .78rem;
}

div.stButton > button {
    border-radius: 11px;
    transition: transform .15s ease, box-shadow .15s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(255,121,0,.12);
}

a {
    color: var(--orange-dark);
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "AAPL"


# ============================================================
# FORMAT HELPERS
# ============================================================

def clean_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def first_valid(*values):
    for value in values:
        if value is not None:
            return value
    return None


def clean_text(value):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    value = str(value).strip()
    if not value or value.lower() in {"none", "nan", "n/a", "null", "nat"}:
        return "—"
    return value


def money(value, decimals=2):
    value = clean_number(value)
    return "—" if value is None else f"${value:,.{decimals}f}"


def plain_number(value, decimals=2):
    value = clean_number(value)
    return "—" if value is None else f"{value:,.{decimals}f}"


def integer(value):
    value = clean_number(value)
    return "—" if value is None else f"{int(value):,}"


def large_money(value):
    value = clean_number(value)
    if value is None:
        return "—"

    absolute = abs(value)
    if absolute >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"${value:,.2f}"


def percent(value):
    value = clean_number(value)
    return "—" if value is None else f"{value:+.2f}%"


def format_date(value):
    if value is None:
        return "—"
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return "—"
        return parsed.strftime("%b %d, %Y")
    except Exception:
        return "—"


def info_value(info, *keys):
    if not isinstance(info, dict):
        return None

    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        return value

    return None


def metric_card(label, value):
    html = (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# DATA FETCHING
# ============================================================

def get_history(ticker, period="1y"):
    try:
        frame = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

        if frame is None or frame.empty:
            return None

        required = {"Open", "High", "Low", "Close", "Volume"}

        if not required.issubset(frame.columns):
            return None

        frame = frame.dropna(subset=["Close"]).copy()

        if frame.empty:
            return None

        return frame

    except Exception:
        return None


def get_fast_info(ticker):
    try:
        return dict(ticker.fast_info)
    except Exception:
        return {}


def get_info(ticker):
    try:
        result = ticker.get_info()
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def get_earnings_date(ticker):
    dates = []

    try:
        frame = ticker.get_earnings_dates(limit=12)

        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for value in frame.index:
                parsed = pd.to_datetime(value, errors="coerce")
                if not pd.isna(parsed):
                    dates.append(parsed.to_pydatetime())

    except Exception:
        pass

    if dates:
        now = datetime.now()
        future = [item for item in dates if item >= now]
        return min(future) if future else max(dates)

    try:
        calendar = ticker.get_calendar()

        if isinstance(calendar, dict):
            value = calendar.get("Earnings Date")

            if isinstance(value, (list, tuple)):
                parsed_dates = []

                for item in value:
                    parsed = pd.to_datetime(item, errors="coerce")
                    if not pd.isna(parsed):
                        parsed_dates.append(parsed)

                if parsed_dates:
                    return min(parsed_dates)

            parsed = pd.to_datetime(value, errors="coerce")

            if not pd.isna(parsed):
                return parsed

    except Exception:
        pass

    return None


def parse_holder_value(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        if "%" in value:
            return value

    numeric = clean_number(value)

    if numeric is None:
        return clean_text(value)

    if 0 <= numeric <= 1:
        return f"{numeric * 100:.2f}%"

    return f"{numeric:.2f}%"


def parse_major_holders(frame):
    result = {}

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return result

    for _, row in frame.iterrows():
        values = row.tolist()

        if len(values) < 2:
            continue

        label = None
        value = None

        if isinstance(values[0], str):
            label = values[0]
            value = values[-1]
        elif isinstance(values[-1], str):
            label = values[-1]
            value = values[0]

        if not label:
            continue

        key = label.lower()

        if "insider" in key:
            result["insider"] = parse_holder_value(value)
        elif "institution" in key:
            result["institution"] = parse_holder_value(value)
        elif "float" in key:
            result["float"] = parse_holder_value(value)

    return result


def holder_data_from_info(info):
    """
    Yahoo's quote/info payload contains more stable ownership fields
    than parsing the presentation-oriented major_holders DataFrame.
    These are used first, with get_major_holders() as fallback.
    """
    result = {}

    insider = info_value(
        info,
        "heldPercentInsiders",
        "heldPercentInsidersRaw",
    )

    institution = info_value(
        info,
        "heldPercentInstitutions",
        "heldPercentInstitutionsRaw",
    )

    float_shares = info_value(
        info,
        "floatShares",
    )

    if insider is not None:
        result["insider"] = parse_holder_value(insider)

    if institution is not None:
        result["institution"] = parse_holder_value(institution)

    if float_shares is not None:
        result["float_shares"] = float_shares

    return result


# ============================================================
# TECHNICAL ANALYSIS
# IMPORTANT: ATR HAS BEEN REMOVED COMPLETELY.
# ============================================================

def calculate_technical_indicators(data):
    df = data.copy()

    # Moving averages
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # Exponential moving averages
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()

    df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
    df["BB_lower"] = df["BB_middle"] - (bb_std * 2)

    # Volume
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_SMA"].replace(0, np.nan)

    # Price metrics
    df["High_Low_Pct"] = (
        (df["High"] - df["Low"]) / df["Close"] * 100
    )

    df["Price_Change"] = df["Close"] - df["Open"]

    df["Price_Change_Pct"] = (
        (df["Close"] - df["Open"]) / df["Open"] * 100
    )

    # Stochastic oscillator
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()

    range_14 = (high_14 - low_14).replace(0, np.nan)

    df["Stoch_K"] = (
        100 * (df["Close"] - low_14) / range_14
    )

    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    return df


# ============================================================
# ML
# ============================================================

class StockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    def prepare_ml_features(self, data):
        df = data.copy()

        df["Returns"] = df["Close"].pct_change()
        df["Returns_5d"] = df["Close"].pct_change(5)
        df["Returns_10d"] = df["Close"].pct_change(10)

        for lag in [1, 2, 3, 5, 10]:
            df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
            df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)
            df[f"Returns_lag_{lag}"] = df["Returns"].shift(lag)

        for window in [5, 10, 20, 50]:
            df[f"Close_mean_{window}"] = df["Close"].rolling(window).mean()
            df[f"Close_std_{window}"] = df["Close"].rolling(window).std()
            df[f"Volume_mean_{window}"] = df["Volume"].rolling(window).mean()
            df[f"High_mean_{window}"] = df["High"].rolling(window).mean()
            df[f"Low_mean_{window}"] = df["Low"].rolling(window).mean()

        df["Price_vs_SMA20"] = (
            (df["Close"] - df["SMA_20"]) / df["SMA_20"] * 100
        )

        df["Price_vs_SMA50"] = (
            (df["Close"] - df["SMA_50"]) / df["SMA_50"] * 100
        )

        df["Price_volatility_10d"] = df["Returns"].rolling(10).std()
        df["Price_volatility_20d"] = df["Returns"].rolling(20).std()

        return df

    def train_prediction_model(self, data):
        df = self.prepare_ml_features(data).dropna()

        if len(df) < 100:
            return None

        excluded = {
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock Splits",
            "Returns",
            "Returns_5d",
            "Returns_10d",
        }

        feature_cols = []

        for column in df.columns:
            if column in excluded:
                continue

            if (
                "lag" in column
                or "mean" in column
                or "std" in column
                or column in [
                    "RSI",
                    "MACD",
                    "Price_vs_SMA20",
                    "Price_vs_SMA50",
                    "Price_volatility_10d",
                    "Price_volatility_20d",
                ]
            ):
                feature_cols.append(column)

        if len(feature_cols) < 5:
            return None

        X = df[feature_cols].replace(
            [np.inf, -np.inf],
            np.nan,
        ).ffill().bfill()

        y = df["Close"].shift(-1)

        X = X.iloc[:-1]
        y = y.iloc[:-1]

        mask = ~(X.isna().any(axis=1) | y.isna())

        X = X.loc[mask]
        y = y.loc[mask]

        if len(X) < 50:
            return None

        split = int(len(X) * 0.8)

        if split < 30 or len(X) - split < 10:
            return None

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(
            X_train_scaled,
            y_train,
        )

        train_score = self.model.score(
            X_train_scaled,
            y_train,
        )

        test_score = self.model.score(
            X_test_scaled,
            y_test,
        )

        return {
            "train_score": train_score,
            "test_score": test_score,
            "feature_importance": dict(
                zip(
                    feature_cols,
                    self.model.feature_importances_,
                )
            ),
            "last_features": X.iloc[-1:],
            "feature_cols": feature_cols,
        }

    def predict_next_price(self, model_info):
        if model_info is None:
            return None

        last_features_scaled = self.scaler.transform(
            model_info["last_features"]
        )

        return float(
            self.model.predict(last_features_scaled)[0]
        )

    def generate_market_analysis(
        self,
        data,
        info,
        symbol,
    ):
        latest = data.iloc[-1]

        if len(data) >= 2:
            previous = data.iloc[-2]
        else:
            previous = latest

        price_change = (
            latest["Close"] - previous["Close"]
        )

        if previous["Close"] != 0:
            price_change_pct = (
                price_change / previous["Close"] * 100
            )
        else:
            price_change_pct = 0

        rsi = clean_number(latest.get("RSI"))
        sma_20 = clean_number(latest.get("SMA_20"))
        sma_50 = clean_number(latest.get("SMA_50"))
        bb_upper = clean_number(latest.get("BB_upper"))
        bb_lower = clean_number(latest.get("BB_lower"))
        macd = clean_number(latest.get("MACD"))
        macd_signal = clean_number(
            latest.get("MACD_signal")
        )

        avg_volume = (
            data["Volume"].rolling(20).mean().iloc[-1]
        )

        volume_ratio = (
            latest["Volume"] / avg_volume
            if avg_volume and avg_volume > 0
            else 1
        )

        analysis = []

        if price_change_pct > 3:
            analysis.append(
                f"🚀 {symbol} shows strong bullish momentum "
                f"with a {price_change_pct:.2f}% move."
            )
        elif price_change_pct > 1:
            analysis.append(
                f"🟢 {symbol} is showing solid upward movement "
                f"({price_change_pct:+.2f}%)."
            )
        elif price_change_pct > 0:
            analysis.append(
                f"🟡 {symbol} is showing a modest gain "
                f"({price_change_pct:+.2f}%)."
            )
        elif price_change_pct > -1:
            analysis.append(
                f"🟡 {symbol} is showing a modest decline "
                f"({price_change_pct:.2f}%)."
            )
        elif price_change_pct > -3:
            analysis.append(
                f"🔴 {symbol} is under moderate selling pressure "
                f"({price_change_pct:.2f}%)."
            )
        else:
            analysis.append(
                f"🔻 {symbol} is experiencing significant "
                f"downward pressure ({price_change_pct:.2f}%)."
            )

        if rsi is not None:
            if rsi > 80:
                analysis.append(
                    f"🚨 RSI at {rsi:.1f} indicates very strong "
                    "overbought conditions."
                )
            elif rsi > 70:
                analysis.append(
                    f"⚠️ RSI at {rsi:.1f} is in overbought territory."
                )
            elif rsi < 20:
                analysis.append(
                    f"🛒 RSI at {rsi:.1f} indicates very strong "
                    "oversold conditions."
                )
            elif rsi < 30:
                analysis.append(
                    f"💡 RSI at {rsi:.1f} is in oversold territory."
                )
            elif 40 <= rsi <= 60:
                analysis.append(
                    f"⚖️ RSI at {rsi:.1f} indicates relatively balanced momentum."
                )
            else:
                bias = "bullish" if rsi > 50 else "bearish"
                analysis.append(
                    f"📊 RSI at {rsi:.1f} shows a {bias} bias."
                )

        if (
            sma_20 is not None
            and sma_50 is not None
        ):
            if latest["Close"] > sma_20 > sma_50:
                analysis.append(
                    "📈 Price is above both the 20-day and 50-day "
                    "moving averages with bullish alignment."
                )
            elif latest["Close"] < sma_20 < sma_50:
                analysis.append(
                    "📉 Price is below both the 20-day and 50-day "
                    "moving averages with bearish alignment."
                )
            else:
                analysis.append(
                    "➡️ Moving averages are giving mixed signals."
                )

        if (
            bb_upper is not None
            and bb_lower is not None
        ):
            if latest["Close"] > bb_upper:
                analysis.append(
                    "📊 Price is above the upper Bollinger Band."
                )
            elif latest["Close"] < bb_lower:
                analysis.append(
                    "📊 Price is below the lower Bollinger Band."
                )

        if (
            macd is not None
            and macd_signal is not None
        ):
            if macd > macd_signal and macd > 0:
                analysis.append(
                    "⚡ MACD shows positive bullish momentum."
                )
            elif macd < macd_signal and macd < 0:
                analysis.append(
                    "⚡ MACD shows negative bearish momentum."
                )
            elif macd > macd_signal:
                analysis.append(
                    "⚡ MACD is above its signal line."
                )
            else:
                analysis.append(
                    "⚡ MACD is below its signal line."
                )

        if volume_ratio > 2:
            analysis.append(
                "🔥 Volume is more than twice its recent average."
            )
        elif volume_ratio > 1.5:
            analysis.append(
                "📊 Above-average volume is supporting the move."
            )
        elif volume_ratio < 0.5:
            analysis.append(
                "📊 Volume is substantially below its recent average."
            )
        else:
            analysis.append(
                "📊 Volume is close to its recent average."
            )

        market_cap = clean_number(
            info_value(info, "marketCap")
        )

        if market_cap:
            if market_cap > 200e9:
                analysis.append(
                    "🏢 This is a large-cap company."
                )
            elif market_cap > 10e9:
                analysis.append(
                    "🏢 This is a mid/large-cap company."
                )
            else:
                analysis.append(
                    "🏢 This is a smaller-cap company with potentially higher volatility."
                )

        return analysis


# ============================================================
# CHARTS
# ============================================================

def create_advanced_chart(data, symbol):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"{symbol} Price Action & Moving Averages",
            "Volume",
            "MACD",
            "RSI & Stochastic",
        ),
        row_heights=[0.50, 0.15, 0.20, 0.15],
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="#00cc88",
            decreasing_line_color="#ff4d4d",
        ),
        row=1,
        col=1,
    )

    moving_averages = [
        ("SMA_20", "SMA 20", "#ff9500"),
        ("SMA_50", "SMA 50", "#007aff"),
        ("SMA_200", "SMA 200", "#5856d6"),
    ]

    for column, name, color in moving_averages:
        if (
            column in data.columns
            and not data[column].isna().all()
        ):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    line=dict(
                        color=color,
                        width=1.5,
                    ),
                    name=name,
                ),
                row=1,
                col=1,
            )

    if all(
        column in data.columns
        for column in ["BB_upper", "BB_lower"]
    ):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_upper"],
                line=dict(
                    color="rgba(128,128,128,.5)",
                    width=1,
                ),
                name="BB Upper",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_lower"],
                line=dict(
                    color="rgba(128,128,128,.5)",
                    width=1,
                ),
                name="BB Lower",
                fill="tonexty",
                fillcolor="rgba(128,128,128,.08)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    volume_colors = [
        "#00cc88"
        if close >= open_
        else "#ff4d4d"
        for open_, close
        in zip(
            data["Open"],
            data["Close"],
        )
    ]

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            marker_color=volume_colors,
            name="Volume",
            opacity=.7,
        ),
        row=2,
        col=1,
    )

    if "Volume_SMA" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Volume_SMA"],
                line=dict(
                    color="white",
                    width=1,
                ),
                name="Volume SMA",
            ),
            row=2,
            col=1,
        )

    if all(
        column in data.columns
        for column in [
            "MACD",
            "MACD_signal",
            "MACD_histogram",
        ]
    ):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD"],
                line=dict(
                    color="#007aff",
                    width=2,
                ),
                name="MACD",
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD_signal"],
                line=dict(
                    color="#ff9500",
                    width=2,
                ),
                name="Signal",
            ),
            row=3,
            col=1,
        )

        histogram_colors = [
            "#00cc88"
            if value >= 0
            else "#ff4d4d"
            for value in data["MACD_histogram"]
        ]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["MACD_histogram"],
                marker_color=histogram_colors,
                name="Histogram",
                opacity=.6,
            ),
            row=3,
            col=1,
        )

    if "RSI" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["RSI"],
                line=dict(
                    color="#af52de",
                    width=2,
                ),
                name="RSI",
            ),
            row=4,
            col=1,
        )

        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            opacity=.7,
            row=4,
            col=1,
        )

        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            opacity=.7,
            row=4,
            col=1,
        )

        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="gray",
            opacity=.5,
            row=4,
            col=1,
        )

    if "Stoch_K" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Stoch_K"],
                line=dict(
                    color="#ffcc00",
                    width=1.5,
                ),
                name="Stoch %K",
            ),
            row=4,
            col=1,
        )

    if "Stoch_D" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Stoch_D"],
                line=dict(
                    color="#ff6600",
                    width=1.5,
                ),
                name="Stoch %D",
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title=f"{symbol} — Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template="plotly_dark",
        font=dict(size=10),
    )

    for row in range(1, 4):
        fig.update_xaxes(
            showticklabels=False,
            row=row,
            col=1,
        )

    return fig


def create_performance_metrics(data, symbol):
    frame = data.copy()

    frame["Daily_Returns"] = frame["Close"].pct_change()
    frame["Cumulative_Returns"] = (
        1 + frame["Daily_Returns"]
    ).cumprod() - 1

    total_return = (
        frame["Cumulative_Returns"].iloc[-1] * 100
    )

    daily_std = frame["Daily_Returns"].std()

    volatility = (
        daily_std * np.sqrt(252) * 100
        if pd.notna(daily_std)
        else np.nan
    )

    sharpe = (
        (frame["Daily_Returns"].mean() * 252)
        / (daily_std * np.sqrt(252))
        if daily_std and daily_std > 0
        else np.nan
    )

    max_drawdown = (
        (
            frame["Close"]
            / frame["Close"].cummax()
        ) - 1
    ).min() * 100

    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "Total Return",
            f"{total_return:.1f}%",
        )

    with cols[1]:
        st.metric(
            "Volatility (Ann.)",
            "—" if pd.isna(volatility)
            else f"{volatility:.1f}%",
        )

    with cols[2]:
        st.metric(
            "Sharpe Ratio",
            "—" if pd.isna(sharpe)
            else f"{sharpe:.2f}",
        )

    with cols[3]:
        st.metric(
            "Max Drawdown",
            f"{max_drawdown:.1f}%",
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["Cumulative_Returns"] * 100,
            mode="lines",
            name="Cumulative Returns",
            line=dict(
                color="#ff7900",
                width=2,
            ),
        )
    )

    fig.update_layout(
        title=f"{symbol} Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# STOCK DATA LOADER
# ============================================================

@st.cache_data(
    ttl=900,
    max_entries=100,
    show_spinner=False,
)
def load_stock(symbol, selected_period, cache_day):
    del cache_day

    symbol = symbol.strip().upper()

    ticker = yf.Ticker(symbol)

    info = get_info(ticker)
    fast = get_fast_info(ticker)

    history = get_history(
        ticker,
        selected_period,
    )

    errors = []

    if history is None:
        errors.append(
            "Historical price data was unavailable."
        )

        return {
            "usable": False,
            "errors": errors,
        }

    history = calculate_technical_indicators(
        history
    )

    current_price = first_valid(
        info_value(
            info,
            "currentPrice",
            "regularMarketPrice",
        ),
        fast.get("last_price"),
        clean_number(history["Close"].iloc[-1]),
    )

    previous_close = first_valid(
        info_value(
            info,
            "previousClose",
            "regularMarketPreviousClose",
        ),
        fast.get("previous_close"),
    )

    open_price = first_valid(
        info_value(
            info,
            "open",
            "regularMarketOpen",
        ),
        fast.get("open"),
    )

    day_high = first_valid(
        info_value(
            info,
            "dayHigh",
            "regularMarketDayHigh",
        ),
        fast.get("day_high"),
    )

    day_low = first_valid(
        info_value(
            info,
            "dayLow",
            "regularMarketDayLow",
        ),
        fast.get("day_low"),
    )

    latest = history.iloc[-1]

    open_price = first_valid(
        open_price,
        clean_number(latest["Open"]),
    )

    day_high = first_valid(
        day_high,
        clean_number(latest["High"]),
    )

    day_low = first_valid(
        day_low,
        clean_number(latest["Low"]),
    )

    if previous_close is None and len(history) >= 2:
        previous_close = clean_number(
            history["Close"].iloc[-2]
        )

    current_price = clean_number(
        current_price
    )

    previous_close = clean_number(
        previous_close
    )

    change = None
    change_pct = None

    if (
        current_price is not None
        and previous_close not in (None, 0)
    ):
        change = current_price - previous_close
        change_pct = (
            change / previous_close * 100
        )

    # Fundamental fields.
    trailing_pe = info_value(
        info,
        "trailingPE",
    )

    forward_pe = info_value(
        info,
        "forwardPE",
    )

    eps = first_valid(
        info_value(
            info,
            "trailingEps",
        ),
        info_value(
            info,
            "forwardEps",
        ),
    )

    week_high = info_value(
        info,
        "fiftyTwoWeekHigh",
    )

    week_low = info_value(
        info,
        "fiftyTwoWeekLow",
    )

    # If Yahoo quote fields are unavailable, derive 52-week range
    # from a one-year daily history.
    if week_high is None or week_low is None:
        year_history = (
            get_history(ticker, "1y")
        )

        if (
            year_history is not None
            and not year_history.empty
        ):
            if week_high is None:
                week_high = year_history["High"].max()

            if week_low is None:
                week_low = year_history["Low"].min()

    holder_info = holder_data_from_info(
        info
    )

    try:
        major_holders = (
            ticker.get_major_holders()
        )

        parsed_major = parse_major_holders(
            major_holders
        )

        for key, value in parsed_major.items():
            holder_info.setdefault(
                key,
                value,
            )

    except Exception as exc:
        errors.append(
            "Major-holder data unavailable: "
            f"{type(exc).__name__}"
        )

    try:
        institutional = (
            ticker.get_institutional_holders()
        )
    except Exception as exc:
        institutional = None
        errors.append(
            "Institutional-holder data unavailable: "
            f"{type(exc).__name__}"
        )

    # Stable Yahoo info fields are used for ownership.
    # floatShares is a share count, not a percentage.
    float_shares = holder_info.get(
        "float_shares"
    )

    return {
        "usable": True,
        "symbol": symbol,
        "history": history,
        "info": info,
        "current_price": current_price,
        "previous_close": previous_close,
        "open": clean_number(open_price),
        "high": clean_number(day_high),
        "low": clean_number(day_low),
        "change": clean_number(change),
        "change_pct": clean_number(change_pct),
        "trailing_pe": clean_number(trailing_pe),
        "forward_pe": clean_number(forward_pe),
        "eps": clean_number(eps),
        "week_high": clean_number(week_high),
        "week_low": clean_number(week_low),
        "earnings_date": get_earnings_date(ticker),
        "company_name": clean_text(
            info_value(
                info,
                "longName",
                "shortName",
            )
        ),
        "sector": clean_text(
            info_value(info, "sector")
        ),
        "industry": clean_text(
            info_value(info, "industry")
        ),
        "country": clean_text(
            info_value(info, "country")
        ),
        "website": clean_text(
            info_value(info, "website")
        ),
        "employees": info_value(
            info,
            "fullTimeEmployees",
        ),
        "currency": clean_text(
            info_value(info, "currency")
        ),
        "exchange": clean_text(
            info_value(
                info,
                "fullExchangeName",
                "exchange",
            )
        ),
        "market_cap": clean_number(
            info_value(
                info,
                "marketCap",
            )
        ),
        "summary": clean_text(
            info_value(
                info,
                "longBusinessSummary",
            )
        ),
        "holder_info": holder_info,
        "institutional": institutional,
        "errors": errors,
        "loaded_at": datetime.now(),
    }


# ============================================================
# SIDEBAR
# ============================================================

popular_stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Intel": "INTC",
}

with st.sidebar:
    st.markdown("## 📈 Stock Dashboard")
    st.caption(
        "Professional stock analysis with live Yahoo Finance data."
    )

    st.divider()

    selected_symbol_upper = st.session_state.selected_symbol.upper().strip()

    popular_symbols = list(popular_stocks.values())

    if selected_symbol_upper in popular_symbols:
        default_stock_index = popular_symbols.index(selected_symbol_upper)
    else:
        default_stock_index = len(popular_stocks)

    stock_choice = st.selectbox(
        "🏢 Select Stock",
        options=list(popular_stocks.keys())
        + ["Custom"],
        index=default_stock_index,
    )

    if stock_choice == "Custom":
        symbol = st.text_input(
            "Enter Stock Symbol",
            value=st.session_state.selected_symbol,
            max_chars=12,
        ).upper().strip()
    else:
        symbol = popular_stocks[stock_choice]

    period = st.selectbox(
        "📅 Analysis Period",
        options=[
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
        ],
        index=3,
    )

    st.divider()

    st.subheader("🔧 Analysis Options")

    show_prediction = st.checkbox(
        "🔮 ML Price Prediction",
        value=True,
    )

    show_technical = st.checkbox(
        "📈 Technical Charts",
        value=True,
    )

    show_performance = st.checkbox(
        "📊 Performance Metrics",
        value=True,
    )

    show_analysis = st.checkbox(
        "🧠 AI Market Analysis",
        value=True,
    )

    st.divider()

    if st.button(
        "Load Stock",
        type="primary",
        use_container_width=True,
    ):
        if symbol:
            st.session_state.selected_symbol = (
                symbol
            )
            st.rerun()

    if st.button(
        "↻ Refresh Data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.subheader("⭐ Favorites")

    if st.session_state.favorites:
        for favorite in st.session_state.favorites:
            if st.button(
                f"★ {favorite}",
                key=f"sidebar_favorite_{favorite}",
                use_container_width=True,
            ):
                st.session_state.selected_symbol = (
                    favorite
                )
                st.rerun()
    else:
        st.caption(
            "Add stocks to build your favorites list."
        )

    st.divider()

    st.caption(
        "Yahoo Finance data can be delayed, incomplete, "
        "rate-limited or temporarily unavailable."
    )


# ============================================================
# MAIN LOAD
# ============================================================

symbol = (
    st.session_state.selected_symbol
    if st.session_state.selected_symbol
    else symbol
)

symbol = symbol.upper().strip()

# Keep the active ticker synchronized with the sidebar
# without overwriting a custom/favorite selection unexpectedly.
if stock_choice != "Custom":
    symbol = popular_stocks[stock_choice]
    st.session_state.selected_symbol = symbol
else:
    symbol = symbol.upper().strip()
    if symbol:
        st.session_state.selected_symbol = symbol

st.markdown(
    '<h1 style="margin-bottom:0;">AI Stock Dashboard</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="color:#6b7280;margin-bottom:1.1rem;">'
    'Market overview, fundamentals, technical analysis, '
    'machine learning and company information'
    '</div>',
    unsafe_allow_html=True,
)

with st.spinner(
    f"📡 Fetching data for {symbol}..."
):
    data = load_stock(
        symbol,
        period,
        date.today().isoformat(),
    )

if not data.get("usable"):
    st.error(
        f"Could not fetch usable data for {symbol}."
    )

    for error in data.get("errors", []):
        st.caption(error)

    st.stop()


# ============================================================
# HERO
# ============================================================

change_pct = data["change_pct"]

if change_pct is not None and change_pct > 0:
    change_class = "hero-up"
elif change_pct is not None and change_pct < 0:
    change_class = "hero-down"
else:
    change_class = "hero-flat"

change_text = (
    f'{money(data["change"])} '
    f'({percent(change_pct)})'
    if data["change"] is not None
    else "—"
)

hero_html = (
    '<div class="hero">'
    f'<div class="hero-symbol">{symbol}</div>'
    f'<div class="hero-name">{data["company_name"]}</div>'
    f'<div class="hero-price">{money(data["current_price"])}</div>'
    f'<div class="{change_class}">{change_text}</div>'
    '</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


# ============================================================
# FAVORITE BUTTON
# ============================================================

is_favorite = (
    symbol in st.session_state.favorites
)

if is_favorite:
    if st.button(
        "★ Remove Favorite",
        use_container_width=True,
    ):
        st.session_state.favorites.remove(
            symbol
        )
        st.rerun()
else:
    if st.button(
        "☆ Add Favorite",
        type="primary",
        use_container_width=True,
    ):
        if (
            symbol
            not in st.session_state.favorites
        ):
            st.session_state.favorites.append(
                symbol
            )
        st.rerun()


# ============================================================
# REQUESTED MARKET METRICS
# ============================================================

st.markdown(
    '<div class="section-title">Market Snapshot</div>',
    unsafe_allow_html=True,
)

cols = st.columns(4)

with cols[0]:
    metric_card(
        "Previous Close",
        money(data["previous_close"]),
    )

with cols[1]:
    metric_card(
        "Open",
        money(data["open"]),
    )

with cols[2]:
    metric_card(
        "Day High",
        money(data["high"]),
    )

with cols[3]:
    metric_card(
        "Day Low",
        money(data["low"]),
    )

cols = st.columns(4)

with cols[0]:
    metric_card(
        "P/E Ratio",
        plain_number(data["trailing_pe"]),
    )

with cols[1]:
    metric_card(
        "EPS",
        money(data["eps"]),
    )

with cols[2]:
    metric_card(
        "52 Week High",
        money(data["week_high"]),
    )

with cols[3]:
    metric_card(
        "52 Week Low",
        money(data["week_low"]),
    )

cols = st.columns(3)

with cols[0]:
    metric_card(
        "Earnings Date",
        format_date(data["earnings_date"]),
    )

with cols[1]:
    metric_card(
        "Market Cap",
        large_money(data["market_cap"]),
    )

with cols[2]:
    metric_card(
        "Exchange",
        data["exchange"],
    )


# ============================================================
# ORIGINAL KEY METRICS
# ============================================================

st.markdown(
    '<div class="section-title">Dashboard Metrics</div>',
    unsafe_allow_html=True,
)

metric_cols = st.columns(5)

latest = data["history"].iloc[-1]

with metric_cols[0]:
    st.metric(
        "💰 Current Price",
        money(data["current_price"]),
        (
            f'{data["change"]:.2f} '
            f'({data["change_pct"]:+.2f}%)'
            if data["change"] is not None
            else None
        ),
    )

with metric_cols[1]:
    volume = clean_number(
        latest.get("Volume")
    )

    average_volume = clean_number(
        data["history"]["Volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    volume_change = (
        (volume - average_volume)
        / average_volume
        * 100
        if volume is not None
        and average_volume
        and average_volume > 0
        else None
    )

    st.metric(
        "📊 Volume",
        "—"
        if volume is None
        else f"{volume:,.0f}",
        None
        if volume_change is None
        else f"{volume_change:+.1f}% vs 20d avg",
    )

with metric_cols[2]:
    rsi = clean_number(
        latest.get("RSI")
    )

    if rsi is None:
        st.metric(
            "⚡ RSI (14)",
            "—",
        )
    else:
        status = (
            "Overbought"
            if rsi > 70
            else "Oversold"
            if rsi < 30
            else "Neutral"
        )

        st.metric(
            "⚡ RSI (14)",
            f"{rsi:.1f}",
            status,
        )

with metric_cols[3]:
    sma_20 = clean_number(
        latest.get("SMA_20")
    )

    if sma_20 and data["current_price"] is not None:
        distance = (
            (data["current_price"] - sma_20)
            / sma_20
            * 100
        )

        st.metric(
            "📈 vs SMA 20",
            f"{distance:+.1f}%",
            "Above"
            if distance > 0
            else "Below",
        )
    else:
        st.metric(
            "📈 vs SMA 20",
            "—",
        )

with metric_cols[4]:
    st.metric(
        "🏢 Market Cap",
        large_money(data["market_cap"]),
    )


# ============================================================
# TECHNICAL CHART
# ============================================================

if show_technical:
    st.markdown(
        '<div class="section-title">'
        '📈 Advanced Technical Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    chart = create_advanced_chart(
        data["history"],
        symbol,
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )


# ============================================================
# PERFORMANCE
# ============================================================

if show_performance:
    st.markdown(
        '<div class="section-title">'
        '📊 Performance Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    create_performance_metrics(
        data["history"],
        symbol,
    )


# ============================================================
# ML PREDICTION
# ============================================================

if show_prediction:
    st.markdown(
        '<div class="section-title">'
        '🔮 Machine Learning Price Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    analyzer = StockAnalyzer()

    with st.spinner(
        "🤖 Training prediction model..."
    ):
        model_info = (
            analyzer.train_prediction_model(
                data["history"]
            )
        )

    left, right = st.columns(2)

    with left:
        if model_info:
            prediction = (
                analyzer.predict_next_price(
                    model_info
                )
            )

            current_price = data[
                "current_price"
            ]

            predicted_change = (
                (prediction - current_price)
                / current_price
                * 100
                if current_price
                else None
            )

            st.success(
                "Model trained successfully."
            )

            p1, p2 = st.columns(2)

            with p1:
                st.metric(
                    "🎯 Next Day Prediction",
                    money(prediction),
                    (
                        f"{predicted_change:+.2f}%"
                        if predicted_change is not None
                        else None
                    ),
                )

            with p2:
                score = model_info[
                    "test_score"
                ]

                confidence = (
                    "High"
                    if score > .80
                    else "Medium"
                    if score > .60
                    else "Low"
                )

                st.metric(
                    "🎲 Model Score",
                    f"{score:.1%}",
                    confidence,
                )

            st.info(
                f'📈 Training R²: '
                f'{model_info["train_score"]:.1%} '
                f'| Test R²: '
                f'{model_info["test_score"]:.1%}'
            )

            st.caption(
                "This is a statistical model, not a guaranteed forecast."
            )

        else:
            st.warning(
                "Insufficient clean historical data for "
                "the ML model. At least 100 usable rows are required."
            )

    with right:
        if model_info:
            importance_df = pd.DataFrame(
                list(
                    model_info[
                        "feature_importance"
                    ].items()
                ),
                columns=[
                    "Feature",
                    "Importance",
                ],
            ).sort_values(
                "Importance",
                ascending=False,
            ).head(10)

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Model Features",
                template="plotly_white",
            )

            fig_importance.update_layout(
                height=400,
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True,
            )


# ============================================================
# AI MARKET ANALYSIS
# ============================================================

if show_analysis:
    st.markdown(
        '<div class="section-title">'
        '🧠 AI-Powered Market Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    analyzer = StockAnalyzer()

    with st.spinner(
        "Generating market analysis..."
    ):
        analysis = (
            analyzer.generate_market_analysis(
                data["history"],
                data["info"],
                symbol,
            )
        )

    for index, insight in enumerate(
        analysis
    ):
        if index == 0:
            if (
                "🚀" in insight
                or "🟢" in insight
            ):
                st.success(insight)
            elif (
                "🔴" in insight
                or "🔻" in insight
            ):
                st.error(insight)
            else:
                st.warning(insight)
        else:
            st.info(insight)


# ============================================================
# TABS
# ============================================================

st.divider()

tab_company, tab_raw, tab_technical = st.tabs(
    [
        "📋 Company Info",
        "📊 Raw Data",
        "🔧 Technical Indicators",
    ]
)


# ============================================================
# COMPANY TAB
# ============================================================

with tab_company:
    left, right = st.columns(2)

    with left:
        st.subheader("🏢 Company Details")

        st.write(
            f"**Company Name:** "
            f"{data['company_name']}"
        )

        st.write(
            f"**Sector:** "
            f"{data['sector']}"
        )

        st.write(
            f"**Industry:** "
            f"{data['industry']}"
        )

        st.write(
            f"**Country:** "
            f"{data['country']}"
        )

        if data["website"] != "—":
            st.link_button(
                "Open Company Website",
                data["website"],
            )
        else:
            st.write(
                "**Website:** —"
            )

        st.write(
            f"**Employees:** "
            f"{integer(data['employees'])}"
        )

    with right:
        st.subheader("📈 Financial Metrics")

        st.write(
            f"**P/E Ratio:** "
            f"{plain_number(data['trailing_pe'])}"
        )

        st.write(
            f"**Forward P/E:** "
            f"{plain_number(data['forward_pe'])}"
        )

        st.write(
            f"**EPS:** "
            f"{money(data['eps'])}"
        )

        peg = info_value(
            data["info"],
            "pegRatio",
        )

        price_to_book = info_value(
            data["info"],
            "priceToBook",
        )

        dividend_yield = info_value(
            data["info"],
            "dividendYield",
        )

        beta = info_value(
            data["info"],
            "beta",
        )

        st.write(
            f"**PEG Ratio:** "
            f"{plain_number(peg)}"
        )

        st.write(
            f"**Price to Book:** "
            f"{plain_number(price_to_book)}"
        )

        st.write(
            f"**Dividend Yield:** "
            f"{'—' if clean_number(dividend_yield) is None else f'{clean_number(dividend_yield) * 100:.2f}%'}"
        )

        st.write(
            f"**Beta:** "
            f"{plain_number(beta)}"
        )

        st.write(
            f"**52 Week High:** "
            f"{money(data['week_high'])}"
        )

        st.write(
            f"**52 Week Low:** "
            f"{money(data['week_low'])}"
        )

    if data["summary"] != "—":
        with st.expander(
            "Business Description"
        ):
            st.write(
                data["summary"]
            )

    # Ownership section with stable Yahoo info fields.
    st.subheader(
        "Major Holders Breakdown"
    )

    holder = data["holder_info"]

    h1, h2, h3 = st.columns(3)

    with h1:
        metric_card(
            "Insider Ownership",
            clean_text(
                holder.get("insider")
            ),
        )

    with h2:
        metric_card(
            "Institutional Ownership",
            clean_text(
                holder.get("institution")
            ),
        )

    with h3:
        float_shares = holder.get(
            "float_shares"
        )

        metric_card(
            "Float Shares",
            integer(float_shares),
        )

    institutional = data[
        "institutional"
    ]

    if (
        isinstance(
            institutional,
            pd.DataFrame,
        )
        and not institutional.empty
    ):
        st.subheader(
            "Institutional Holders"
        )

        preferred = [
            "Holder",
            "Shares",
            "Date Reported",
            "% Out",
            "Value",
        ]

        available = [
            column
            for column in preferred
            if column
            in institutional.columns
        ]

        display = (
            institutional[available]
            if available
            else institutional
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(
            "Yahoo Finance did not return an institutional-holder table for this symbol."
        )


# ============================================================
# RAW DATA TAB
# ============================================================

with tab_raw:
    st.subheader(
        "📊 Recent Price Data"
    )

    raw = data["history"][
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ].tail(30).copy()

    raw.index = raw.index.strftime(
        "%Y-%m-%d"
    )

    st.dataframe(
        raw,
        use_container_width=True,
    )

    csv = raw.to_csv()

    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{symbol}_stock_data.csv",
        mime="text/csv",
    )


# ============================================================
# TECHNICAL TAB
# ============================================================

with tab_technical:
    st.subheader(
        "🔧 Technical Indicators"
    )

    # ATR intentionally excluded.
    technical_columns = [
        "Close",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "EMA_12",
        "EMA_26",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_histogram",
        "BB_upper",
        "BB_middle",
        "BB_lower",
        "Volume_ratio",
        "Stoch_K",
        "Stoch_D",
    ]

    available = [
        column
        for column in technical_columns
        if column
        in data["history"].columns
    ]

    technical = (
        data["history"][available]
        .tail(20)
        .copy()
    )

    technical.index = (
        technical.index.strftime(
            "%Y-%m-%d"
        )
    )

    st.dataframe(
        technical.round(4),
        use_container_width=True,
    )


# ============================================================
# FAVORITES SECTION
# ============================================================

st.markdown(
    '<div class="section-title">⭐ Favorites</div>',
    unsafe_allow_html=True,
)

if st.session_state.favorites:
    favorite_columns = st.columns(
        min(
            4,
            len(
                st.session_state.favorites
            ),
        )
    )

    for index, favorite in enumerate(
        st.session_state.favorites
    ):
        with favorite_columns[
            index
            % len(favorite_columns)
        ]:
            st.markdown(
                '<div class="favorite-box">'
                f'<strong>★ {favorite}</strong>'
                '</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                f"Open {favorite}",
                key=f"open_main_{favorite}",
                use_container_width=True,
            ):
                st.session_state.selected_symbol = (
                    favorite
                )
                st.rerun()
else:
    st.caption(
        "No favorite stocks yet."
    )


# ============================================================
# DATA INTEGRITY
# ============================================================

st.markdown(
    '<div class="notice">'
    '<strong>Data integrity:</strong> Missing values are shown as '
    '<strong>—</strong>. The app never invents prices, EPS, P/E, '
    'company information or earnings dates. Yahoo Finance is a free '
    'data source and may provide delayed, incomplete or rate-limited data.'
    '</div>',
    unsafe_allow_html=True,
)

if data["errors"]:
    with st.expander(
        "Non-fatal data-source notices"
    ):
        for error in data["errors"]:
            st.warning(error)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    """
<div style="text-align:center;color:#777;padding:18px;">
    <strong>📈 AI Stock Dashboard</strong><br>
    <span>For educational and informational purposes only. Not financial advice.</span>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    f"Loaded {data['loaded_at'].strftime('%b %d, %Y at %I:%M:%S %p')} "
    "· Data cache: 15 minutes · Daily cache key: enabled"
)