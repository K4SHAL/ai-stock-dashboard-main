import math
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --orange: #ff7a00;
            --orange-dark: #e96500;
            --orange-light: #fff4e8;
            --text: #171717;
            --muted: #6b7280;
            --border: #e7e7e7;
            --white: #ffffff;
            --green: #159447;
            --red: #d83a3a;
        }

        .stApp {
            background: #ffffff;
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: #fafafa;
            border-right: 1px solid #eeeeee;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--text);
        }

        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.15rem;
            color: #111111;
        }

        .subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        .stock-header {
            background: linear-gradient(
                135deg,
                #fff8f1 0%,
                #ffffff 65%
            );
            border: 1px solid #f0dfcf;
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 5px 25px rgba(0, 0, 0, 0.035);
        }

        .ticker {
            font-size: 2rem;
            font-weight: 850;
            color: #111111;
            letter-spacing: -0.03em;
        }

        .company-name {
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: 2px;
        }

        .price {
            font-size: 2.5rem;
            font-weight: 850;
            letter-spacing: -0.04em;
            margin-top: 10px;
        }

        .positive {
            color: var(--green);
            font-weight: 700;
        }

        .negative {
            color: var(--red);
            font-weight: 700;
        }

        .neutral {
            color: var(--muted);
            font-weight: 700;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            margin: 24px 0 12px 0;
            color: #171717;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 17px;
            min-height: 105px;
            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease,
                border-color 0.18s ease;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            border-color: #ffc38d;
            box-shadow: 0 10px 28px rgba(255, 122, 0, 0.09);
        }

        .metric-label {
            color: #737373;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .metric-value {
            color: #111111;
            font-size: 1.25rem;
            font-weight: 800;
            word-break: break-word;
        }

        .info-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 14px;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            padding: 9px 0;
            border-bottom: 1px solid #f1f1f1;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            color: #737373;
            font-size: 0.88rem;
        }

        .info-value {
            color: #171717;
            font-weight: 700;
            text-align: right;
            word-break: break-word;
        }

        .source-box {
            background: #fff8f1;
            border: 1px solid #ffd7b0;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 20px;
            color: #663500;
            font-size: 0.86rem;
        }

        .favorite-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 15px;
            margin-bottom: 8px;
            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease;
        }

        .favorite-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
        }

        div.stButton > button {
            border-radius: 11px;
            border: 1px solid #dedede;
            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                background 0.15s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(255, 122, 0, 0.14);
            border-color: #ffb36d;
        }

        div.stButton > button[kind="primary"] {
            background: var(--orange);
            color: white;
            border: none;
        }

        div.stButton > button[kind="primary"]:hover {
            background: var(--orange-dark);
        }

        .small-muted {
            color: #888888;
            font-size: 0.78rem;
        }

        .error-box {
            background: #fff1f1;
            border: 1px solid #f3b5b5;
            color: #8b1e1e;
            padding: 14px;
            border-radius: 12px;
        }

        .warning-box {
            background: #fff9ed;
            border: 1px solid #f3d79a;
            color: #765000;
            padding: 14px;
            border-radius: 12px;
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

if "last_loaded" not in st.session_state:
    st.session_state.last_loaded = None


# ============================================================
# HELPERS
# ============================================================

def clean_number(value):
    """Return None for invalid numeric values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def first_valid(*values):
    """Return the first valid non-null value."""
    for value in values:
        if value is not None:
            return value
    return None


def fmt_num(value, decimals=2, prefix=""):
    value = clean_number(value)

    if value is None:
        return "—"

    return f"{prefix}{value:,.{decimals}f}"


def fmt_integer(value):
    value = clean_number(value)

    if value is None:
        return "—"

    return f"{int(value):,}"


def fmt_large_number(value):
    value = clean_number(value)

    if value is None:
        return "—"

    abs_value = abs(value)

    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"

    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    if abs_value >= 1_000:
        return f"${value / 1_000:.2f}K"

    return f"${value:,.2f}"


def fmt_pct(value):
    value = clean_number(value)

    if value is None:
        return "—"

    return f"{value:+.2f}%"


def fmt_date(value):
    if value is None:
        return "—"

    try:
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()

        if isinstance(value, datetime):
            return value.strftime("%b %d, %Y")

        if isinstance(value, date):
            return value.strftime("%b %d, %Y")

        parsed = pd.to_datetime(value, errors="coerce")

        if pd.isna(parsed):
            return "—"

        return parsed.strftime("%b %d, %Y")

    except Exception:
        return "—"


def safe_text(value):
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "n/a",
        "null",
        "nat",
    }:
        return "—"

    return text


def get_info_value(info, *keys):
    if not isinstance(info, dict):
        return None

    for key in keys:
        value = info.get(key)

        if value is not None:
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass

            return value

    return None


def safe_history(ticker, period="5d", interval="1d"):
    try:
        df = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
        )

        if df is None or df.empty:
            return None

        df = df.copy()

        required = {"Open", "High", "Low", "Close"}

        if not required.issubset(df.columns):
            return None

        df = df.dropna(subset=["Close"])

        if df.empty:
            return None

        return df

    except Exception:
        return None


def extract_date_from_value(value):
    """Try to extract a date from various Yahoo/yfinance formats."""
    if value is None:
        return None

    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()

        if isinstance(value, datetime):
            return value

        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())

        parsed = pd.to_datetime(value, errors="coerce")

        if pd.isna(parsed):
            return None

        if isinstance(parsed, pd.Timestamp):
            return parsed.to_pydatetime()

        return parsed

    except Exception:
        return None


# ============================================================
# EARNINGS DATE
# ============================================================

def get_earnings_date(ticker):
    """
    Prefer the next future earnings date.
    If unavailable, use the latest known earnings date.
    """

    try:
        earnings = ticker.get_earnings_dates(limit=12)

        if earnings is not None and not earnings.empty:
            index = earnings.index

            dates = []

            for item in index:
                parsed = extract_date_from_value(item)

                if parsed is not None:
                    dates.append(parsed)

            if dates:
                now = datetime.now()

                future_dates = [
                    d for d in dates
                    if d >= now
                ]

                if future_dates:
                    return min(future_dates)

                return max(dates)

    except Exception:
        pass

    # Fallback to calendar
    try:
        calendar = ticker.get_calendar()

        if isinstance(calendar, dict):
            for key in [
                "Earnings Date",
                "earningsDate",
                "Earnings Dates",
            ]:
                value = calendar.get(key)

                if value is None:
                    continue

                if isinstance(value, (list, tuple)):
                    dates = []

                    for item in value:
                        parsed = extract_date_from_value(item)

                        if parsed:
                            dates.append(parsed)

                    if dates:
                        return min(dates)

                parsed = extract_date_from_value(value)

                if parsed:
                    return parsed

        elif isinstance(calendar, pd.DataFrame):
            for column in calendar.columns:
                if "earn" in str(column).lower():
                    for value in calendar[column].tolist():
                        parsed = extract_date_from_value(value)

                        if parsed:
                            return parsed

            for index in calendar.index:
                if "earn" in str(index).lower():
                    values = calendar.loc[index]

                    if isinstance(values, pd.Series):
                        for value in values.tolist():
                            parsed = extract_date_from_value(value)

                            if parsed:
                                return parsed

    except Exception:
        pass

    return None


# ============================================================
# HOLDER DATA
# ============================================================

def parse_major_holders(holders):
    """
    yfinance's major_holders DataFrame can change orientation.
    Convert it into a simple dictionary without assuming one layout.
    """

    result = {}

    if holders is None:
        return result

    if isinstance(holders, pd.Series):
        holders = holders.to_frame()

    if not isinstance(holders, pd.DataFrame):
        return result

    if holders.empty:
        return result

    # Typical format:
    #
    #                    Value
    #  0.01%    % of Shares Held by All Insider
    #  0.65%    % of Shares Held by Institutions
    #
    for _, row in holders.iterrows():

        values = row.tolist()

        if len(values) < 2:
            continue

        value_a = values[0]
        value_b = values[-1]

        text = None
        metric = None

        if isinstance(value_a, str):
            text = value_a
            metric = value_b

        elif isinstance(value_b, str):
            text = value_b
            metric = value_a

        if text is None:
            continue

        text_lower = text.lower()

        if "insider" in text_lower:
            result["insider"] = metric

        elif "institution" in text_lower:
            result["institution"] = metric

        elif "float" in text_lower:
            result["float"] = metric

    return result


# ============================================================
# LOAD STOCK DATA
# ============================================================

@st.cache_data(
    ttl=86400,
    max_entries=250,
    show_spinner=False,
)
def load_stock(symbol, refresh_day):
    """
    Load stock data.

    refresh_day is intentionally passed into the cache key so the
    cache naturally refreshes once per calendar day.
    """

    symbol = symbol.upper().strip()

    ticker = yf.Ticker(symbol)

    errors = []

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history_5d = safe_history(
        ticker,
        period="5d",
        interval="1d",
    )

    history_1y = None

    if history_5d is None or history_5d.empty:
        history_1y = safe_history(
            ticker,
            period="1y",
            interval="1d",
        )

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    info = {}

    try:
        info = ticker.get_info()

        if not isinstance(info, dict):
            info = {}

    except Exception as exc:
        errors.append(
            f"Company/fundamental data request failed: {type(exc).__name__}"
        )

    # --------------------------------------------------------
    # FAST INFO
    # --------------------------------------------------------

    fast_info = {}

    try:
        fast_info = dict(ticker.fast_info)

    except Exception:
        fast_info = {}

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    current_price = first_valid(
        get_info_value(
            info,
            "currentPrice",
            "regularMarketPrice",
        ),
        fast_info.get("last_price"),
    )

    # --------------------------------------------------------
    # OHLC
    # --------------------------------------------------------

    open_price = None
    high_price = None
    low_price = None
    previous_close = None

    if history_5d is not None and not history_5d.empty:

        latest = history_5d.iloc[-1]

        open_price = clean_number(latest.get("Open"))
        high_price = clean_number(latest.get("High"))
        low_price = clean_number(latest.get("Low"))

        if len(history_5d) >= 2:
            previous_close = clean_number(
                history_5d.iloc[-2].get("Close")
            )

    # Use Yahoo quote values if available.
    open_price = first_valid(
        get_info_value(info, "open"),
        open_price,
        fast_info.get("open"),
    )

    high_price = first_valid(
        get_info_value(info, "dayHigh"),
        high_price,
        fast_info.get("day_high"),
    )

    low_price = first_valid(
        get_info_value(info, "dayLow"),
        low_price,
        fast_info.get("day_low"),
    )

    previous_close = first_valid(
        get_info_value(
            info,
            "previousClose",
            "regularMarketPreviousClose",
        ),
        previous_close,
        fast_info.get("previous_close"),
    )

    # --------------------------------------------------------
    # CURRENT PRICE FALLBACK
    # --------------------------------------------------------

    if current_price is None:

        if history_5d is not None and not history_5d.empty:
            current_price = clean_number(
                history_5d.iloc[-1]["Close"]
            )

        elif history_1y is not None and not history_1y.empty:
            current_price = clean_number(
                history_1y.iloc[-1]["Close"]
            )

    # --------------------------------------------------------
    # DAILY CHANGE
    # --------------------------------------------------------

    change = None
    change_percent = None

    if current_price is not None and previous_close:
        change = current_price - previous_close
        change_percent = (
            change / previous_close
        ) * 100

    # --------------------------------------------------------
    # PE
    # --------------------------------------------------------

    pe_ratio = first_valid(
        get_info_value(
            info,
            "trailingPE",
        ),
        get_info_value(
            info,
            "forwardPE",
        ),
    )

    # --------------------------------------------------------
    # EPS
    # --------------------------------------------------------

    eps = first_valid(
        get_info_value(
            info,
            "trailingEps",
        ),
        get_info_value(
            info,
            "forwardEps",
        ),
    )

    # --------------------------------------------------------
    # 52 WEEK HIGH / LOW
    # --------------------------------------------------------

    week_52_high = get_info_value(
        info,
        "fiftyTwoWeekHigh",
    )

    week_52_low = get_info_value(
        info,
        "fiftyTwoWeekLow",
    )

    # Fallback to 1-year history if Yahoo's quote fields
    # don't provide these values.

    if (
        week_52_high is None
        or week_52_low is None
    ):

        if history_1y is None:
            history_1y = safe_history(
                ticker,
                period="1y",
                interval="1d",
            )

        if history_1y is not None and not history_1y.empty:

            if week_52_high is None:
                try:
                    week_52_high = history_1y[
                        "High"
                    ].max()
                except Exception:
                    pass

            if week_52_low is None:
                try:
                    week_52_low = history_1y[
                        "Low"
                    ].min()
                except Exception:
                    pass

    # --------------------------------------------------------
    # EARNINGS
    # --------------------------------------------------------

    earnings_date = get_earnings_date(ticker)

    # --------------------------------------------------------
    # COMPANY DETAILS
    # --------------------------------------------------------

    company_name = safe_text(
        get_info_value(
            info,
            "longName",
            "shortName",
        )
    )

    sector = safe_text(
        get_info_value(
            info,
            "sector",
        )
    )

    industry = safe_text(
        get_info_value(
            info,
            "industry",
        )
    )

    country = safe_text(
        get_info_value(
            info,
            "country",
        )
    )

    website = safe_text(
        get_info_value(
            info,
            "website",
        )
    )

    employees = get_info_value(
        info,
        "fullTimeEmployees",
    )

    currency = safe_text(
        get_info_value(
            info,
            "currency",
        )
    )

    exchange = safe_text(
        get_info_value(
            info,
            "exchange",
            "fullExchangeName",
        )
    )

    market_cap = get_info_value(
        info,
        "marketCap",
    )

    business_summary = safe_text(
        get_info_value(
            info,
            "longBusinessSummary",
        )
    )

    # --------------------------------------------------------
    # MAJOR HOLDERS
    # --------------------------------------------------------

    major_holders = None

    try:
        major_holders = ticker.get_major_holders()

    except Exception as exc:
        errors.append(
            f"Major holders request failed: {type(exc).__name__}"
        )

    holder_data = parse_major_holders(
        major_holders
    )

    # --------------------------------------------------------
    # INSTITUTIONAL HOLDERS
    # --------------------------------------------------------

    institutional_holders = None

    try:
        institutional_holders = (
            ticker.get_institutional_holders()
        )

    except Exception as exc:
        errors.append(
            f"Institutional holders request failed: "
            f"{type(exc).__name__}"
        )

    institutional_count = None

    if (
        isinstance(
            institutional_holders,
            pd.DataFrame,
        )
        and not institutional_holders.empty
    ):
        institutional_count = len(
            institutional_holders
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    usable_data = (
        current_price is not None
        or company_name != "—"
        or (
            history_5d is not None
            and not history_5d.empty
        )
    )

    return {
        "symbol": symbol,
        "current_price": current_price,
        "previous_close": previous_close,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "change": change,
        "change_percent": change_percent,
        "pe_ratio": pe_ratio,
        "eps": eps,
        "earnings_date": earnings_date,
        "week_52_high": week_52_high,
        "week_52_low": week_52_low,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "country": country,
        "website": website,
        "employees": employees,
        "currency": currency,
        "exchange": exchange,
        "market_cap": market_cap,
        "business_summary": business_summary,
        "major_holders": holder_data,
        "institutional_holders": institutional_holders,
        "institutional_count": institutional_count,
        "usable_data": usable_data,
        "errors": errors,
        "loaded_at": datetime.now(),
    }


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size: 1.5rem;
            font-weight: 850;
            margin-bottom: 4px;
        ">
            📈 Stock Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Clean market data, fundamentals and favorites."
    )

    st.divider()

    ticker_input = st.text_input(
        "Stock Symbol",
        value=st.session_state.selected_symbol,
        placeholder="AAPL",
        help="Enter a stock ticker such as AAPL, MSFT, NVDA or TSLA.",
    ).upper().strip()

    if st.button(
        "Load Stock",
        type="primary",
        use_container_width=True,
    ):
        if ticker_input:

            st.session_state.selected_symbol = (
                ticker_input
            )

            st.session_state.last_loaded = (
                datetime.now()
            )

            st.rerun()

    st.markdown("### Favorites")

    if st.session_state.favorites:

        for favorite in st.session_state.favorites:

            if st.button(
                f"★ {favorite}",
                key=f"favorite_{favorite}",
                use_container_width=True,
            ):
                st.session_state.selected_symbol = (
                    favorite
                )

                st.session_state.last_loaded = (
                    datetime.now()
                )

                st.rerun()

    else:
        st.caption(
            "No favorite stocks yet."
        )

    st.divider()

    if st.button(
        "↻ Refresh Data",
        use_container_width=True,
    ):

        st.cache_data.clear()

        st.session_state.last_loaded = (
            datetime.now()
        )

        st.rerun()

    st.caption(
        "Data cache refreshes automatically every 24 hours."
    )

    st.divider()

    st.markdown(
        """
        <div class="small-muted">
        Data source: Yahoo Finance through yfinance.<br><br>
        Market data can be delayed and may be unavailable
        during provider outages or rate limits.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD SELECTED STOCK
# ============================================================

symbol = (
    st.session_state.selected_symbol
    .upper()
    .strip()
)

if not symbol:
    symbol = "AAPL"

today = date.today().isoformat()

with st.spinner(
    f"Loading {symbol} market data..."
):

    data = load_stock(
        symbol,
        today,
    )


# ============================================================
# ERROR HANDLING
# ============================================================

if not data["usable_data"]:

    st.markdown(
        f"""
        <div class="error-box">
            <strong>Could not load {symbol}.</strong><br><br>
            Yahoo Finance did not return usable market data.
            This dashboard will not invent numbers just to make
            the interface look pretty.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data["errors"]:

        st.write("Technical notices:")

        for error in data["errors"]:
            st.caption(error)

    st.stop()


# ============================================================
# HEADER
# ============================================================

company_name = data["company_name"]

price = data["current_price"]

change = data["change"]

change_percent = data["change_percent"]


if change_percent is not None:

    if change_percent > 0:

        change_class = "positive"

    elif change_percent < 0:

        change_class = "negative"

    else:

        change_class = "neutral"

else:

    change_class = "neutral"


change_text = (
    f"{fmt_num(change, 2)} "
    f"({fmt_pct(change_percent)})"
)


st.markdown(
    '<div class="main-title">AI Stock Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Market overview, fundamentals and company information'
    '</div>',
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div class="stock-header">
        <div class="ticker">{symbol}</div>
        <div class="company-name">{company_name}</div>

        <div class="price">
            {fmt_num(price, 2)}
        </div>

        <div class="{change_class}">
            {change_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FAVORITE BUTTON
# ============================================================

is_favorite = (
    symbol in st.session_state.favorites
)

favorite_col1, favorite_col2, _ = st.columns(
    [1.2, 1.2, 5]
)

with favorite_col1:

    if not is_favorite:

        if st.button(
            "☆ Add Favorite",
            type="primary",
            use_container_width=True,
        ):

            if symbol not in st.session_state.favorites:

                st.session_state.favorites.append(
                    symbol
                )

            st.rerun()

    else:

        if st.button(
            "★ Remove Favorite",
            use_container_width=True,
        ):

            st.session_state.favorites = [
                x
                for x in st.session_state.favorites
                if x != symbol
            ]

            st.rerun()


# ============================================================
# MARKET SNAPSHOT
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Market Snapshot'
    '</div>',
    unsafe_allow_html=True,
)

snapshot_cols = st.columns(4)

with snapshot_cols[0]:
    metric_card(
        "Previous Close",
        fmt_num(
            data["previous_close"],
            2,
        ),
    )

with snapshot_cols[1]:
    metric_card(
        "Open",
        fmt_num(
            data["open"],
            2,
        ),
    )

with snapshot_cols[2]:
    metric_card(
        "Day High",
        fmt_num(
            data["high"],
            2,
        ),
    )

with snapshot_cols[3]:
    metric_card(
        "Day Low",
        fmt_num(
            data["low"],
            2,
        ),
    )


snapshot_cols_2 = st.columns(4)

with snapshot_cols_2[0]:
    metric_card(
        "P/E Ratio",
        fmt_num(
            data["pe_ratio"],
            2,
        ),
    )

with snapshot_cols_2[1]:
    metric_card(
        "EPS",
        fmt_num(
            data["eps"],
            2,
        ),
    )

with snapshot_cols_2[2]:
    metric_card(
        "52 Week High",
        fmt_num(
            data["week_52_high"],
            2,
        ),
    )

with snapshot_cols_2[3]:
    metric_card(
        "52 Week Low",
        fmt_num(
            data["week_52_low"],
            2,
        ),
    )


# ============================================================
# EARNINGS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Earnings'
    '</div>',
    unsafe_allow_html=True,
)

earn_cols = st.columns(3)

with earn_cols[0]:

    metric_card(
        "Next / Latest Earnings Date",
        fmt_date(
            data["earnings_date"]
        ),
    )

with earn_cols[1]:

    metric_card(
        "Market Cap",
        fmt_large_number(
            data["market_cap"]
        ),
    )

with earn_cols[2]:

    metric_card(
        "Exchange",
        safe_text(
            data["exchange"]
        ),
    )


# ============================================================
# COMPANY DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Company Details'
    '</div>',
    unsafe_allow_html=True,
)

details_left, details_right = st.columns(2)

with details_left:

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True,
    )

    rows = [
        (
            "Company Name",
            data["company_name"],
        ),
        (
            "Sector",
            data["sector"],
        ),
        (
            "Industry",
            data["industry"],
        ),
        (
            "Country",
            data["country"],
        ),
    ]

    for label, value in rows:

        st.markdown(
            f"""
            <div class="info-row">
                <div class="info-label">
                    {label}
                </div>

                <div class="info-value">
                    {safe_text(value)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


with details_right:

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True,
    )

    website = data["website"]

    if website != "—":

        website_display = (
            f'<a href="{website}" '
            'target="_blank">'
            f'{website}'
            '</a>'
        )

    else:

        website_display = "—"

    rows = [
        (
            "Website",
            website_display,
        ),
        (
            "Employees",
            fmt_integer(
                data["employees"]
            ),
        ),
        (
            "Currency",
            data["currency"],
        ),
        (
            "Exchange",
            data["exchange"],
        ),
    ]

    for label, value in rows:

        st.markdown(
            f"""
            <div class="info-row">
                <div class="info-label">
                    {label}
                </div>

                <div class="info-value">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# BUSINESS DESCRIPTION
# ============================================================

if data["business_summary"] != "—":

    with st.expander(
        "Business Description"
    ):

        st.write(
            data["business_summary"]
        )


# ============================================================
# HOLDERS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Major Holders Breakdown'
    '</div>',
    unsafe_allow_html=True,
)

holders = data["major_holders"]

holder_cols = st.columns(3)

with holder_cols[0]:

    insider = holders.get(
        "insider"
    )

    metric_card(
        "Insider Ownership",
        safe_text(insider),
    )

with holder_cols[1]:

    institution = holders.get(
        "institution"
    )

    metric_card(
        "Institutional Ownership",
        safe_text(institution),
    )

with holder_cols[2]:

    float_value = holders.get(
        "float"
    )

    metric_card(
        "Float",
        safe_text(float_value),
    )


# ============================================================
# INSTITUTIONAL HOLDERS
# ============================================================

institutional_df = data[
    "institutional_holders"
]

if (
    isinstance(
        institutional_df,
        pd.DataFrame,
    )
    and not institutional_df.empty
):

    st.markdown(
        '<div class="section-title">'
        'Institutional Holders'
        '</div>',
        unsafe_allow_html=True,
    )

    display_df = institutional_df.copy()

    # Limit display columns to useful ones.
    preferred_columns = [
        "Holder",
        "Shares",
        "Date Reported",
        "% Out",
        "Value",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in display_df.columns
    ]

    if available_columns:

        display_df = display_df[
            available_columns
        ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FAVORITES SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Favorites'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.favorites:

    favorite_cols = st.columns(
        min(
            len(
                st.session_state.favorites
            ),
            4,
        )
    )

    for index, favorite in enumerate(
        st.session_state.favorites
    ):

        with favorite_cols[
            index % len(favorite_cols)
        ]:

            if favorite == symbol:

                favorite_label = (
                    f"★ {favorite}"
                )

            else:

                favorite_label = (
                    f"☆ {favorite}"
                )

            st.markdown(
                f"""
                <div class="favorite-card">
                    <strong>
                        {favorite_label}
                    </strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:

    st.caption(
        "Your favorite stocks will appear here."
    )


# ============================================================
# DATA INTEGRITY
# ============================================================

st.markdown(
    '<div class="source-box">'
    '<strong>Data integrity</strong><br>'
    'This dashboard uses Yahoo Finance data through '
    'yfinance. Missing values are displayed as '
    '<strong>—</strong> rather than being fabricated. '
    'Yahoo Finance data may be delayed, incomplete, '
    'temporarily unavailable, or rate-limited.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# NON-FATAL ERRORS
# ============================================================

if data["errors"]:

    with st.expander(
        "Non-fatal data-source notices"
    ):

        for error in data["errors"]:

            st.warning(error)


# ============================================================
# LAST UPDATED
# ============================================================

loaded_at = data["loaded_at"]

st.markdown(
    f"""
    <div style="
        text-align: center;
        color: #999;
        font-size: 0.75rem;
        margin-top: 28px;
        padding-bottom: 20px;
    ">
        Data loaded: {loaded_at.strftime("%b %d, %Y at %I:%M:%S %p")}
        <br>
        Cache refresh interval: 24 hours
    </div>
    """,
    unsafe_allow_html=True,
)