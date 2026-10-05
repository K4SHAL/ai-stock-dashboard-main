import math
from datetime import date, datetime

import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Stock Dashboard",
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
.stApp {
    background: #ffffff;
}
section[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #eeeeee;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.hero {
    background: linear-gradient(135deg, #fff7ef 0%, #ffffff 72%);
    border: 1px solid #f4dcc7;
    border-radius: 22px;
    padding: 26px;
    margin: 12px 0 20px 0;
}
.hero-symbol {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #111111;
}
.hero-name {
    color: #6b7280;
    margin-top: 2px;
    font-size: 0.95rem;
}
.hero-price {
    font-size: 2.7rem;
    font-weight: 850;
    letter-spacing: -0.045em;
    color: #111111;
    margin-top: 14px;
}
.hero-change-up {
    color: #168447;
    font-weight: 750;
    font-size: 1rem;
}
.hero-change-down {
    color: #d13a3a;
    font-weight: 750;
    font-size: 1rem;
}
.hero-change-flat {
    color: #6b7280;
    font-weight: 750;
    font-size: 1rem;
}
.section-heading {
    font-size: 1.35rem;
    font-weight: 800;
    color: #171717;
    margin: 24px 0 12px 0;
}
.metric {
    background: #ffffff;
    border: 1px solid #e7e7e7;
    border-radius: 16px;
    padding: 17px;
    min-height: 102px;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.metric:hover {
    transform: translateY(-2px);
    border-color: #ffbd7b;
    box-shadow: 0 9px 25px rgba(255, 122, 0, .09);
}
.metric-label {
    color: #737373;
    font-size: .78rem;
    font-weight: 650;
    margin-bottom: 8px;
}
.metric-value {
    color: #111111;
    font-size: 1.25rem;
    font-weight: 800;
}
.notice {
    background: #fff7ef;
    border: 1px solid #ffd4aa;
    border-radius: 14px;
    padding: 14px 16px;
    color: #704000;
}
.small-muted {
    color: #8a8a8a;
    font-size: .78rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# STATE
# ============================================================

if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "AAPL"


# ============================================================
# FORMATTERS
# ============================================================

def number(value):
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
    for value in values:
        if value is not None:
            return value
    return None


def text(value):
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
    value = number(value)
    return "—" if value is None else f"${value:,.{decimals}f}"


def plain_number(value, decimals=2):
    value = number(value)
    return "—" if value is None else f"{value:,.{decimals}f}"


def integer(value):
    value = number(value)
    return "—" if value is None else f"{int(value):,}"


def large_money(value):
    value = number(value)
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
    value = number(value)
    return "—" if value is None else f"{value:+.2f}%"


def fmt_date(value):
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
        if value is not None:
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value
    return None


# ============================================================
# DATA HELPERS
# ============================================================

def get_history(ticker, period="5d"):
    try:
        frame = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        if frame is None or frame.empty:
            return None
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(frame.columns):
            return None
        return frame.dropna(subset=["Close"])
    except Exception:
        return None


def get_earnings_date(ticker):
    try:
        frame = ticker.get_earnings_dates(limit=12)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            dates = []
            for value in frame.index:
                parsed = pd.to_datetime(value, errors="coerce")
                if not pd.isna(parsed):
                    dates.append(parsed.to_pydatetime())
            if dates:
                now = datetime.now()
                future = [d for d in dates if d >= now]
                return min(future) if future else max(dates)
    except Exception:
        pass

    try:
        calendar = ticker.get_calendar()
        if isinstance(calendar, dict):
            value = calendar.get("Earnings Date")
            if isinstance(value, (list, tuple)):
                dates = []
                for item in value:
                    parsed = pd.to_datetime(item, errors="coerce")
                    if not pd.isna(parsed):
                        dates.append(parsed)
                if dates:
                    return min(dates)
            parsed = pd.to_datetime(value, errors="coerce")
            if not pd.isna(parsed):
                return parsed
    except Exception:
        pass

    return None


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
            label, value = values[0], values[-1]
        elif isinstance(values[-1], str):
            label, value = values[-1], values[0]

        if not label:
            continue

        key = label.lower()

        if "insider" in key:
            result["insider"] = value
        elif "institution" in key:
            result["institution"] = value
        elif "float" in key:
            result["float"] = value

    return result


def load_holders(ticker):
    major = None
    institutions = None
    errors = []

    try:
        major = ticker.get_major_holders()
    except Exception as exc:
        errors.append(f"Major holders unavailable: {type(exc).__name__}")

    try:
        institutions = ticker.get_institutional_holders()
    except Exception as exc:
        errors.append(f"Institutional holders unavailable: {type(exc).__name__}")

    return major, institutions, errors


# ============================================================
# STOCK LOADER
# ============================================================

@st.cache_data(ttl=86400, max_entries=250, show_spinner=False)
def load_stock(symbol, cache_day):
    symbol = symbol.strip().upper()
    ticker = yf.Ticker(symbol)

    errors = []

    # Fast/quote information.
    info = {}
    try:
        info = ticker.get_info()
        if not isinstance(info, dict):
            info = {}
    except Exception as exc:
        errors.append(f"Yahoo fundamentals unavailable: {type(exc).__name__}")

    fast = {}
    try:
        fast = dict(ticker.fast_info)
    except Exception:
        fast = {}

    # Daily history.
    history = get_history(ticker, "5d")
    if history is None:
        history = get_history(ticker, "1y")

    # Current quote.
    current_price = first_valid(
        info_value(info, "currentPrice", "regularMarketPrice"),
        fast.get("last_price"),
    )

    previous_close = first_valid(
        info_value(info, "previousClose", "regularMarketPreviousClose"),
        fast.get("previous_close"),
    )

    open_price = first_valid(
        info_value(info, "open", "regularMarketOpen"),
        fast.get("open"),
    )

    day_high = first_valid(
        info_value(info, "dayHigh", "regularMarketDayHigh"),
        fast.get("day_high"),
    )

    day_low = first_valid(
        info_value(info, "dayLow", "regularMarketDayLow"),
        fast.get("day_low"),
    )

    # Historical fallback.
    if history is not None and not history.empty:
        latest = history.iloc[-1]

        open_price = first_valid(
            open_price,
            number(latest.get("Open")),
        )

        day_high = first_valid(
            day_high,
            number(latest.get("High")),
        )

        day_low = first_valid(
            day_low,
            number(latest.get("Low")),
        )

        if previous_close is None and len(history) >= 2:
            previous_close = number(history.iloc[-2].get("Close"))

        if current_price is None:
            current_price = number(latest.get("Close"))

    # Fundamentals.
    pe = first_valid(
        info_value(info, "trailingPE"),
        info_value(info, "forwardPE"),
    )

    eps = first_valid(
        info_value(info, "trailingEps"),
        info_value(info, "forwardEps"),
    )

    week_high = info_value(info, "fiftyTwoWeekHigh")
    week_low = info_value(info, "fiftyTwoWeekLow")

    # 52-week fallback from history.
    if week_high is None or week_low is None:
        year_history = get_history(ticker, "1y")
        if year_history is not None and not year_history.empty:
            if week_high is None:
                try:
                    week_high = year_history["High"].max()
                except Exception:
                    pass
            if week_low is None:
                try:
                    week_low = year_history["Low"].min()
                except Exception:
                    pass

    change = None
    change_pct = None

    if current_price is not None and previous_close not in (None, 0):
        change = current_price - previous_close
        change_pct = change / previous_close * 100

    major, institutions, holder_errors = load_holders(ticker)
    errors.extend(holder_errors)

    company_name = info_value(info, "longName", "shortName")

    usable = any(
        value is not None
        for value in [
            current_price,
            company_name,
            open_price,
            day_high,
            day_low,
        ]
    )

    return {
        "symbol": symbol,
        "price": number(current_price),
        "previous_close": number(previous_close),
        "open": number(open_price),
        "high": number(day_high),
        "low": number(day_low),
        "change": number(change),
        "change_pct": number(change_pct),
        "pe": number(pe),
        "eps": number(eps),
        "week_high": number(week_high),
        "week_low": number(week_low),
        "earnings_date": get_earnings_date(ticker),
        "company_name": text(company_name),
        "sector": text(info_value(info, "sector")),
        "industry": text(info_value(info, "industry")),
        "country": text(info_value(info, "country")),
        "website": text(info_value(info, "website")),
        "employees": info_value(info, "fullTimeEmployees"),
        "currency": text(info_value(info, "currency")),
        "exchange": text(info_value(info, "fullExchangeName", "exchange")),
        "market_cap": number(info_value(info, "marketCap")),
        "summary": text(info_value(info, "longBusinessSummary")),
        "major_holders": parse_major_holders(major),
        "institutional": institutions,
        "errors": errors,
        "usable": usable,
        "loaded_at": datetime.now(),
    }


# ============================================================
# UI HELPERS
# ============================================================

def metric_card(label, value):
    # IMPORTANT: no indentation inside this HTML.
    html = (
        '<div class="metric">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📈 Stock Dashboard")
    st.caption("Market data, fundamentals and favorites.")

    st.divider()

    symbol_input = st.text_input(
        "Stock symbol",
        value=st.session_state.selected_symbol,
        placeholder="AAPL",
    ).strip().upper()

    if st.button(
        "Load Stock",
        type="primary",
        use_container_width=True,
    ):
        if symbol_input:
            st.session_state.selected_symbol = symbol_input
            st.rerun()

    st.markdown("### Favorites")

    if st.session_state.favorites:
        for favorite in st.session_state.favorites:
            if st.button(
                f"★ {favorite}",
                key=f"favorite_{favorite}",
                use_container_width=True,
            ):
                st.session_state.selected_symbol = favorite
                st.rerun()
    else:
        st.caption("No favorites yet.")

    st.divider()

    if st.button(
        "↻ Refresh Data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.caption("Data cache refreshes once per day, or immediately with Refresh Data.")

    st.divider()

    st.caption(
        "Source: Yahoo Finance via yfinance. "
        "Free market data can be delayed or temporarily unavailable."
    )


# ============================================================
# MAIN
# ============================================================

symbol = st.session_state.selected_symbol.upper().strip() or "AAPL"

st.markdown(
    '<h1 style="margin-bottom:0;">AI Stock Dashboard</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="color:#6b7280;margin-bottom:1.2rem;">'
    'Market overview, fundamentals and company information'
    '</div>',
    unsafe_allow_html=True,
)

with st.spinner(f"Loading {symbol}..."):
    data = load_stock(symbol, date.today().isoformat())

if not data["usable"]:
    st.error(
        f"Could not load usable data for {symbol}. "
        "Yahoo Finance may be rate-limiting or temporarily unavailable."
    )
    for error in data["errors"]:
        st.caption(error)
    st.stop()


# ============================================================
# HERO
# ============================================================

price = data["price"]
change_pct = data["change_pct"]

if change_pct is not None and change_pct > 0:
    change_class = "hero-change-up"
elif change_pct is not None and change_pct < 0:
    change_class = "hero-change-down"
else:
    change_class = "hero-change-flat"

change_line = (
    f'{money(data["change"])} ({percent(change_pct)})'
    if data["change"] is not None
    else "—"
)

hero_html = (
    '<div class="hero">'
    f'<div class="hero-symbol">{symbol}</div>'
    f'<div class="hero-name">{data["company_name"]}</div>'
    f'<div class="hero-price">{money(price)}</div>'
    f'<div class="{change_class}">{change_line}</div>'
    '</div>'
)

st.markdown(hero_html, unsafe_allow_html=True)


# ============================================================
# FAVORITE
# ============================================================

is_favorite = symbol in st.session_state.favorites

if is_favorite:
    if st.button(
        "★ Remove Favorite",
        use_container_width=True,
    ):
        st.session_state.favorites.remove(symbol)
        st.rerun()
else:
    if st.button(
        "☆ Add Favorite",
        type="primary",
        use_container_width=True,
    ):
        if symbol not in st.session_state.favorites:
            st.session_state.favorites.append(symbol)
        st.rerun()


# ============================================================
# MARKET SNAPSHOT
# ============================================================

st.markdown(
    '<div class="section-heading">Market Snapshot</div>',
    unsafe_allow_html=True,
)

cols = st.columns(4)

with cols[0]:
    metric_card("Previous Close", money(data["previous_close"]))

with cols[1]:
    metric_card("Open", money(data["open"]))

with cols[2]:
    metric_card("Day High", money(data["high"]))

with cols[3]:
    metric_card("Day Low", money(data["low"]))

cols = st.columns(4)

with cols[0]:
    metric_card("P/E Ratio", plain_number(data["pe"]))

with cols[1]:
    metric_card("EPS", money(data["eps"]))

with cols[2]:
    metric_card("52 Week High", money(data["week_high"]))

with cols[3]:
    metric_card("52 Week Low", money(data["week_low"]))


# ============================================================
# EARNINGS / MARKET CAP
# ============================================================

st.markdown(
    '<div class="section-heading">Key Information</div>',
    unsafe_allow_html=True,
)

cols = st.columns(3)

with cols[0]:
    metric_card(
        "Earnings Date",
        fmt_date(data["earnings_date"]),
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
# COMPANY DETAILS
# ============================================================

st.markdown(
    '<div class="section-heading">Company Details</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    with st.container(border=True):
        st.write("**Company Name**")
        st.write(data["company_name"])

        st.write("**Sector**")
        st.write(data["sector"])

        st.write("**Industry**")
        st.write(data["industry"])

        st.write("**Country**")
        st.write(data["country"])

with right:
    with st.container(border=True):
        st.write("**Employees**")
        st.write(integer(data["employees"]))

        st.write("**Currency**")
        st.write(data["currency"])

        st.write("**Exchange**")
        st.write(data["exchange"])

        if data["website"] != "—":
            st.link_button(
                "Open Company Website",
                data["website"],
                use_container_width=True,
            )
        else:
            st.write("**Website**")
            st.write("—")


# ============================================================
# BUSINESS DESCRIPTION
# ============================================================

if data["summary"] != "—":
    with st.expander("Business Description"):
        st.write(data["summary"])


# ============================================================
# HOLDERS
# ============================================================

st.markdown(
    '<div class="section-heading">Major Holders Breakdown</div>',
    unsafe_allow_html=True,
)

holders = data["major_holders"]

cols = st.columns(3)

with cols[0]:
    metric_card(
        "Insider Ownership",
        text(holders.get("insider")),
    )

with cols[1]:
    metric_card(
        "Institutional Ownership",
        text(holders.get("institution")),
    )

with cols[2]:
    metric_card(
        "Float",
        text(holders.get("float")),
    )


# ============================================================
# INSTITUTIONAL HOLDERS TABLE
# ============================================================

institutional = data["institutional"]

if isinstance(institutional, pd.DataFrame) and not institutional.empty:
    st.markdown(
        '<div class="section-heading">Institutional Holders</div>',
        unsafe_allow_html=True,
    )

    preferred = [
        "Holder",
        "Shares",
        "Date Reported",
        "% Out",
        "Value",
    ]

    columns = [
        column
        for column in preferred
        if column in institutional.columns
    ]

    display = institutional[columns] if columns else institutional

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FAVORITES
# ============================================================

st.markdown(
    '<div class="section-heading">Favorites</div>',
    unsafe_allow_html=True,
)

if st.session_state.favorites:
    fav_cols = st.columns(
        min(4, len(st.session_state.favorites))
    )

    for index, favorite in enumerate(st.session_state.favorites):
        with fav_cols[index % len(fav_cols)]:
            if st.button(
                f"★ {favorite}",
                key=f"main_favorite_{favorite}",
                use_container_width=True,
            ):
                st.session_state.selected_symbol = favorite
                st.rerun()
else:
    st.caption("Your favorite stocks will appear here.")


# ============================================================
# DATA INTEGRITY
# ============================================================

st.markdown(
    '<div class="notice">'
    '<strong>Data integrity:</strong> Missing values are shown as '
    '<strong>—</strong>. The app does not invent prices, EPS, P/E, '
    'company information or earnings dates. Yahoo Finance data can '
    'be delayed, incomplete, rate-limited or temporarily unavailable.'
    '</div>',
    unsafe_allow_html=True,
)

if data["errors"]:
    with st.expander("Non-fatal data-source notices"):
        for error in data["errors"]:
            st.warning(error)

st.markdown(
    '<div class="small-muted" style="text-align:center;margin-top:24px;">'
    f'Loaded {data["loaded_at"].strftime("%b %d, %Y at %I:%M:%S %p")} '
    '· Cache: 24 hours'
    '</div>',
    unsafe_allow_html=True,
)
