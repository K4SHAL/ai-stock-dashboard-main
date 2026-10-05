
import warnings
from datetime import date, datetime

import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

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
# CLEAN ORANGE / WHITE THEME
# ============================================================

st.markdown(
    """
<style>
:root {
    --orange: #ff7900;
    --orange-dark: #e76600;
    --orange-soft: #fff5eb;
    --border: #e8e8e8;
    --text: #171717;
    --muted: #737373;
    --green: #168447;
    --red: #c93434;
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
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    letter-spacing: -0.035em;
}

.hero {
    background: linear-gradient(135deg, #fff7ef 0%, #ffffff 78%);
    border: 1px solid #f1dcc8;
    border-radius: 22px;
    padding: 28px;
    margin: 18px 0 20px;
    box-shadow: 0 8px 28px rgba(0,0,0,.035);
}

.hero-symbol {
    font-size: 2.15rem;
    font-weight: 850;
    letter-spacing: -.045em;
}

.hero-name {
    color: var(--muted);
    font-size: .98rem;
    margin-top: 3px;
}

.hero-price {
    font-size: 3rem;
    font-weight: 850;
    letter-spacing: -.055em;
    margin-top: 14px;
}

.hero-up {
    color: var(--green);
    font-weight: 750;
    font-size: 1.05rem;
}

.hero-down {
    color: var(--red);
    font-weight: 750;
