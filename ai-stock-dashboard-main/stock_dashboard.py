import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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
        """Fetch stock data and holder breakdown info with error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            
            try:
                major_holders = stock.major_holders
            except:
                major_holders = None
                
            try:
                institutional_holders = stock.institutional_holders
            except:
                institutional_holders = None
                
            try:
                mutualfund_holders = stock.mutualfund_holders
            except:
                mutualfund_holders = None
                
            return data, info, major_holders, institutional_holders, mutualfund_holders
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None, None, None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators including EMA 20, 50, 200, ATR, and Volume metrics"""
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
        
        # Average True Range (ATR)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
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
                                              'Price_volatility_10d', 'Price_volatility_20d', 'ATR']]
        
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
            line=dict(color='#00ff88', width=2)
        )
    )
    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit App
def main():
    st.title("🚀 Professional AI Stock Market Dashboard")
    st.markdown("*Advanced financial metrics and machine learning predictions*")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    popular_stocks = {
        'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL', 
        'Amazon': 'AMZN', 'Tesla': 'TSLA', 'NVIDIA': 'NVDA',
        'Meta': 'META', 'Netflix': 'NFLX', 'AMD': 'AMD', 'Intel': 'INTC'
    }
    
    stock_choice = st.sidebar.selectbox(
        "🏢 Select Stock:",
        options=list(popular_stocks.keys()) + ['Custom'],
        index=0
    )
    
    if stock_choice == 'Custom':
        symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL", max_chars=10).upper()
    else:
        symbol = popular_stocks[stock_choice]
    
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
    
    analyzer = StockAnalyzer()
    
    with st.spinner(f"📡 Fetching live data for {symbol}..."):
        data, info, major_holders, institutional_holders, mutualfund_holders = analyzer.fetch_stock_data(symbol, period)
    
    if data is None or data.empty:
        st.error(f"❌ Could not fetch data for {symbol}. Please verify the symbol and try again.")
        st.info("💡 Try popular symbols like AAPL, MSFT, GOOGL, TSLA, etc.")
        return
    
    with st.spinner("⚙️ Calculating metrics..."):
        data = analyzer.calculate_technical_indicators(data)
    
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Robust price extraction with fallback to info if data Close is NaN
    latest_price = data['Close'].iloc[-1] if not data.empty and not pd.isna(data['Close'].iloc[-1]) else 0.0
    if latest_price == 0.0 and info:
        latest_price = info.get('regularMarketPrice', info.get('currentPrice', 0.0))

    if len(data) > 1 and not pd.isna(data['Close'].iloc[-2]):
        prev_price = data['Close'].iloc[-2]
    else:
        prev_price = data['Open'].iloc[-1] if not data.empty and 'Open' in data.columns and not pd.isna(data['Open'].iloc[-1]) else latest_price
        
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price > 0 else 0.0
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest_price:,.2f}" if latest_price > 0 else "N/A",
            delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)" if latest_price > 0 else "N/A"
        )
    
    with col2:
        volume = data['Volume'].iloc[-1] if not data.empty and not pd.isna(data['Volume'].iloc[-1]) else 0
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else (data['Volume'].mean() if not data.empty else 0)
        volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 and not pd.isna(avg_volume) else 0
        st.metric(
            label="📊 Volume",
            value=f"{volume:,.0f}" if volume > 0 else "N/A",
            delta=f"{volume_change:+.1f}% vs avg" if volume > 0 else "N/A"
        )
    
    with col3:
        market_cap = info.get('marketCap', 0) if info else 0
        if market_cap:
            if market_cap > 1e12:
                cap_display = f"${market_cap/1e12:,.2f}T"
            elif market_cap > 1e9:
                cap_display = f"${market_cap/1e9:,.1f}B"
            else:
                cap_display = f"${market_cap/1e6:,.0f}M"
            st.metric(label="🏢 Market Cap", value=cap_display)
        else:
            st.metric(label="🏢 Market Cap", value="N/A")
            
    with col4:
        ema_20 = data['EMA_20'].iloc[-1] if 'EMA_20' in data.columns and not pd.isna(data['EMA_20'].iloc[-1]) else 0
        st.metric(label="📈 EMA 20", value=f"${ema_20:,.2f}" if ema_20 > 0 else "N/A")

    with col5:
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]) else 0
        st.metric(label="📉 ATR (14)", value=f"${atr:,.2f}" if atr > 0 else "N/A")
    
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
                    confidence = model_info['test_score']
                    confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric(
                        label="🎲 Model Confidence",
                        value=f"{confidence:.1%}",
                        delta=confidence_level
                    )
                
                st.info(f"📈 **Training Accuracy:** {model_info['train_score']:.1%} | **Test Accuracy:** {model_info['test_score']:.1%}")
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
                    template='plotly_dark'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Company Info & Holders", "📊 Volume Comparison & Raw Data", "🔧 Technical Metrics"])
    
    with tab1:
        if info:
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
                insider_pct = info.get('heldPercentInsiders')
                inst_pct = info.get('heldPercentInstitutions')
                inst_float_pct = info.get('heldPercentInstitutions')
                inst_count = info.get('institutionsCount')
                
                breakdown_data = {
                    "Value": [
                        f"{insider_pct*100:.2f}%" if insider_pct is not None else "N/A",
                        f"{inst_pct*100:.2f}%" if inst_pct is not None else "N/A",
                        f"{inst_float_pct*100:.2f}%" if inst_float_pct is not None else "N/A",
                        f"{inst_count:,}" if inst_count is not None and isinstance(inst_count, (int, float)) else "N/A"
                    ],
                    "Breakdown Metric": [
                        "% of Shares Held by All Insiders",
                        "% of Shares Held by Institutions",
                        "% of Float Held by Institutions",
                        "Number of Institutions Holding Shares"
                    ]
                }
                st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)
            
            st.markdown("---")
            st.write("### 🏛️ Top Institutional Holders & Live Comparison")
            
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
                    st.dataframe(comp_df, use_container_width=True)
                else:
                    st.info("💡 First snapshot for this stock cached in session state. Refresh or update to see comparative deltas.")
                
                st.session_state['hist_holders'][symbol] = current_raw_dict
                
                st.write("📋 **Current Live Institutional Holders Data:**")
                st.dataframe(formatted_inst, use_container_width=True)
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
                
                st.dataframe(formatted_mf, use_container_width=True)
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
                temp_hist = yf.Ticker(symbol).history(period=h_key)
                if not temp_hist.empty:
                    vol_comparison_data[h_label] = temp_hist['Volume'].mean()
                else:
                    vol_comparison_data[h_label] = 0
            except:
                vol_comparison_data[h_label] = 0
                
        vol_comp_df = pd.DataFrame(list(vol_comparison_data.items()), columns=['Time Horizon', 'Average Volume'])
        
        fig_multi_vol = px.bar(
            vol_comp_df,
            x='Time Horizon',
            y='Average Volume',
            text_auto=',.2s',
            title=f"{symbol} - Average Volume Across Time Horizons",
            template='plotly_dark',
            color='Average Volume',
            color_continuous_scale='Greens'
        )
        fig_multi_vol.update_layout(height=400)
        st.plotly_chart(fig_multi_vol, use_container_width=True)
        
        st.markdown("---")
        st.write("### 📊 Recent Price & Volume Table")
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Volume_SMA']].tail(20).copy()
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        
        for col in ['Open', 'High', 'Low', 'Close']:
            display_data[col] = display_data[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
        display_data['Volume_SMA'] = display_data['Volume_SMA'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
        
        st.dataframe(display_data, use_container_width=True)
        
        csv = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20).to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_stock_data.csv',
            mime='text/csv'
        )
    
    with tab3:
        st.write("### 🔧 Technical Indicators (EMA 20, EMA 50, EMA 200, ATR)")
        tech_columns = ['Close', 'EMA_20', 'EMA_50', 'EMA_200', 'ATR']
        available_columns = [col for col in tech_columns if col in data.columns]
        
        if available_columns:
            tech_data = data[available_columns].tail(10).copy()
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            
            for col in available_columns:
                tech_data[col] = tech_data[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
                
            st.dataframe(tech_data, use_container_width=True)
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
