import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
        """Calculate technical indicators including EMA 20, 50, 200, RSI, ATR, and Volume metrics"""
        df = data.copy()
        
        # Exponential Moving Averages
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
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
        
        if len(df) < 100:
            return None
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                       'Returns', 'Returns_5d', 'Returns_10d']
        feature_cols = [col for col in df.columns if not any(exc in col for exc in exclude_cols)]
        feature_cols = [col for col in feature_cols if 'lag' in col or 'mean' in col or 
                       'std' in col or col in ['RSI', 'Price_vs_EMA20', 'Price_vs_EMA50', 
                                              'Price_volatility_10d', 'Price_volatility_20d', 'ATR']]
        
        if len(feature_cols) < 5:
            return None
        
        X = df[feature_cols].ffill().bfill()
        y = df['Close'].shift(-1)
        
        X = X[:-1]
        y = y[:-1]
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
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
    
    def generate_market_analysis(self, data, info, symbol):
        """Generate AI-powered market analysis"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        rsi = latest.get('RSI', 50)
        ema_20 = latest.get('EMA_20', latest['Close'])
        ema_50 = latest.get('EMA_50', latest['Close'])
        
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
        
        analysis = []
        
        if price_change_pct > 3:
            analysis.append(f"🚀 {symbol} shows exceptional bullish momentum with a {price_change_pct:.2f}% surge")
        elif price_change_pct > 1:
            analysis.append(f"🟢 {symbol} demonstrates strong upward movement (+{price_change_pct:.2f}%)")
        elif price_change_pct > 0:
            analysis.append(f"🟡 {symbol} shows modest gains (+{price_change_pct:.2f}%)")
        elif price_change_pct > -1:
            analysis.append(f"🟡 {symbol} experiences slight decline ({price_change_pct:.2f}%)")
        elif price_change_pct > -3:
            analysis.append(f"🔴 {symbol} shows moderate bearish pressure ({price_change_pct:.2f}%)")
        else:
            analysis.append(f"🔻 {symbol} faces significant selling pressure ({price_change_pct:.2f}%)")
        
        if rsi > 70:
            analysis.append(f"⚠️ RSI at {rsi:.1f} shows overbought territory - exercise caution")
        elif rsi < 30:
            analysis.append(f"💡 RSI at {rsi:.1f} suggests oversold conditions - potential buying opportunity")
        elif 40 <= rsi <= 60:
            analysis.append(f"⚖️ RSI at {rsi:.1f} indicates balanced momentum")
        else:
            analysis.append(f"📊 RSI at {rsi:.1f} shows {('bullish' if rsi > 50 else 'bearish')} bias")
        
        if latest['Close'] > ema_20 > ema_50:
            analysis.append("📈 Strong bullish alignment - price above both 20 and 50-day EMAs")
        elif latest['Close'] < ema_20 < ema_50:
            analysis.append("📉 Bearish trend confirmed - price below key exponential moving averages")
        else:
            analysis.append("➡️ Consolidation phase - awaiting directional breakout")
        
        if volume_ratio > 2:
            analysis.append("🔥 Exceptional volume surge confirms strong conviction")
        elif volume_ratio > 1.5:
            analysis.append("📊 High volume validates price movement")
        elif volume_ratio < 0.5:
            analysis.append("📊 Below-average volume suggests weak conviction")
        else:
            analysis.append("📊 Normal volume levels")
        
        return analysis

def create_performance_metrics(data, symbol):
    """Create performance metrics visualization"""
    data['Daily_Returns'] = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    
    total_return = data['Cumulative_Returns'].iloc[-1] * 100
    volatility = data['Daily_Returns'].std() * np.sqrt(252) * 100  
    sharpe_ratio = (data['Daily_Returns'].mean() * 252) / (data['Daily_Returns'].std() * np.sqrt(252))
    max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
    
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
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
    
    st.sidebar.markdown("---")
    
    show_prediction = st.sidebar.checkbox("🔮 ML Price Prediction", value=True)
    show_performance = st.sidebar.checkbox("📊 Performance Metrics", value=True)
    show_analysis = st.sidebar.checkbox("🧠 AI Market Analysis", value=True)
    
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
    
    with st.spinner("⚙️ Calculating technical metrics..."):
        data = analyzer.calculate_technical_indicators(data)
    
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest_price:,.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
        st.metric(
            label="📊 Volume",
            value=f"{volume:,.0f}",
            delta=f"{volume_change:+.1f}% vs 20d avg"
        )
    
    with col3:
        market_cap = info.get('marketCap', 0)
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
        if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            st.metric(label="⚡ RSI (14)", value=f"{rsi:.1f}")
        else:
            st.metric(label="⚡ RSI (14)", value="N/A")

    with col5:
        if 'ATR' in data.columns and not pd.isna(data['ATR'].iloc[-1]):
            atr = data['ATR'].iloc[-1]
            st.metric(label="📉 ATR (14)", value=f"${atr:,.2f}")
        else:
            st.metric(label="📉 ATR (14)", value="N/A")
    
    st.markdown("---")
    
    if show_performance:
        st.subheader("📊 Performance Analysis")
        create_performance_metrics(data, symbol)
    
    if show_prediction:
        st.subheader("🔮 Machine Learning Price Prediction")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.spinner("🤖 Training AI prediction model..."):
                model_info = analyzer.train_prediction_model(data)
            
            if model_info:
                prediction = analyzer.predict_next_price(model_info)
                current_price = data['Close'].iloc[-1]
                predicted_change = ((prediction - current_price) / current_price) * 100
                
                st.success("✅ Model trained successfully!")
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric(
                        label="🎯 Next Day Prediction",
                        value=f"${prediction:,.2f}",
                        delta=f"{predicted_change:+.2f}%"
                    )
                
                with pred_col2:
                    confidence = model_info['test_score']
                    confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric(
                        label="🎲 Model Confidence",
                        value=f"{confidence:.1%}",
                        delta=confidence_level
                    )
                
                st.info(f"📈 **Training Accuracy:** {model_info['train_score']:.1%} | **Test Accuracy:** {model_info['test_score']:.1%}")
            else:
                st.warning("⚠️ Insufficient data for reliable ML prediction. Need more historical data.")
        
        with col2:
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
    
    if show_analysis:
        st.subheader("🧠 AI-Powered Market Analysis")
        with st.spinner("🤖 Generating intelligent market insights..."):
            analysis = analyzer.generate_market_analysis(data, info, symbol)
        
        for i, insight in enumerate(analysis):
            if i == 0:
                if "🚀" in insight or "🟢" in insight:
                    st.success(insight)
                elif "🔴" in insight or "🔻" in insight:
                    st.error(insight)
                else:
                    st.warning(insight)
            else:
                st.info(insight)
    
    st.markdown("---")
    
    # 📋 Formatted Tabs including Major, Institutional & Top Mutual Fund Holders with comma formatting
    tab1, tab2, tab3 = st.tabs(["📋 Company Info & Holders", "📊 Raw Data", "🔧 Technical Indicators"])
    
    with tab1:
        if info:
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
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
                        "% of Shares Held by All Insider",
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
        st.write("### 📊 Recent Price Data")
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20).copy()
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        
        for col in ['Open', 'High', 'Low', 'Close']:
            display_data[col] = display_data[col].apply(lambda x: f"{x:,.2f}")
        display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_data, use_container_width=True)
        
        csv = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20).to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_stock_data.csv',
            mime='text/csv'
        )
    
    with tab3:
        st.write("### 🔧 Technical Indicators (Last 10 Days)")
        tech_columns = ['Close', 'EMA_20', 'EMA_50', 'EMA_200', 'RSI', 'ATR']
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
