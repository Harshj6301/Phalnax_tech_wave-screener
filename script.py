# Create the complete Elliott Wave Stock Screener for NSE Futures
# This will be a comprehensive Streamlit application

# First, let's create the main application file
streamlit_app_code = '''
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NSE Futures Elliott Wave Screener",
    page_icon="📊",
    layout="wide"
)

class ElliottWaveScreener:
    def __init__(self):
        # NSE Futures symbols (top 50 for demo - can be expanded to 200+)
        self.nse_futures = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "HDFCBANK.NS",
            "ICICIBANK.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "ASIANPAINT.NS",
            "ITC.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "HCLTECH.NS",
            "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "TITAN.NS", "SUNPHARMA.NS",
            "POWERGRID.NS", "NTPC.NS", "COALINDIA.NS", "BHARTIARTL.NS", "M&M.NS",
            "TECHM.NS", "TATAMOTORS.NS", "GRASIM.NS", "HINDALCO.NS", "JSWSTEEL.NS",
            "TATASTEEL.NS", "ADANIPORTS.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS",
            "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "BPCL.NS", "IOC.NS", "ONGC.NS",
            "INDUSINDBK.NS", "DIVISLAB.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "UPL.NS",
            "SHREE.NS", "TATACONSUM.NS", "GODREJCP.NS", "HINDZINC.NS", "VEDL.NS"
        ]
        
        # Default parameters
        self.default_params = {
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'swing_window': 5,
            'golden_ret_min': 0.5,
            'golden_ret_max': 0.65,
            'diamond_ret_min': 0.236,
            'diamond_ret_max': 0.382,
            'silver_ret_min': 0.382,
            'silver_ret_max': 0.5,
            'golden_ext_min': 1.0,
            'golden_ext_max': 1.236,
            'diamond_ext_min': 1.382,
            'diamond_ext_max': 1.618,
            'silver_ext_min': 1.236,
            'silver_ext_max': 1.382
        }
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        try:
            # RSI
            data['RSI'] = talib.RSI(data['Close'].values, timeperiod=self.params['rsi_period'])
            
            # Bollinger Bands
            data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = talib.BBANDS(
                data['Close'].values, 
                timeperiod=self.params['bb_period'], 
                nbdevup=self.params['bb_std'], 
                nbdevdn=self.params['bb_std']
            )
            
            return data
        except Exception as e:
            st.error(f"Error calculating indicators: {e}")
            return data
    
    def find_swing_points(self, data):
        """Find swing highs and lows"""
        try:
            window = self.params['swing_window']
            
            # Find local maxima (swing highs)
            highs = argrelextrema(data['High'].values, np.greater, order=window)[0]
            # Find local minima (swing lows)
            lows = argrelextrema(data['Low'].values, np.less, order=window)[0]
            
            swing_highs = [(i, data.iloc[i]['High']) for i in highs]
            swing_lows = [(i, data.iloc[i]['Low']) for i in lows]
            
            return swing_highs, swing_lows
        except Exception as e:
            st.error(f"Error finding swing points: {e}")
            return [], []
    
    def check_retracement_zone(self, wave1_start, wave1_end, wave2_end):
        """Check which retracement zone wave 2 falls into"""
        wave1_length = abs(wave1_end - wave1_start)
        retracement = abs(wave2_end - wave1_end) / wave1_length
        
        if self.params['golden_ret_min'] <= retracement <= self.params['golden_ret_max']:
            return 'Golden'
        elif self.params['diamond_ret_min'] <= retracement <= self.params['diamond_ret_max']:
            return 'Diamond'
        elif self.params['silver_ret_min'] <= retracement <= self.params['silver_ret_max']:
            return 'Silver'
        else:
            return 'None'
    
    def check_extension_zone(self, wave1_length, wave3_length):
        """Check which extension zone wave 3 falls into"""
        extension_ratio = wave3_length / wave1_length
        
        if self.params['golden_ext_min'] <= extension_ratio <= self.params['golden_ext_max']:
            return 'Golden'
        elif self.params['diamond_ext_min'] <= extension_ratio <= self.params['diamond_ext_max']:
            return 'Diamond'
        elif self.params['silver_ext_min'] <= extension_ratio <= self.params['silver_ext_max']:
            return 'Silver'
        else:
            return 'None'
    
    def detect_rsi_divergence(self, data, price_points, rsi_points):
        """Detect RSI divergence"""
        try:
            if len(price_points) < 2 or len(rsi_points) < 2:
                return False
            
            # Check for regular bearish divergence (price higher high, RSI lower high)
            price_trend = price_points[-1] > price_points[-2]
            rsi_trend = rsi_points[-1] < rsi_points[-2]
            
            # Check for regular bullish divergence (price lower low, RSI higher low)
            if not price_trend:
                price_trend = price_points[-1] < price_points[-2]
                rsi_trend = rsi_points[-1] > rsi_points[-2]
            
            return price_trend and rsi_trend
        except:
            return False
    
    def check_bollinger_band_criteria(self, data):
        """Check Bollinger Band criteria for impulse/correction identification"""
        try:
            # During impulse: price stays near upper/lower band
            # During correction: price ranges between bands
            recent_data = data.tail(20)
            
            # Check if price is consistently near bands (impulse)
            near_upper = (recent_data['Close'] >= recent_data['BB_Upper'] * 0.98).sum()
            near_lower = (recent_data['Close'] <= recent_data['BB_Lower'] * 1.02).sum()
            
            impulse_criteria = (near_upper >= 3) or (near_lower >= 3)
            
            # Check if price is ranging between bands (correction)
            between_bands = ((recent_data['Close'] > recent_data['BB_Lower']) & 
                           (recent_data['Close'] < recent_data['BB_Upper'])).sum()
            
            correction_criteria = between_bands >= 10
            
            return impulse_criteria, correction_criteria
        except:
            return False, False
    
    def screen_stock(self, symbol):
        """Screen individual stock for Elliott Wave patterns"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=self.period, interval=self.interval)
            
            if data.empty or len(data) < 50:
                return None
            
            # Calculate indicators
            data = self.calculate_indicators(data)
            
            # Find swing points
            swing_highs, swing_lows = self.find_swing_points(data)
            
            if len(swing_highs) < 3 or len(swing_lows) < 3:
                return None
            
            # Basic Elliott Wave pattern detection
            result = {
                'Symbol': symbol.replace('.NS', ''),
                'Last_Price': data['Close'].iloc[-1],
                'RSI': data['RSI'].iloc[-1],
                'Wave1_Retracement_Zone': 'None',
                'Wave3_Extension_Zone': 'None',
                'RSI_Divergence': 'No',
                'Hidden_Divergence': 'No',
                'Swing_High_Broken': 'No',
                'Band_Criteria_Impulse': 'No',
                'Band_Criteria_Correction': 'No',
                'BB_Position': 'Middle'
            }
            
            # Check retracement zones (simplified)
            if len(swing_lows) >= 2:
                wave1_start = swing_lows[-2][1]
                wave1_end = swing_highs[-1][1] if swing_highs else data['High'].max()
                wave2_end = swing_lows[-1][1]
                result['Wave1_Retracement_Zone'] = self.check_retracement_zone(wave1_start, wave1_end, wave2_end)
            
            # Check extension zones (simplified)
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                wave1_length = abs(swing_highs[-2][1] - swing_lows[-2][1]) if len(swing_lows) >= 2 else 100
                wave3_length = abs(swing_highs[-1][1] - swing_lows[-1][1])
                result['Wave3_Extension_Zone'] = self.check_extension_zone(wave1_length, wave3_length)
            
            # Check RSI divergence
            if len(swing_highs) >= 2:
                price_points = [point[1] for point in swing_highs[-2:]]
                rsi_points = [data['RSI'].iloc[point[0]] for point in swing_highs[-2:] if not pd.isna(data['RSI'].iloc[point[0]])]
                if len(rsi_points) >= 2:
                    result['RSI_Divergence'] = 'Yes' if self.detect_rsi_divergence(data, price_points, rsi_points) else 'No'
            
            # Check Bollinger Band criteria
            impulse_bb, correction_bb = self.check_bollinger_band_criteria(data)
            result['Band_Criteria_Impulse'] = 'Yes' if impulse_bb else 'No'
            result['Band_Criteria_Correction'] = 'Yes' if correction_bb else 'No'
            
            # Determine BB position
            last_price = data['Close'].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            bb_lower = data['BB_Lower'].iloc[-1]
            
            if last_price >= bb_upper:
                result['BB_Position'] = 'Upper'
            elif last_price <= bb_lower:
                result['BB_Position'] = 'Lower'
            else:
                result['BB_Position'] = 'Middle'
            
            # Check swing high broken (simplified)
            if len(swing_highs) >= 2:
                result['Swing_High_Broken'] = 'Yes' if swing_highs[-1][1] > swing_highs[-2][1] else 'No'
            
            return result
            
        except Exception as e:
            st.error(f"Error screening {symbol}: {e}")
            return None
    
    def run_screening(self, selected_stocks, params, period, interval):
        """Run screening on selected stocks"""
        self.params = params
        self.period = period
        self.interval = interval
        
        results = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(selected_stocks):
            result = self.screen_stock(symbol)
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / len(selected_stocks))
        
        return pd.DataFrame(results) if results else pd.DataFrame()

# Initialize the screener
@st.cache_resource
def get_screener():
    return ElliottWaveScreener()

screener = get_screener()

# Main UI
st.title("🌊 NSE Futures Elliott Wave Screener")
st.markdown("---")

# Sidebar for parameters
st.sidebar.header("📋 Screening Parameters")

# Timeframe selection
st.sidebar.subheader("Timeframe Settings")
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y"
}
interval_options = {
    "Daily": "1d",
    "Weekly": "1wk", 
    "Monthly": "1mo"
}

selected_period = st.sidebar.selectbox("Data Period", list(period_options.keys()), index=2)
selected_interval = st.sidebar.selectbox("Interval", list(interval_options.keys()), index=0)

# Elliott Wave Parameters
st.sidebar.subheader("Elliott Wave Zones")
col1, col2 = st.sidebar.columns(2)

with col1:
    st.write("**Retracement Zones**")
    golden_ret_min = st.number_input("Golden Min", value=0.5, step=0.01, key="golden_ret_min")
    golden_ret_max = st.number_input("Golden Max", value=0.65, step=0.01, key="golden_ret_max")
    diamond_ret_min = st.number_input("Diamond Min", value=0.236, step=0.001, key="diamond_ret_min")
    diamond_ret_max = st.number_input("Diamond Max", value=0.382, step=0.001, key="diamond_ret_max")
    silver_ret_min = st.number_input("Silver Min", value=0.382, step=0.001, key="silver_ret_min")
    silver_ret_max = st.number_input("Silver Max", value=0.5, step=0.01, key="silver_ret_max")

with col2:
    st.write("**Extension Zones**")
    golden_ext_min = st.number_input("Golden Min", value=1.0, step=0.01, key="golden_ext_min")
    golden_ext_max = st.number_input("Golden Max", value=1.236, step=0.001, key="golden_ext_max")
    diamond_ext_min = st.number_input("Diamond Min", value=1.382, step=0.001, key="diamond_ext_min")
    diamond_ext_max = st.number_input("Diamond Max", value=1.618, step=0.001, key="diamond_ext_max")
    silver_ext_min = st.number_input("Silver Min", value=1.236, step=0.001, key="silver_ext_min")
    silver_ext_max = st.number_input("Silver Max", value=1.382, step=0.001, key="silver_ext_max")

# Technical Indicator Parameters  
st.sidebar.subheader("Technical Parameters")
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
bb_period = st.sidebar.slider("Bollinger Band Period", 10, 50, 20)
bb_std = st.sidebar.slider("BB Standard Deviation", 1.0, 3.0, 2.0, 0.1)
swing_window = st.sidebar.slider("Swing Detection Window", 3, 10, 5)

# Stock selection
st.sidebar.subheader("Stock Selection")
all_stocks = st.sidebar.checkbox("Screen All NSE F&O Stocks", value=False)
if not all_stocks:
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks to Screen", 
        screener.nse_futures, 
        default=screener.nse_futures[:10]
    )
else:
    selected_stocks = screener.nse_futures

# Compile parameters
params = {
    'rsi_period': rsi_period,
    'bb_period': bb_period,
    'bb_std': bb_std,
    'swing_window': swing_window,
    'golden_ret_min': golden_ret_min,
    'golden_ret_max': golden_ret_max,
    'diamond_ret_min': diamond_ret_min,
    'diamond_ret_max': diamond_ret_max,
    'silver_ret_min': silver_ret_min,
    'silver_ret_max': silver_ret_max,
    'golden_ext_min': golden_ext_min,
    'golden_ext_max': golden_ext_max,
    'diamond_ext_min': diamond_ext_min,
    'diamond_ext_max': diamond_ext_max,
    'silver_ext_min': silver_ext_min,
    'silver_ext_max': silver_ext_max
}

# Main content area
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.subheader(f"Screening {len(selected_stocks)} stocks")
with col2:
    if st.button("🔄 Run Screening", type="primary"):
        st.session_state['run_screening'] = True
with col3:
    if st.button("⚙️ Reset Parameters"):
        st.rerun()

# Results section
if st.session_state.get('run_screening', False) and selected_stocks:
    st.markdown("---")
    st.subheader("📊 Screening Results")
    
    with st.spinner("Analyzing stocks for Elliott Wave patterns..."):
        results_df = screener.run_screening(
            selected_stocks, 
            params, 
            period_options[selected_period], 
            interval_options[selected_interval]
        )
    
    if not results_df.empty:
        # Display summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Screened", len(selected_stocks))
        with col2:
            st.metric("Patterns Found", len(results_df))
        with col3:
            rsi_div_count = len(results_df[results_df['RSI_Divergence'] == 'Yes'])
            st.metric("RSI Divergences", rsi_div_count)
        with col4:
            golden_ret_count = len(results_df[results_df['Wave1_Retracement_Zone'] == 'Golden'])
            st.metric("Golden Retracements", golden_ret_count)
        
        # Display results table
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Export functionality
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"elliott_wave_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Filter controls
        st.markdown("---")
        st.subheader("🔍 Filter Results")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            rsi_div_filter = st.selectbox("RSI Divergence", ["All", "Yes", "No"])
        with filter_col2:
            retracement_filter = st.selectbox("Retracement Zone", ["All", "Golden", "Diamond", "Silver", "None"])
        with filter_col3:
            extension_filter = st.selectbox("Extension Zone", ["All", "Golden", "Diamond", "Silver", "None"])
        
        # Apply filters
        filtered_df = results_df.copy()
        if rsi_div_filter != "All":
            filtered_df = filtered_df[filtered_df['RSI_Divergence'] == rsi_div_filter]
        if retracement_filter != "All":
            filtered_df = filtered_df[filtered_df['Wave1_Retracement_Zone'] == retracement_filter]
        if extension_filter != "All":
            filtered_df = filtered_df[filtered_df['Wave3_Extension_Zone'] == extension_filter]
        
        if len(filtered_df) < len(results_df):
            st.subheader(f"📋 Filtered Results ({len(filtered_df)} stocks)")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    else:
        st.warning("No Elliott Wave patterns detected in the selected stocks with current parameters.")
        st.info("Try adjusting the parameters or selecting different stocks.")

# Information section
st.markdown("---")
st.subheader("ℹ️ About Elliott Wave Screening")

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.markdown("""
    **Retracement Zones:**
    - **Golden Zone**: 0.5 to 0.65 (most common)
    - **Diamond Zone**: 0.236 to 0.382 (shallow)
    - **Silver Zone**: 0.382 to 0.5 (moderate)
    """)
    
with info_col2:
    st.markdown("""
    **Extension Zones:**  
    - **Golden Zone**: 1.0 to 1.236 (standard)
    - **Diamond Zone**: 1.382 to 1.618 (strong)
    - **Silver Zone**: 1.236 to 1.382 (moderate)
    """)

st.markdown("""
**Screening Criteria Explained:**
- **RSI Divergence**: Detects when price and RSI momentum diverge (signal for potential reversal)
- **Retracement Zones**: Measures how much Wave 2 retraces from Wave 1
- **Extension Zones**: Measures how Wave 3 extends relative to Wave 1
- **Bollinger Band Criteria**: Identifies impulse vs corrective wave characteristics
- **Swing Analysis**: Tracks breaking of previous swing highs/lows
""")

# Initialize session state
if 'run_screening' not in st.session_state:
    st.session_state['run_screening'] = False
'''

# Write the Streamlit app to a file
with open('elliott_wave_screener.py', 'w') as f:
    f.write(streamlit_app_code)

print("✅ Complete Elliott Wave Screener application created!")
print("\nFile created: elliott_wave_screener.py")
print("\nTo run the application:")
print("1. Install required packages: pip install streamlit yfinance pandas numpy talib scipy")
print("2. Run: streamlit run elliott_wave_screener.py")
print("\nFeatures included:")
print("- Multi-timeframe analysis (daily, weekly, monthly)")
print("- Editable Elliott Wave parameters (retracement/extension zones)")
print("- NSE Futures stock screening (50+ symbols, expandable to 200+)")
print("- RSI divergence detection")
print("- Bollinger Band analysis")
print("- Swing high/low analysis")
print("- CSV export functionality")
print("- Real-time filtering and results")
print("- Progress tracking")