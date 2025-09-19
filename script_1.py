# Create a requirements.txt file for easy installation
requirements_content = '''streamlit==1.28.0
yfinance==0.2.21
pandas==2.0.3
numpy==1.24.3
TA-Lib==0.4.25
scipy==1.11.1
plotly==5.15.0
'''

# Write requirements file
with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

# Create a setup guide
setup_guide = '''# Elliott Wave Stock Screener Setup Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Install TA-Lib (Required for technical indicators)

#### On Windows:
```bash
# Download appropriate wheel file from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Then install:
pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl  # Replace with your Python version
```

#### On macOS:
```bash
brew install ta-lib
pip install TA-Lib
```

#### On Linux:
```bash
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

### 2. Install Other Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run elliott_wave_screener.py
```

## Application Features

### Core Functionality
- **Multi-timeframe Analysis**: Daily, Weekly, Monthly intervals
- **Elliott Wave Detection**: Automated pattern recognition for impulse waves
- **Technical Indicators**: RSI, Bollinger Bands integration
- **NSE F&O Stocks**: Pre-loaded with 50+ symbols, expandable to 200+

### Screening Criteria
- **Retracement Zones**: Golden (0.5-0.65), Diamond (0.236-0.382), Silver (0.382-0.5)
- **Extension Zones**: Golden (1-1.236), Diamond (1.382-1.618), Silver (1.236-1.382)
- **RSI Divergence**: Regular and hidden divergence detection
- **Bollinger Band Analysis**: Impulse vs correction identification
- **Swing Analysis**: High/low breakout detection

### User Interface
- **Parameter Editing**: Real-time adjustment of all Elliott Wave parameters
- **Stock Selection**: Choose specific stocks or screen all F&O stocks
- **Results Filtering**: Filter by divergence, retracement zones, extension zones
- **Data Export**: CSV download functionality
- **Progress Tracking**: Real-time screening progress

### Data Source
- **yfinance**: Real-time and historical data from Yahoo Finance
- **NSE Symbols**: Automatic .NS suffix handling for Indian stocks

## Usage Instructions

### 1. Parameter Configuration
- Use sidebar to adjust Elliott Wave zones, RSI periods, Bollinger Band settings
- Modify swing detection window for sensitivity adjustment

### 2. Stock Selection
- Choose individual stocks or select "Screen All NSE F&O Stocks"
- Default includes top 50 F&O stocks

### 3. Timeframe Selection
- Select data period (1 month to 2 years)
- Choose interval (daily, weekly, monthly)

### 4. Run Screening
- Click "Run Screening" to start analysis
- Monitor progress bar for completion status

### 5. Analyze Results
- Review tabular results with all screening criteria
- Use filters to narrow down specific patterns
- Export results as CSV for further analysis

## Troubleshooting

### Common Issues
1. **TA-Lib Installation**: Most common issue. Follow OS-specific installation steps above
2. **Data Fetch Errors**: Check internet connection and symbol validity
3. **Memory Issues**: Reduce number of stocks or shorter time periods for large datasets

### Performance Tips
- Start with smaller stock selections for initial testing
- Use daily intervals for faster processing
- Longer periods provide more reliable Elliott Wave patterns

## Extending the Application

### Adding More Stocks
Edit the `nse_futures` list in the `ElliottWaveScreener` class to include additional symbols.

### Custom Indicators
Add new technical indicators in the `calculate_indicators` method.

### Additional Screening Criteria
Implement new criteria functions and add columns to the results DataFrame.
'''

# Write setup guide
with open('README.md', 'w') as f:
    f.write(setup_guide)

print("📋 Additional files created:")
print("- requirements.txt (dependency list)")
print("- README.md (comprehensive setup guide)")
print("\n🚀 Your Elliott Wave Screener is ready!")
print("\nQuick start:")
print("1. pip install -r requirements.txt")
print("2. streamlit run elliott_wave_screener.py")