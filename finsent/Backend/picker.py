from flask import Blueprint, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

picker_bp = Blueprint('picker', __name__)

# Cache file path
CACHE_FILE = 'picker_cache.json'
CACHE_DURATION = 86400  # 24 hours in seconds

# Curated list of liquid, tradeable stocks (reduced from S&P 500 for performance)
# These are well-known, liquid stocks across various sectors
STOCK_UNIVERSE = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'CRM',
    'ADBE', 'ORCL', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'NOW', 'INTU', 'IBM', 'AMAT',
    'MU', 'LRCX', 'ADI', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'FTNT', 'PANW', 'CRWD',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'MDT', 'SYK', 'BSX', 'ZTS', 'CI',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'USB',
    'PNC', 'TFC', 'COF', 'CME', 'ICE', 'CB', 'MMC', 'AON', 'SPGI', 'MCO',
    # Consumer
    'WMT', 'HD', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'TJX', 'ROST',
    'DG', 'DLTR', 'YUM', 'CMG', 'DPZ', 'ORLY', 'AZO', 'ULTA', 'BBY', 'EBAY',
    # Communication
    'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'TTWO', 'WBD',
    # Industrials
    'CAT', 'DE', 'BA', 'HON', 'UPS', 'UNP', 'RTX', 'LMT', 'GE', 'MMM',
    'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'FAST', 'ODFL', 'URI', 'PWR', 'CARR',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'STLD', 'CF',
    # Real Estate
    'PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'DLR', 'O', 'WELL', 'AVB', 'EQR',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED',
    # Consumer Staples
    'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'EL', 'KMB', 'GIS', 'K',
    'MDLZ', 'HSY', 'STZ', 'KHC', 'SJM', 'CAG', 'CPB', 'HRL', 'MKC', 'TSN'
]


def clean_value(value):
    """Clean NaN, Inf, and None values"""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
    return value


def safe_round(value, decimals=2):
    """Safely round a value"""
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return round(float(cleaned), decimals)
    except:
        return None


def format_large_number(value):
    """Format large numbers with suffixes"""
    if value is None:
        return 'N/A'
    try:
        value = float(value)
    except:
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value / 1e6:.2f}M"
    else:
        return f"${value:,.0f}"


def format_volume(value):
    """Format volume numbers"""
    if value is None:
        return 'N/A'
    try:
        value = float(value)
    except:
        return 'N/A'
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:,.0f}"


def download_batch_data(tickers, period='1y'):
    """
    Download historical data for multiple tickers in a single batch request.
    This is MUCH faster than individual downloads.
    """
    try:
        # Download all tickers at once
        data = yf.download(
            tickers=tickers,
            period=period,
            interval='1d',
            progress=False,
            threads=True,
            group_by='ticker'
        )
        return data
    except Exception as e:
        print(f"Batch download error: {e}")
        return None


def download_index_data():
    """Download S&P 500 index data for RS calculation"""
    try:
        index_df = yf.download('^GSPC', period='1y', interval='1d', progress=False)
        if not index_df.empty:
            # Handle multi-index columns
            if isinstance(index_df.columns, pd.MultiIndex):
                index_df.columns = index_df.columns.get_level_values(0)
            return index_df
    except Exception as e:
        print(f"Index download error: {e}")
    return None


def get_ticker_info_batch(tickers):
    """Get company info for multiple tickers using threading"""
    info_dict = {}
    
    def fetch_info(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return ticker, {
                'longName': info.get('longName') or info.get('shortName', ticker),
                'shortName': info.get('shortName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap'),
                'volume': info.get('volume')
            }
        except:
            return ticker, None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_info, t): t for t in tickers}
        for future in as_completed(futures, timeout=60):
            try:
                ticker, info = future.result(timeout=5)
                if info:
                    info_dict[ticker] = info
            except:
                pass
    
    return info_dict


def calculate_minervini_picks(batch_data, index_df, tickers):
    """
    Calculate Minervini criteria for all tickers using batch data.
    This processes everything in memory without additional API calls.
    """
    if batch_data is None or batch_data.empty:
        return []
    
    # Calculate index return for RS rating
    if index_df is not None and not index_df.empty:
        try:
            index_close = index_df['Close']
            if isinstance(index_close, pd.DataFrame):
                index_close = index_close.iloc[:, 0]
            index_return = (float(index_close.iloc[-1]) / float(index_close.iloc[0])) - 1
        except:
            index_return = 0.15  # Default 15%
    else:
        index_return = 0.15
    
    qualifying_stocks = []
    
    for ticker in tickers:
        try:
            # Extract data for this ticker
            if len(tickers) == 1:
                ticker_data = batch_data.copy()
            else:
                try:
                    ticker_data = batch_data[ticker].copy()
                except KeyError:
                    continue
            
            # Handle multi-index columns
            if isinstance(ticker_data.columns, pd.MultiIndex):
                ticker_data.columns = ticker_data.columns.get_level_values(0)
            
            # Skip if not enough data
            if ticker_data.empty or len(ticker_data) < 200:
                continue
            
            # Get close prices - handle both Series and DataFrame
            close = ticker_data['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            
            if len(close) < 200:
                continue
            
            # Calculate moving averages
            sma_50 = close.rolling(window=50).mean()
            sma_150 = close.rolling(window=150).mean()
            sma_200 = close.rolling(window=200).mean()
            
            # Get current values
            current_close = clean_value(float(close.iloc[-1]))
            current_sma_50 = clean_value(float(sma_50.iloc[-1]))
            current_sma_150 = clean_value(float(sma_150.iloc[-1]))
            current_sma_200 = clean_value(float(sma_200.iloc[-1]))
            
            if any(v is None for v in [current_close, current_sma_50, current_sma_150, current_sma_200]):
                continue
            
            # Get high/low for 52-week calculations
            high = ticker_data['High']
            low = ticker_data['Low']
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]
            
            high_52_week = clean_value(float(high.tail(260).max()))
            low_52_week = clean_value(float(low.tail(260).min()))
            
            if high_52_week is None or low_52_week is None:
                continue
            
            # Calculate RS Rating
            try:
                stock_return = (current_close / float(close.iloc[0])) - 1
                if index_return != 0:
                    returns_multiple = stock_return / index_return
                else:
                    returns_multiple = 1.0
                
                # Convert to RS rating (0-100 scale)
                if returns_multiple >= 2.0:
                    rs_rating = 95
                elif returns_multiple >= 1.5:
                    rs_rating = 85
                elif returns_multiple >= 1.2:
                    rs_rating = 75
                elif returns_multiple >= 1.0:
                    rs_rating = 65
                elif returns_multiple >= 0.8:
                    rs_rating = 50
                elif returns_multiple >= 0.5:
                    rs_rating = 35
                else:
                    rs_rating = 20
            except:
                rs_rating = 50
            
            # SMA trend checks (get values from 20 days ago)
            try:
                sma_150_20_ago = clean_value(float(sma_150.iloc[-20])) if len(sma_150) >= 20 else None
                sma_200_20_ago = clean_value(float(sma_200.iloc[-20])) if len(sma_200) >= 20 else None
            except:
                sma_150_20_ago = None
                sma_200_20_ago = None
            
            # Check all 8 Minervini criteria
            criteria_results = []
            
            # 1. Current price > 150 MA > 200 MA
            c1 = current_close > current_sma_150 > current_sma_200
            criteria_results.append(c1)
            
            # 2. 150 MA trending up
            c2 = sma_150_20_ago is not None and current_sma_150 > sma_150_20_ago
            criteria_results.append(c2)
            
            # 3. 200 MA trending up (or flat)
            c3 = sma_200_20_ago is not None and current_sma_200 >= sma_200_20_ago
            criteria_results.append(c3)
            
            # 4. 50 MA > 150 MA > 200 MA
            c4 = current_sma_50 > current_sma_150 > current_sma_200
            criteria_results.append(c4)
            
            # 5. Current price > 50 MA
            c5 = current_close > current_sma_50
            criteria_results.append(c5)
            
            # 6. Current price >= 30% above 52-week low
            c6 = current_close >= (1.30 * low_52_week)
            criteria_results.append(c6)
            
            # 7. Current price within 25% of 52-week high
            c7 = current_close >= (0.75 * high_52_week)
            criteria_results.append(c7)
            
            # 8. RS Rating >= 70
            c8 = rs_rating >= 70
            criteria_results.append(c8)
            
            num_criteria_met = sum(criteria_results)
            
            # Must meet all 8 criteria
            if num_criteria_met == 8:
                # Calculate day change
                change_pct = None
                if len(close) >= 2:
                    prev_close = clean_value(float(close.iloc[-2]))
                    if prev_close and prev_close != 0:
                        change_pct = ((current_close - prev_close) / prev_close) * 100
                
                qualifying_stocks.append({
                    'ticker': ticker,
                    'price': current_close,
                    'change_pct': safe_round(change_pct, 2),
                    'rs_rating': rs_rating,
                    'sma_50': safe_round(current_sma_50, 2),
                    'sma_150': safe_round(current_sma_150, 2),
                    'sma_200': safe_round(current_sma_200, 2),
                    'low_52_week': safe_round(low_52_week, 2),
                    'high_52_week': safe_round(high_52_week, 2),
                    'criteria_met': f"{num_criteria_met}/8"
                })
                
        except Exception as e:
            # Skip stocks that error
            continue
    
    return qualifying_stocks


def run_minervini_screener():
    """
    Run the complete Minervini screening process using efficient batch operations.
    """
    print(f"Starting Minervini screener with {len(STOCK_UNIVERSE)} stocks...")
    start_time = time.time()
    
    # Step 1: Download index data
    print("Downloading index data...")
    index_df = download_index_data()
    
    # Step 2: Download all stock data in one batch
    print("Downloading stock data (batch)...")
    batch_data = download_batch_data(STOCK_UNIVERSE, period='1y')
    
    if batch_data is None or batch_data.empty:
        print("Failed to download batch data")
        return []
    
    # Step 3: Calculate Minervini criteria for all stocks
    print("Calculating Minervini criteria...")
    qualifying_stocks = calculate_minervini_picks(batch_data, index_df, STOCK_UNIVERSE)
    
    print(f"Found {len(qualifying_stocks)} stocks meeting all criteria")
    
    # Step 4: Get company info for qualifying stocks only (much fewer API calls)
    if qualifying_stocks:
        qualifying_tickers = [s['ticker'] for s in qualifying_stocks]
        print(f"Fetching info for {len(qualifying_tickers)} qualifying stocks...")
        info_dict = get_ticker_info_batch(qualifying_tickers)
        
        # Merge info into results
        for stock in qualifying_stocks:
            ticker = stock['ticker']
            info = info_dict.get(ticker, {})
            stock['company'] = info.get('longName') or info.get('shortName', ticker)
            stock['sector'] = info.get('sector', 'N/A')
            stock['industry'] = info.get('industry', 'N/A')
            stock['market_cap'] = info.get('marketCap')
            stock['volume'] = info.get('volume')
            stock['price_display'] = f"${stock['price']:.2f}" if stock['price'] else 'N/A'
            stock['market_cap_display'] = format_large_number(stock['market_cap'])
            stock['volume_display'] = format_volume(stock['volume'])
    
    # Sort by RS rating
    qualifying_stocks.sort(key=lambda x: x['rs_rating'], reverse=True)
    
    elapsed = time.time() - start_time
    print(f"Screener complete in {elapsed:.1f} seconds")
    
    return qualifying_stocks


def load_cache():
    """Load cached picks from file"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            age = (datetime.now() - cache_time).total_seconds()
            
            if age < CACHE_DURATION:
                return cache_data
    except Exception as e:
        print(f"Error loading cache: {e}")
    
    return None


def save_cache(data):
    """Save picks to cache file"""
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cache saved: {CACHE_FILE}")
    except Exception as e:
        print(f"Error saving cache: {e}")


def build_response(minervini_picks):
    """Build the API response from picks data"""
    # Calculate statistics
    avg_rs = np.mean([s['rs_rating'] for s in minervini_picks]) if minervini_picks else 0
    
    # Count sectors
    sector_counts = {}
    for stock in minervini_picks:
        sector = stock.get('sector', 'Unknown')
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    return {
        'timestamp': datetime.now().isoformat(),
        'next_update': (datetime.now() + timedelta(seconds=CACHE_DURATION)).isoformat(),
        'portfolios': {
            'minervini': {
                'name': 'Minervini Momentum',
                'description': 'Stocks meeting Mark Minervini\'s trend template criteria: strong uptrends, price above key moving averages, and superior relative strength.',
                'strategy': 'Growth momentum stocks with established uptrends',
                'count': len(minervini_picks),
                'stocks': minervini_picks,
                'statistics': {
                    'avg_rs_rating': safe_round(avg_rs, 1),
                    'sectors': sector_counts,
                    'total_stocks': len(minervini_picks)
                }
            }
        }
    }


@picker_bp.route('/api/picks', methods=['GET'])
def get_picks():
    """
    Main endpoint to get today's stock picks.
    Returns cached data if available and fresh, otherwise runs screener.
    """
    try:
        # Try to load from cache first
        cached = load_cache()
        
        if cached:
            print("Returning cached picks")
            return jsonify(cached['data'])
        
        # Run screener if no valid cache
        print("Running fresh screener...")
        minervini_picks = run_minervini_screener()
        
        response = build_response(minervini_picks)
        
        # Save to cache
        save_cache(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in get_picks: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@picker_bp.route('/api/picks/refresh', methods=['POST'])
def refresh_picks():
    """
    Force refresh the picks (clears cache and runs screener).
    """
    try:
        # Clear cache
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print("Cache cleared")
        
        # Run screener
        print("Running fresh screener (forced refresh)...")
        minervini_picks = run_minervini_screener()
        
        response = build_response(minervini_picks)
        
        # Save to cache
        save_cache(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Refresh failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Refresh failed: {str(e)}',
            'error_type': 'server_error'
        }), 500


@picker_bp.route('/api/picks/health', methods=['GET'])
def picks_health():
    """Health check endpoint"""
    cache_status = 'fresh' if load_cache() else 'stale'
    
    cache_age = None
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            cache_age = (datetime.now() - cache_time).total_seconds()
        except:
            pass
    
    return jsonify({
        'status': 'healthy',
        'service': 'stock_picker',
        'cache_status': cache_status,
        'cache_age_seconds': cache_age,
        'cache_duration': CACHE_DURATION,
        'stock_universe_size': len(STOCK_UNIVERSE)
    })
```

---

## 2. `/Backend/Procfile`
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 600 --worker-class gthread --max-requests 100 --max-requests-jitter 20 --graceful-timeout 300
```

---

## 3. `/Backend/requirements.txt`
```
Flask
flask-cors
feedparser
requests
beautifulsoup4
vaderSentiment
yfinance
gunicorn
pandas
numpy
