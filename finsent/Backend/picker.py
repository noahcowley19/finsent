from flask import Blueprint, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import math
import time

picker_bp = Blueprint('picker', __name__)

# Cache file path
CACHE_FILE = 'picker_cache.json'
CACHE_DURATION = 86400  # 24 hours in seconds


def normalize_dataframe(df):
    """Normalize DataFrame columns from yfinance to handle multi-index"""
    if df.empty:
        return df

    # If columns are multi-indexed, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def clean_value(value):
    """Clean NaN, Inf, and None values"""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def safe_round(value, decimals=2):
    """Safely round a value"""
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return round(cleaned, decimals)
    except:
        return None


def get_sp500_tickers():
    """
    Get S&P 500 tickers from Wikipedia.
    Fallback to a subset if download fails.
    """
    try:
        # Try to get from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Clean tickers
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except:
        # Fallback to a curated list of major stocks if Wikipedia fails
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'MA', 'HD', 'CVX', 'MRK',
            'ABBV', 'PEP', 'COST', 'AVGO', 'KO', 'LLY', 'WMT', 'MCD', 'CSCO',
            'TMO', 'ABT', 'ACN', 'ORCL', 'DIS', 'VZ', 'ADBE', 'NKE', 'CMCSA',
            'PFE', 'NFLX', 'DHR', 'CRM', 'INTC', 'AMD', 'TXN', 'PM', 'NEE',
            'UPS', 'RTX', 'QCOM', 'HON', 'UNP', 'IBM', 'SBUX', 'INTU', 'BA',
            'CAT', 'GE', 'LOW', 'AMGN', 'ELV', 'SPGI', 'DE', 'GS', 'BLK',
            'AMAT', 'AXP', 'BKNG', 'LMT', 'SYK', 'MDT', 'GILD', 'ADI', 'PLD',
            'TJX', 'CVS', 'MMC', 'AMT', 'VRTX', 'CI', 'ISRG', 'ZTS', 'ADP',
            'REGN', 'CB', 'MO', 'SLB', 'SO', 'CME', 'NOW', 'DUK', 'PGR', 'BDX',
            'TMUS', 'EOG', 'ITW', 'CSX', 'WM', 'CL', 'HUM', 'USB', 'BSX', 'MDLZ'
        ]


def calculate_rs_rating(ticker_returns, index_return):
    """
    Calculate Relative Strength rating (0-100 scale).
    Higher is better - shows stock outperformance vs index.
    """
    if index_return == 0:
        return 50  # Neutral if no index movement
    
    returns_multiple = ticker_returns / index_return
    # Normalize to 0-100 scale
    # RS > 1.5 = 90+, RS > 1.0 = 70+, RS < 0.5 = low
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
    
    return rs_rating


def check_minervini_criteria(ticker, rs_rating):
    """
    Check if a stock meets Mark Minervini's trend template criteria.
    
    Criteria:
    1. Current price > 150-day MA > 200-day MA
    2. 150-day MA is trending up (> 150-day MA from 20 days ago)
    3. 200-day MA is trending up for at least 1 month
    4. 50-day MA > 150-day MA > 200-day MA
    5. Current price > 50-day MA
    6. Current price >= 30% above 52-week low
    7. Current price within 25% of 52-week high (at least 75% of high)
    8. RS Rating >= 70 (strong relative strength)
    """
    try:
        # Download 1 year of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df = normalize_dataframe(df)

        if df.empty or len(df) < 200:
            return None, "Insufficient data"

        # Calculate moving averages - use 'Close' for all to ensure consistency
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Get current values
        current_close = clean_value(df['Close'].iloc[-1])
        sma_50 = clean_value(df['SMA_50'].iloc[-1])
        sma_150 = clean_value(df['SMA_150'].iloc[-1])
        sma_200 = clean_value(df['SMA_200'].iloc[-1])
        
        if None in [current_close, sma_50, sma_150, sma_200]:
            return None, "Missing MA data"
        
        # 52-week high/low
        low_52_week = clean_value(df['Low'].tail(260).min())
        high_52_week = clean_value(df['High'].tail(260).max())
        
        if None in [low_52_week, high_52_week]:
            return None, "Missing 52-week data"
        
        # SMA trend checks
        sma_150_20_days_ago = clean_value(df['SMA_150'].iloc[-20]) if len(df) >= 20 else None
        sma_200_20_days_ago = clean_value(df['SMA_200'].iloc[-20]) if len(df) >= 20 else None
        
        # Criteria checks
        criteria_met = []
        total_criteria = 8
        
        # 1. Current price > 150 MA > 200 MA
        c1 = current_close > sma_150 > sma_200
        criteria_met.append(c1)
        
        # 2. 150 MA trending up
        c2 = sma_150_20_days_ago and sma_150 > sma_150_20_days_ago
        criteria_met.append(c2)
        
        # 3. 200 MA trending up
        c3 = sma_200_20_days_ago and sma_200 >= sma_200_20_days_ago
        criteria_met.append(c3)
        
        # 4. 50 MA > 150 MA > 200 MA
        c4 = sma_50 > sma_150 > sma_200
        criteria_met.append(c4)
        
        # 5. Current price > 50 MA
        c5 = current_close > sma_50
        criteria_met.append(c5)
        
        # 6. Current price >= 30% above 52-week low
        c6 = current_close >= (1.30 * low_52_week)
        criteria_met.append(c6)
        
        # 7. Current price within 25% of 52-week high
        c7 = current_close >= (0.75 * high_52_week)
        criteria_met.append(c7)
        
        # 8. RS Rating >= 70
        c8 = rs_rating >= 70
        criteria_met.append(c8)
        
        # Count how many criteria met
        num_criteria_met = sum(criteria_met)
        
        # Must meet all 8 criteria to be included
        if num_criteria_met == total_criteria:
            # Get additional info
            stock_info = yf.Ticker(ticker).info
            
            stock_data = {
                'ticker': ticker,
                'company': stock_info.get('longName') or stock_info.get('shortName', ticker),
                'sector': stock_info.get('sector', 'N/A'),
                'industry': stock_info.get('industry', 'N/A'),
                'price': safe_round(current_close, 2),
                'price_display': f"${current_close:.2f}" if current_close else 'N/A',
                'change_pct': None,  # Will be calculated separately
                'rs_rating': rs_rating,
                'sma_50': safe_round(sma_50, 2),
                'sma_150': safe_round(sma_150, 2),
                'sma_200': safe_round(sma_200, 2),
                'low_52_week': safe_round(low_52_week, 2),
                'high_52_week': safe_round(high_52_week, 2),
                'criteria_met': f"{num_criteria_met}/{total_criteria}",
                'market_cap': clean_value(stock_info.get('marketCap')),
                'volume': clean_value(stock_info.get('volume'))
            }
            
            # Calculate % change from previous close
            if len(df) >= 2:
                prev_close = clean_value(df['Close'].iloc[-2])
                if prev_close and prev_close != 0:
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    stock_data['change_pct'] = safe_round(change_pct, 2)
            
            return stock_data, None
        else:
            return None, f"Only {num_criteria_met}/{total_criteria} criteria met"
            
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_minervini_screener():
    """
    Run the complete Minervini screening process.
    Returns list of stocks meeting all criteria.
    """
    print("Starting Minervini screener...")
    
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    print(f"Screening {len(tickers)} stocks...")
    
    # Calculate index return for RS rating
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        index_df = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        index_df = normalize_dataframe(index_df)

        if not index_df.empty:
            index_return = (index_df['Close'].iloc[-1] / index_df['Close'].iloc[0]) - 1
        else:
            index_return = 0.20  # Default 20% if index data fails
    except Exception as e:
        print(f"Error downloading index data: {e}")
        index_return = 0.20
    
    print(f"S&P 500 return: {index_return*100:.2f}%")
    
    # First pass: Calculate RS ratings for all stocks
    rs_ratings = {}
    valid_tickers = []
    
    for ticker in tickers:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df = normalize_dataframe(df)

            if not df.empty and len(df) >= 200:
                stock_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
                rs_rating = calculate_rs_rating(stock_return, index_return)

                # Only continue with stocks that have RS >= 70
                if rs_rating >= 70:
                    rs_ratings[ticker] = rs_rating
                    valid_tickers.append(ticker)

            time.sleep(0.05)  # Rate limiting

        except Exception as e:
            print(f"Error calculating RS for {ticker}: {e}")
            continue

    print(f"Found {len(valid_tickers)} stocks with RS >= 70")

    # Second pass: Apply full Minervini criteria to high RS stocks
    minervini_picks = []

    for ticker in valid_tickers:
        rs_rating = rs_ratings[ticker]
        stock_data, error = check_minervini_criteria(ticker, rs_rating)

        if stock_data:
            minervini_picks.append(stock_data)
            print(f"✓ {ticker} - RS: {rs_rating}")

        time.sleep(0.05)  # Rate limiting
    
    # Sort by RS rating (highest first)
    minervini_picks.sort(key=lambda x: x['rs_rating'], reverse=True)
    
    print(f"\nScreener complete: {len(minervini_picks)} stocks meet all criteria")
    
    return minervini_picks


def format_large_number(value):
    """Format large numbers with suffixes"""
    if value is None:
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
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:,.0f}"


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
            return jsonify(cached)
        
        # Run screener if no valid cache
        print("Running fresh screener...")
        minervini_picks = run_minervini_screener()
        
        # Calculate statistics
        avg_rs = np.mean([s['rs_rating'] for s in minervini_picks]) if minervini_picks else 0
        
        # Count sectors
        sector_counts = {}
        for stock in minervini_picks:
            sector = stock.get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Format stocks for response
        for stock in minervini_picks:
            if stock.get('market_cap'):
                stock['market_cap_display'] = format_large_number(stock['market_cap'])
            else:
                stock['market_cap_display'] = 'N/A'
            
            if stock.get('volume'):
                stock['volume_display'] = format_volume(stock['volume'])
            else:
                stock['volume_display'] = 'N/A'
        
        response = {
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
        
        # Save to cache
        save_cache(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in get_picks: {e}")
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@picker_bp.route('/api/picks/refresh', methods=['POST'])
def refresh_picks():
    """
    Force refresh the picks (clears cache and runs screener).
    Can be used for manual updates.
    """
    try:
        # Clear cache
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        
        # Run screener
        return get_picks()
        
    except Exception as e:
        return jsonify({
            'error': f'Refresh failed: {str(e)}'
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
        'cache_duration': CACHE_DURATION
    })
