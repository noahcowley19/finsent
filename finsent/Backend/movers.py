from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from cache import financial_cache, Cache

movers_bp = Blueprint('movers', __name__)


# =============================================================================
# STOCK UNIVERSE FOR MOVERS
# =============================================================================

# S&P 500 representative sample (expand as needed)
MOVER_UNIVERSE = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'CRM',
    'ADBE', 'CSCO', 'ACN', 'IBM', 'INTC', 'AMD', 'QCOM', 'TXN', 'NOW', 'INTU',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
    'PNC', 'TFC', 'COF', 'BK', 'STT', 'AIG', 'MET', 'PRU', 'AFL', 'ALL',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'CVS', 'CI', 'ELV', 'MDT', 'ISRG', 'SYK', 'ZTS', 'VRTX',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    'LOW', 'TJX', 'BKNG', 'MAR', 'ORLY', 'AZO', 'ROST', 'DG', 'DLTR', 'YUM',
    # Industrial
    'CAT', 'DE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'UNP',
    'FDX', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'CMI', 'PCAR', 'NSC', 'CSX',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
    'WMB', 'HES', 'DVN', 'BKR', 'HAL', 'FANG', 'PXD', 'MRO', 'APA', 'CTRA',
    # Communication
    'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'TTWO', 'WBD',
]


def clean_value(value):
    """Clean NaN and Inf values"""
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
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


def get_stock_mover_data(ticker):
    """Get basic mover data for a stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            return None
        
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('regularMarketPreviousClose')
        
        if not price or not prev_close:
            return None
        
        change = price - prev_close
        change_pct = (change / prev_close) * 100
        
        data = {
            'ticker': ticker.upper(),
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown'),
            'price': safe_round(price, 2),
            'change': safe_round(change, 2),
            'change_pct': safe_round(change_pct, 2),
            'volume': info.get('regularMarketVolume'),
            'avg_volume': info.get('averageVolume'),
            'market_cap': info.get('marketCap'),
            'fifty_two_week_high': safe_round(info.get('fiftyTwoWeekHigh'), 2),
            'fifty_two_week_low': safe_round(info.get('fiftyTwoWeekLow'), 2),
        }
        
        # Calculate volume ratio
        if data['avg_volume'] and data['volume']:
            data['volume_ratio'] = safe_round(data['volume'] / data['avg_volume'], 2)
        
        # Check 52-week extremes
        if data['price'] and data['fifty_two_week_high']:
            data['near_52_high'] = data['price'] >= data['fifty_two_week_high'] * 0.98
        if data['price'] and data['fifty_two_week_low']:
            data['near_52_low'] = data['price'] <= data['fifty_two_week_low'] * 1.02
        
        return data
        
    except:
        return None


def fetch_all_movers():
    """Fetch mover data for all stocks in universe"""
    cache_key = 'all_movers_data'
    cached = financial_cache.get('MOVERS', cache_key)
    if cached is not None:
        return cached
    
    all_data = []
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(get_stock_mover_data, ticker): ticker for ticker in MOVER_UNIVERSE}
        
        for future in as_completed(futures):
            try:
                data = future.result()
                if data:
                    all_data.append(data)
            except:
                continue
    
    financial_cache.set('MOVERS', cache_key, all_data, 300)  # 5 min cache
    return all_data


# =============================================================================
# API ENDPOINTS
# =============================================================================

@movers_bp.route('/api/movers/gainers', methods=['GET'])
def get_gainers():
    """Get top gaining stocks"""
    try:
        limit = min(int(request.args.get('limit', 20)), 50)
        
        all_data = fetch_all_movers()
        
        # Filter positive change and sort
        gainers = [d for d in all_data if d.get('change_pct', 0) > 0]
        gainers.sort(key=lambda x: x.get('change_pct', 0), reverse=True)
        
        return jsonify({
            'gainers': gainers[:limit],
            'count': len(gainers[:limit]),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@movers_bp.route('/api/movers/losers', methods=['GET'])
def get_losers():
    """Get top losing stocks"""
    try:
        limit = min(int(request.args.get('limit', 20)), 50)
        
        all_data = fetch_all_movers()
        
        # Filter negative change and sort
        losers = [d for d in all_data if d.get('change_pct', 0) < 0]
        losers.sort(key=lambda x: x.get('change_pct', 0))
        
        return jsonify({
            'losers': losers[:limit],
            'count': len(losers[:limit]),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@movers_bp.route('/api/movers/active', methods=['GET'])
def get_most_active():
    """Get most active stocks by volume"""
    try:
        limit = min(int(request.args.get('limit', 20)), 50)
        
        all_data = fetch_all_movers()
        
        # Sort by volume
        active = [d for d in all_data if d.get('volume')]
        active.sort(key=lambda x: x.get('volume', 0), reverse=True)
        
        return jsonify({
            'active': active[:limit],
            'count': len(active[:limit]),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@movers_bp.route('/api/movers/unusual-volume', methods=['GET'])
def get_unusual_volume():
    """Get stocks with unusual volume (2x+ average)"""
    try:
        limit = min(int(request.args.get('limit', 20)), 50)
        threshold = float(request.args.get('threshold', 2.0))
        
        all_data = fetch_all_movers()
        
        # Filter by volume ratio
        unusual = [d for d in all_data if d.get('volume_ratio', 0) >= threshold]
        unusual.sort(key=lambda x: x.get('volume_ratio', 0), reverse=True)
        
        return jsonify({
            'unusual_volume': unusual[:limit],
            'count': len(unusual[:limit]),
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@movers_bp.route('/api/movers/52-week', methods=['GET'])
def get_52_week_extremes():
    """Get stocks at or near 52-week highs/lows"""
    try:
        all_data = fetch_all_movers()
        
        highs = [d for d in all_data if d.get('near_52_high')]
        lows = [d for d in all_data if d.get('near_52_low')]
        
        # Sort highs by how close to the high
        highs.sort(key=lambda x: x.get('change_pct', 0), reverse=True)
        lows.sort(key=lambda x: x.get('change_pct', 0))
        
        return jsonify({
            'new_highs': highs,
            'new_lows': lows,
            'highs_count': len(highs),
            'lows_count': len(lows),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@movers_bp.route('/api/movers/summary', methods=['GET'])
def get_movers_summary():
    """Get summary of all mover categories"""
    try:
        all_data = fetch_all_movers()
        
        gainers = sorted([d for d in all_data if d.get('change_pct', 0) > 0],
                        key=lambda x: x.get('change_pct', 0), reverse=True)[:10]
        
        losers = sorted([d for d in all_data if d.get('change_pct', 0) < 0],
                       key=lambda x: x.get('change_pct', 0))[:10]
        
        active = sorted([d for d in all_data if d.get('volume')],
                       key=lambda x: x.get('volume', 0), reverse=True)[:10]
        
        unusual = sorted([d for d in all_data if d.get('volume_ratio', 0) >= 2],
                        key=lambda x: x.get('volume_ratio', 0), reverse=True)[:10]
        
        # Calculate market breadth
        advances = len([d for d in all_data if d.get('change_pct', 0) > 0])
        declines = len([d for d in all_data if d.get('change_pct', 0) < 0])
        unchanged = len(all_data) - advances - declines
        
        return jsonify({
            'top_gainers': gainers,
            'top_losers': losers,
            'most_active': active,
            'unusual_volume': unusual,
            'market_breadth': {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': safe_round(advances / declines, 2) if declines > 0 else None
            },
            'total_stocks': len(all_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
