from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from cache import financial_cache, Cache

screener_bp = Blueprint('screener', __name__)


# =============================================================================
# STOCK UNIVERSE
# =============================================================================

# Popular stocks for screening (can be expanded)
STOCK_UNIVERSE = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'CRM',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    # Industrial
    'CAT', 'DE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'UNP',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
    # Communication
    'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'TTWO', 'PARA',
    # REITs
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'DLR', 'WELL', 'AVB', 'EQR',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC',
    # Materials
    'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'ECL', 'VMC',
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


def get_stock_metrics(ticker):
    """Get comprehensive metrics for a single stock"""
    cache_key = f'screener_metrics_{ticker}'
    cached = financial_cache.get(ticker, cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            return None
        
        # Get historical data for technical indicators
        hist = stock.history(period='6mo')
        
        metrics = {
            'ticker': ticker.upper(),
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            
            # Price data
            'price': safe_round(info.get('currentPrice') or info.get('regularMarketPrice'), 2),
            'change_pct': safe_round(info.get('regularMarketChangePercent'), 2),
            'volume': info.get('regularMarketVolume'),
            'avg_volume': info.get('averageVolume'),
            
            # Valuation
            'market_cap': info.get('marketCap'),
            'pe_ratio': safe_round(info.get('trailingPE'), 2),
            'forward_pe': safe_round(info.get('forwardPE'), 2),
            'peg_ratio': safe_round(info.get('pegRatio'), 2),
            'ps_ratio': safe_round(info.get('priceToSalesTrailing12Months'), 2),
            'pb_ratio': safe_round(info.get('priceToBook'), 2),
            
            # Dividends
            'dividend_yield': safe_round((info.get('dividendYield') or 0) * 100, 2),
            'dividend_rate': safe_round(info.get('dividendRate'), 2),
            'payout_ratio': safe_round((info.get('payoutRatio') or 0) * 100, 2),
            
            # Profitability
            'profit_margin': safe_round((info.get('profitMargins') or 0) * 100, 2),
            'operating_margin': safe_round((info.get('operatingMargins') or 0) * 100, 2),
            'roe': safe_round((info.get('returnOnEquity') or 0) * 100, 2),
            'roa': safe_round((info.get('returnOnAssets') or 0) * 100, 2),
            
            # Growth
            'revenue_growth': safe_round((info.get('revenueGrowth') or 0) * 100, 2),
            'earnings_growth': safe_round((info.get('earningsGrowth') or 0) * 100, 2),
            
            # Financial health
            'current_ratio': safe_round(info.get('currentRatio'), 2),
            'debt_to_equity': safe_round(info.get('debtToEquity'), 2),
            
            # 52-week range
            'fifty_two_week_high': safe_round(info.get('fiftyTwoWeekHigh'), 2),
            'fifty_two_week_low': safe_round(info.get('fiftyTwoWeekLow'), 2),
            'beta': safe_round(info.get('beta'), 2),
        }
        
        # Calculate 52-week position
        if metrics['fifty_two_week_high'] and metrics['fifty_two_week_low'] and metrics['price']:
            range_size = metrics['fifty_two_week_high'] - metrics['fifty_two_week_low']
            if range_size > 0:
                metrics['range_position'] = safe_round(
                    (metrics['price'] - metrics['fifty_two_week_low']) / range_size * 100, 1
                )
        
        # Technical indicators from historical data
        if not hist.empty and len(hist) >= 14:
            close = hist['Close']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            metrics['rsi'] = safe_round(rsi.iloc[-1], 2)
            
            # Moving Averages
            metrics['sma_20'] = safe_round(close.rolling(20).mean().iloc[-1], 2)
            metrics['sma_50'] = safe_round(close.rolling(50).mean().iloc[-1], 2) if len(close) >= 50 else None
            metrics['sma_200'] = safe_round(close.rolling(200).mean().iloc[-1], 2) if len(close) >= 200 else None
            
            # Above/below MAs
            metrics['above_sma_20'] = metrics['price'] > metrics['sma_20'] if metrics['sma_20'] else None
            metrics['above_sma_50'] = metrics['price'] > metrics['sma_50'] if metrics['sma_50'] else None
            
            # Volume ratio
            if 'Volume' in hist.columns:
                avg_vol = hist['Volume'].rolling(20).mean().iloc[-1]
                current_vol = hist['Volume'].iloc[-1]
                metrics['volume_ratio'] = safe_round(current_vol / avg_vol, 2) if avg_vol else None
        
        financial_cache.set(ticker, cache_key, metrics, Cache.TTL_PRICE)
        return metrics
        
    except Exception as e:
        return None


def filter_stocks(stocks, filters):
    """Apply filters to list of stock metrics"""
    filtered = []
    
    for stock in stocks:
        if stock is None:
            continue
        
        passes = True
        
        for key, condition in filters.items():
            value = stock.get(key)
            
            if value is None:
                # Skip stocks with missing data for this filter
                if condition.get('required', False):
                    passes = False
                    break
                continue
            
            min_val = condition.get('min')
            max_val = condition.get('max')
            equals = condition.get('equals')
            
            if min_val is not None and value < min_val:
                passes = False
                break
            if max_val is not None and value > max_val:
                passes = False
                break
            if equals is not None and value != equals:
                passes = False
                break
        
        if passes:
            filtered.append(stock)
    
    return filtered


def sort_stocks(stocks, sort_by, ascending=False):
    """Sort stocks by a metric"""
    return sorted(
        [s for s in stocks if s.get(sort_by) is not None],
        key=lambda x: x.get(sort_by, 0),
        reverse=not ascending
    )


# =============================================================================
# PRESET SCREENS
# =============================================================================

PRESET_SCREENS = {
    'undervalued_growth': {
        'name': 'Undervalued Growth',
        'description': 'Low P/E with positive earnings growth',
        'filters': {
            'pe_ratio': {'min': 1, 'max': 20},
            'earnings_growth': {'min': 10},
            'market_cap': {'min': 10000000000}  # > $10B
        },
        'sort_by': 'peg_ratio',
        'ascending': True
    },
    'high_dividend': {
        'name': 'High Dividend Yield',
        'description': 'High yield with sustainable payout',
        'filters': {
            'dividend_yield': {'min': 3},
            'payout_ratio': {'max': 80},
            'market_cap': {'min': 5000000000}
        },
        'sort_by': 'dividend_yield',
        'ascending': False
    },
    'momentum': {
        'name': 'Momentum Plays',
        'description': 'Strong price momentum and volume',
        'filters': {
            'rsi': {'min': 50, 'max': 70},
            'above_sma_20': {'equals': True},
            'above_sma_50': {'equals': True},
            'volume_ratio': {'min': 1.2}
        },
        'sort_by': 'change_pct',
        'ascending': False
    },
    'value_stocks': {
        'name': 'Value Stocks',
        'description': 'Low valuation metrics',
        'filters': {
            'pe_ratio': {'min': 1, 'max': 15},
            'pb_ratio': {'min': 0.1, 'max': 3},
            'profit_margin': {'min': 5}
        },
        'sort_by': 'pe_ratio',
        'ascending': True
    },
    'quality': {
        'name': 'Quality Companies',
        'description': 'High profitability and returns',
        'filters': {
            'roe': {'min': 15},
            'profit_margin': {'min': 10},
            'current_ratio': {'min': 1.5},
            'debt_to_equity': {'max': 100}
        },
        'sort_by': 'roe',
        'ascending': False
    },
    'oversold': {
        'name': 'Oversold Stocks',
        'description': 'RSI below 30 - potentially oversold',
        'filters': {
            'rsi': {'max': 30},
            'market_cap': {'min': 1000000000}
        },
        'sort_by': 'rsi',
        'ascending': True
    },
    'near_52_week_low': {
        'name': 'Near 52-Week Low',
        'description': 'Within 10% of yearly low',
        'filters': {
            'range_position': {'max': 10},
            'market_cap': {'min': 5000000000}
        },
        'sort_by': 'range_position',
        'ascending': True
    },
    'small_cap_growth': {
        'name': 'Small Cap Growth',
        'description': 'Smaller companies with high growth',
        'filters': {
            'market_cap': {'min': 300000000, 'max': 2000000000},
            'revenue_growth': {'min': 15},
            'earnings_growth': {'min': 10}
        },
        'sort_by': 'revenue_growth',
        'ascending': False
    }
}


# =============================================================================
# API ENDPOINTS
# =============================================================================

@screener_bp.route('/api/screener/scan', methods=['POST'])
def scan_stocks():
    """Run stock screener with custom filters"""
    try:
        data = request.get_json()
        filters = data.get('filters', {})
        sort_by = data.get('sort_by', 'market_cap')
        ascending = data.get('ascending', False)
        limit = min(data.get('limit', 50), 100)
        preset = data.get('preset')
        
        # Use preset if specified
        if preset and preset in PRESET_SCREENS:
            screen = PRESET_SCREENS[preset]
            filters = screen['filters']
            sort_by = screen.get('sort_by', 'market_cap')
            ascending = screen.get('ascending', False)
        
        # Fetch metrics for all stocks in parallel
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_stock_metrics, ticker): ticker for ticker in STOCK_UNIVERSE}
            
            for future in as_completed(futures):
                try:
                    metrics = future.result()
                    if metrics:
                        all_metrics.append(metrics)
                except:
                    continue
        
        # Apply filters
        filtered = filter_stocks(all_metrics, filters)
        
        # Sort
        sorted_stocks = sort_stocks(filtered, sort_by, ascending)
        
        # Limit results
        results = sorted_stocks[:limit]
        
        return jsonify({
            'results': results,
            'count': len(results),
            'total_scanned': len(all_metrics),
            'filters_applied': filters,
            'sort_by': sort_by,
            'preset': preset,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@screener_bp.route('/api/screener/presets', methods=['GET'])
def get_presets():
    """Get available preset screens"""
    try:
        presets = []
        for key, screen in PRESET_SCREENS.items():
            presets.append({
                'id': key,
                'name': screen['name'],
                'description': screen['description'],
                'filters': screen['filters']
            })
        
        return jsonify({
            'presets': presets,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@screener_bp.route('/api/screener/metrics', methods=['GET'])
def get_available_metrics():
    """Get list of available metrics for filtering"""
    try:
        metrics = [
            {'key': 'price', 'name': 'Price', 'type': 'number'},
            {'key': 'market_cap', 'name': 'Market Cap', 'type': 'number'},
            {'key': 'pe_ratio', 'name': 'P/E Ratio', 'type': 'number'},
            {'key': 'forward_pe', 'name': 'Forward P/E', 'type': 'number'},
            {'key': 'peg_ratio', 'name': 'PEG Ratio', 'type': 'number'},
            {'key': 'ps_ratio', 'name': 'P/S Ratio', 'type': 'number'},
            {'key': 'pb_ratio', 'name': 'P/B Ratio', 'type': 'number'},
            {'key': 'dividend_yield', 'name': 'Dividend Yield %', 'type': 'number'},
            {'key': 'payout_ratio', 'name': 'Payout Ratio %', 'type': 'number'},
            {'key': 'profit_margin', 'name': 'Profit Margin %', 'type': 'number'},
            {'key': 'operating_margin', 'name': 'Operating Margin %', 'type': 'number'},
            {'key': 'roe', 'name': 'ROE %', 'type': 'number'},
            {'key': 'roa', 'name': 'ROA %', 'type': 'number'},
            {'key': 'revenue_growth', 'name': 'Revenue Growth %', 'type': 'number'},
            {'key': 'earnings_growth', 'name': 'Earnings Growth %', 'type': 'number'},
            {'key': 'current_ratio', 'name': 'Current Ratio', 'type': 'number'},
            {'key': 'debt_to_equity', 'name': 'Debt/Equity', 'type': 'number'},
            {'key': 'beta', 'name': 'Beta', 'type': 'number'},
            {'key': 'rsi', 'name': 'RSI', 'type': 'number'},
            {'key': 'range_position', 'name': '52-Week Range %', 'type': 'number'},
            {'key': 'volume_ratio', 'name': 'Volume Ratio', 'type': 'number'},
            {'key': 'above_sma_20', 'name': 'Above SMA 20', 'type': 'boolean'},
            {'key': 'above_sma_50', 'name': 'Above SMA 50', 'type': 'boolean'},
            {'key': 'sector', 'name': 'Sector', 'type': 'string'},
        ]
        
        return jsonify({
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@screener_bp.route('/api/screener/stock/<ticker>', methods=['GET'])
def get_single_stock_metrics(ticker):
    """Get screener metrics for a single stock"""
    try:
        ticker = ticker.strip().upper()
        metrics = get_stock_metrics(ticker)
        
        if metrics is None:
            return jsonify({
                'error': f'Unable to fetch metrics for {ticker}',
                'error_type': 'data_error'
            }), 400
        
        return jsonify(metrics)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
