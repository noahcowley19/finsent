from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from cache import financial_cache, Cache

sectors_bp = Blueprint('sectors', __name__)


# =============================================================================
# SECTOR ETFs
# =============================================================================

SECTOR_ETFS = {
    'Technology': {'etf': 'XLK', 'color': '#3B82F6'},
    'Healthcare': {'etf': 'XLV', 'color': '#10B981'},
    'Financials': {'etf': 'XLF', 'color': '#F59E0B'},
    'Consumer Discretionary': {'etf': 'XLY', 'color': '#EC4899'},
    'Communication Services': {'etf': 'XLC', 'color': '#8B5CF6'},
    'Industrials': {'etf': 'XLI', 'color': '#6366F1'},
    'Consumer Staples': {'etf': 'XLP', 'color': '#14B8A6'},
    'Energy': {'etf': 'XLE', 'color': '#EF4444'},
    'Utilities': {'etf': 'XLU', 'color': '#84CC16'},
    'Real Estate': {'etf': 'XLRE', 'color': '#F97316'},
    'Materials': {'etf': 'XLB', 'color': '#06B6D4'},
}

# Top holdings per sector
SECTOR_STOCKS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD', 'CSCO', 'ACN'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG'],
    'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA'],
    'Industrials': ['CAT', 'DE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'UNP', 'FDX'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'KHC'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC'],
    'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'DLR', 'WELL', 'AVB', 'EQR'],
    'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'ECL', 'VMC'],
}


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def safe_round(value, decimals=2):
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return round(cleaned, decimals)
    except:
        return None


def get_sector_data(sector_name, etf_ticker, period='1d'):
    """Get performance data for a sector ETF"""
    try:
        stock = yf.Ticker(etf_ticker)
        info = stock.info
        
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('regularMarketPreviousClose')
        
        if not price or not prev_close:
            return None
        
        change_pct = ((price - prev_close) / prev_close) * 100
        
        # Get historical data for different periods
        hist = stock.history(period='1mo')
        
        data = {
            'sector': sector_name,
            'etf': etf_ticker,
            'price': safe_round(price, 2),
            'change_1d': safe_round(change_pct, 2),
            'volume': info.get('regularMarketVolume'),
        }
        
        # Calculate period changes
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            
            if len(hist) >= 5:
                week_ago = hist['Close'].iloc[-5]
                data['change_1w'] = safe_round(((current - week_ago) / week_ago) * 100, 2)
            
            if len(hist) >= 21:
                month_ago = hist['Close'].iloc[-21]
                data['change_1m'] = safe_round(((current - month_ago) / month_ago) * 100, 2)
        
        return data
        
    except:
        return None


def get_stock_quick_data(ticker):
    """Get quick price data for a stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('regularMarketPreviousClose')
        
        if not price or not prev_close:
            return None
        
        return {
            'ticker': ticker,
            'name': info.get('longName') or info.get('shortName') or ticker,
            'price': safe_round(price, 2),
            'change_pct': safe_round(((price - prev_close) / prev_close) * 100, 2),
            'market_cap': info.get('marketCap'),
        }
    except:
        return None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@sectors_bp.route('/api/sectors/heatmap', methods=['GET'])
def get_sector_heatmap():
    """Get sector heatmap data"""
    try:
        cache_key = 'sector_heatmap'
        cached = financial_cache.get('SECTORS', cache_key)
        if cached is not None:
            return jsonify(cached)
        
        sectors_data = []
        
        with ThreadPoolExecutor(max_workers=11) as executor:
            futures = {
                executor.submit(get_sector_data, name, info['etf']): (name, info)
                for name, info in SECTOR_ETFS.items()
            }
            
            for future in as_completed(futures):
                name, info = futures[future]
                try:
                    data = future.result()
                    if data:
                        data['color'] = info['color']
                        sectors_data.append(data)
                except:
                    continue
        
        # Sort by daily change
        sectors_data.sort(key=lambda x: x.get('change_1d', 0), reverse=True)
        
        # Calculate market average
        avg_change = np.mean([s.get('change_1d', 0) for s in sectors_data if s.get('change_1d') is not None])
        
        result = {
            'sectors': sectors_data,
            'market_average': safe_round(avg_change, 2),
            'best_sector': sectors_data[0]['sector'] if sectors_data else None,
            'worst_sector': sectors_data[-1]['sector'] if sectors_data else None,
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set('SECTORS', cache_key, result, 300)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@sectors_bp.route('/api/sectors/rotation', methods=['GET'])
def get_sector_rotation():
    """Analyze sector rotation patterns"""
    try:
        cache_key = 'sector_rotation'
        cached = financial_cache.get('SECTORS', cache_key)
        if cached is not None:
            return jsonify(cached)
        
        # Get 1-month data for all sector ETFs
        rotation_data = []
        
        for sector_name, info in SECTOR_ETFS.items():
            try:
                stock = yf.Ticker(info['etf'])
                hist = stock.history(period='3mo')
                
                if hist.empty:
                    continue
                
                close = hist['Close']
                
                # Calculate returns for different periods
                current = close.iloc[-1]
                
                data = {
                    'sector': sector_name,
                    'etf': info['etf'],
                    'current_price': safe_round(current, 2),
                }
                
                # 1-week return
                if len(close) >= 5:
                    data['return_1w'] = safe_round(((current - close.iloc[-5]) / close.iloc[-5]) * 100, 2)
                
                # 1-month return
                if len(close) >= 21:
                    data['return_1m'] = safe_round(((current - close.iloc[-21]) / close.iloc[-21]) * 100, 2)
                
                # 3-month return
                data['return_3m'] = safe_round(((current - close.iloc[0]) / close.iloc[0]) * 100, 2)
                
                # Momentum score (weighted average of period returns)
                weights = {'return_1w': 0.5, 'return_1m': 0.3, 'return_3m': 0.2}
                momentum = sum(data.get(k, 0) * v for k, v in weights.items() if data.get(k) is not None)
                data['momentum_score'] = safe_round(momentum, 2)
                
                rotation_data.append(data)
                
            except:
                continue
        
        # Sort by momentum score
        rotation_data.sort(key=lambda x: x.get('momentum_score', 0), reverse=True)
        
        # Identify rotation patterns
        leaders = rotation_data[:3]
        laggards = rotation_data[-3:]
        
        # Classify market phase based on sector leadership
        tech_momentum = next((s['momentum_score'] for s in rotation_data if s['sector'] == 'Technology'), 0)
        staples_momentum = next((s['momentum_score'] for s in rotation_data if s['sector'] == 'Consumer Staples'), 0)
        utilities_momentum = next((s['momentum_score'] for s in rotation_data if s['sector'] == 'Utilities'), 0)
        
        if tech_momentum > 0 and tech_momentum > utilities_momentum:
            market_phase = 'Risk-On'
            phase_description = 'Growth sectors leading - bullish sentiment'
        elif utilities_momentum > tech_momentum and staples_momentum > tech_momentum:
            market_phase = 'Risk-Off'
            phase_description = 'Defensive sectors leading - cautious sentiment'
        else:
            market_phase = 'Mixed'
            phase_description = 'No clear sector leadership'
        
        result = {
            'sectors': rotation_data,
            'leaders': [s['sector'] for s in leaders],
            'laggards': [s['sector'] for s in laggards],
            'market_phase': market_phase,
            'phase_description': phase_description,
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set('SECTORS', cache_key, result, 600)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@sectors_bp.route('/api/sectors/<sector>/stocks', methods=['GET'])
def get_sector_stocks(sector):
    """Get top stocks in a sector"""
    try:
        # Normalize sector name
        sector_normalized = sector.replace('_', ' ').title()
        
        if sector_normalized not in SECTOR_STOCKS:
            return jsonify({
                'error': f'Unknown sector: {sector}',
                'available_sectors': list(SECTOR_STOCKS.keys()),
                'error_type': 'validation'
            }), 400
        
        tickers = SECTOR_STOCKS[sector_normalized]
        stocks_data = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_stock_quick_data, ticker): ticker for ticker in tickers}
            
            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data:
                        stocks_data.append(data)
                except:
                    continue
        
        # Sort by daily change
        stocks_data.sort(key=lambda x: x.get('change_pct', 0), reverse=True)
        
        return jsonify({
            'sector': sector_normalized,
            'stocks': stocks_data,
            'count': len(stocks_data),
            'top_performer': stocks_data[0]['ticker'] if stocks_data else None,
            'worst_performer': stocks_data[-1]['ticker'] if stocks_data else None,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@sectors_bp.route('/api/sectors/list', methods=['GET'])
def list_sectors():
    """List all available sectors"""
    try:
        sectors = [
            {
                'name': name,
                'etf': info['etf'],
                'color': info['color'],
                'stocks_count': len(SECTOR_STOCKS.get(name, []))
            }
            for name, info in SECTOR_ETFS.items()
        ]
        
        return jsonify({
            'sectors': sectors,
            'count': len(sectors),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
