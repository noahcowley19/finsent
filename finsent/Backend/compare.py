from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from cache import financial_cache, Cache

compare_bp = Blueprint('compare', __name__)


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


def get_comparison_metrics(ticker):
    """Get comprehensive metrics for comparison"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return None
        
        hist = stock.history(period='1y')
        
        metrics = {
            'ticker': ticker.upper(),
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            
            # Price & Performance
            'price': safe_round(info.get('currentPrice') or info.get('regularMarketPrice'), 2),
            'market_cap': info.get('marketCap'),
            'change_pct_1d': safe_round(info.get('regularMarketChangePercent'), 2),
            
            # Valuation
            'pe_ratio': safe_round(info.get('trailingPE'), 2),
            'forward_pe': safe_round(info.get('forwardPE'), 2),
            'peg_ratio': safe_round(info.get('pegRatio'), 2),
            'ps_ratio': safe_round(info.get('priceToSalesTrailing12Months'), 2),
            'pb_ratio': safe_round(info.get('priceToBook'), 2),
            'ev_to_ebitda': safe_round(info.get('enterpriseToEbitda'), 2),
            
            # Profitability
            'profit_margin': safe_round((info.get('profitMargins') or 0) * 100, 2),
            'operating_margin': safe_round((info.get('operatingMargins') or 0) * 100, 2),
            'gross_margin': safe_round((info.get('grossMargins') or 0) * 100, 2),
            'roe': safe_round((info.get('returnOnEquity') or 0) * 100, 2),
            'roa': safe_round((info.get('returnOnAssets') or 0) * 100, 2),
            
            # Growth
            'revenue_growth': safe_round((info.get('revenueGrowth') or 0) * 100, 2),
            'earnings_growth': safe_round((info.get('earningsGrowth') or 0) * 100, 2),
            
            # Dividends
            'dividend_yield': safe_round((info.get('dividendYield') or 0) * 100, 2),
            'payout_ratio': safe_round((info.get('payoutRatio') or 0) * 100, 2),
            
            # Financial Health
            'current_ratio': safe_round(info.get('currentRatio'), 2),
            'debt_to_equity': safe_round(info.get('debtToEquity'), 2),
            
            # Risk
            'beta': safe_round(info.get('beta'), 2),
            'fifty_two_week_high': safe_round(info.get('fiftyTwoWeekHigh'), 2),
            'fifty_two_week_low': safe_round(info.get('fiftyTwoWeekLow'), 2),
        }
        
        # Calculate period returns from history
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            
            if len(hist) >= 5:
                metrics['return_1w'] = safe_round(
                    ((current - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100, 2
                )
            
            if len(hist) >= 21:
                metrics['return_1m'] = safe_round(
                    ((current - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]) * 100, 2
                )
            
            if len(hist) >= 63:
                metrics['return_3m'] = safe_round(
                    ((current - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63]) * 100, 2
                )
            
            if len(hist) >= 252:
                metrics['return_1y'] = safe_round(
                    ((current - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100, 2
                )
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            metrics['volatility'] = safe_round(returns.std() * np.sqrt(252) * 100, 2)
            
            # Sharpe ratio (assuming 5% risk-free rate)
            avg_return = returns.mean() * 252
            if metrics['volatility'] and metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = safe_round(
                    (avg_return * 100 - 5) / metrics['volatility'], 2
                )
        
        return metrics
        
    except:
        return None


def normalize_price_history(tickers, period='1y'):
    """Get normalized price history for overlay chart"""
    all_data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                # Normalize to 100 at start
                normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
                all_data[ticker] = [
                    {'date': d.strftime('%Y-%m-%d'), 'value': safe_round(v, 2)}
                    for d, v in normalized.items()
                ]
        except:
            continue
    
    return all_data


def calculate_winner(stocks, metric, higher_is_better=True):
    """Determine which stock wins for a metric"""
    valid = [(s['ticker'], s.get(metric)) for s in stocks if s.get(metric) is not None]
    if not valid:
        return None
    
    if higher_is_better:
        return max(valid, key=lambda x: x[1])[0]
    else:
        return min(valid, key=lambda x: x[1])[0]


# =============================================================================
# API ENDPOINTS
# =============================================================================

@compare_bp.route('/api/compare', methods=['POST'])
def compare_stocks():
    """Compare multiple stocks side by side"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if len(tickers) < 2:
            return jsonify({
                'error': 'At least 2 tickers required for comparison',
                'error_type': 'validation'
            }), 400
        
        if len(tickers) > 5:
            return jsonify({
                'error': 'Maximum 5 tickers for comparison',
                'error_type': 'validation'
            }), 400
        
        tickers = [t.strip().upper() for t in tickers]
        
        # Fetch metrics for all tickers
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(get_comparison_metrics, ticker): ticker for ticker in tickers}
            
            for future in as_completed(futures):
                try:
                    metrics = future.result()
                    if metrics:
                        all_metrics.append(metrics)
                except:
                    continue
        
        if len(all_metrics) < 2:
            return jsonify({
                'error': 'Could not fetch data for enough tickers',
                'error_type': 'data_error'
            }), 400
        
        # Calculate winners for each metric category
        winners = {
            'valuation': {
                'pe_ratio': calculate_winner(all_metrics, 'pe_ratio', higher_is_better=False),
                'peg_ratio': calculate_winner(all_metrics, 'peg_ratio', higher_is_better=False),
                'pb_ratio': calculate_winner(all_metrics, 'pb_ratio', higher_is_better=False),
            },
            'profitability': {
                'profit_margin': calculate_winner(all_metrics, 'profit_margin', higher_is_better=True),
                'roe': calculate_winner(all_metrics, 'roe', higher_is_better=True),
                'roa': calculate_winner(all_metrics, 'roa', higher_is_better=True),
            },
            'growth': {
                'revenue_growth': calculate_winner(all_metrics, 'revenue_growth', higher_is_better=True),
                'earnings_growth': calculate_winner(all_metrics, 'earnings_growth', higher_is_better=True),
            },
            'performance': {
                'return_1m': calculate_winner(all_metrics, 'return_1m', higher_is_better=True),
                'return_3m': calculate_winner(all_metrics, 'return_3m', higher_is_better=True),
                'return_1y': calculate_winner(all_metrics, 'return_1y', higher_is_better=True),
            },
            'risk': {
                'volatility': calculate_winner(all_metrics, 'volatility', higher_is_better=False),
                'sharpe_ratio': calculate_winner(all_metrics, 'sharpe_ratio', higher_is_better=True),
            },
            'dividends': {
                'dividend_yield': calculate_winner(all_metrics, 'dividend_yield', higher_is_better=True),
            }
        }
        
        # Count overall wins
        win_counts = {m['ticker']: 0 for m in all_metrics}
        for category in winners.values():
            for winner in category.values():
                if winner:
                    win_counts[winner] = win_counts.get(winner, 0) + 1
        
        overall_winner = max(win_counts, key=win_counts.get) if win_counts else None
        
        # Get normalized price history
        price_history = normalize_price_history(tickers)
        
        return jsonify({
            'stocks': all_metrics,
            'winners': winners,
            'win_counts': win_counts,
            'overall_winner': overall_winner,
            'price_history': price_history,
            'tickers_compared': [m['ticker'] for m in all_metrics],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@compare_bp.route('/api/compare/metrics', methods=['GET'])
def get_comparison_metrics_list():
    """Get list of metrics available for comparison"""
    try:
        categories = {
            'Valuation': [
                {'key': 'pe_ratio', 'name': 'P/E Ratio', 'lower_better': True},
                {'key': 'forward_pe', 'name': 'Forward P/E', 'lower_better': True},
                {'key': 'peg_ratio', 'name': 'PEG Ratio', 'lower_better': True},
                {'key': 'ps_ratio', 'name': 'P/S Ratio', 'lower_better': True},
                {'key': 'pb_ratio', 'name': 'P/B Ratio', 'lower_better': True},
                {'key': 'ev_to_ebitda', 'name': 'EV/EBITDA', 'lower_better': True},
            ],
            'Profitability': [
                {'key': 'profit_margin', 'name': 'Profit Margin %', 'lower_better': False},
                {'key': 'operating_margin', 'name': 'Operating Margin %', 'lower_better': False},
                {'key': 'gross_margin', 'name': 'Gross Margin %', 'lower_better': False},
                {'key': 'roe', 'name': 'ROE %', 'lower_better': False},
                {'key': 'roa', 'name': 'ROA %', 'lower_better': False},
            ],
            'Growth': [
                {'key': 'revenue_growth', 'name': 'Revenue Growth %', 'lower_better': False},
                {'key': 'earnings_growth', 'name': 'Earnings Growth %', 'lower_better': False},
            ],
            'Performance': [
                {'key': 'return_1w', 'name': '1 Week Return %', 'lower_better': False},
                {'key': 'return_1m', 'name': '1 Month Return %', 'lower_better': False},
                {'key': 'return_3m', 'name': '3 Month Return %', 'lower_better': False},
                {'key': 'return_1y', 'name': '1 Year Return %', 'lower_better': False},
            ],
            'Risk': [
                {'key': 'beta', 'name': 'Beta', 'lower_better': None},
                {'key': 'volatility', 'name': 'Volatility %', 'lower_better': True},
                {'key': 'sharpe_ratio', 'name': 'Sharpe Ratio', 'lower_better': False},
            ],
            'Dividends': [
                {'key': 'dividend_yield', 'name': 'Dividend Yield %', 'lower_better': False},
                {'key': 'payout_ratio', 'name': 'Payout Ratio %', 'lower_better': None},
            ],
            'Financial Health': [
                {'key': 'current_ratio', 'name': 'Current Ratio', 'lower_better': None},
                {'key': 'debt_to_equity', 'name': 'Debt/Equity', 'lower_better': True},
            ],
        }
        
        return jsonify({
            'categories': categories,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
