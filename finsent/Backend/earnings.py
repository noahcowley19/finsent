from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from cache import financial_cache, Cache

earnings_bp = Blueprint('earnings', __name__)


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


def get_earnings_calendar(ticker):
    """Get upcoming earnings date and info for a ticker"""
    cache_key = f'earnings_calendar_{ticker}'
    cached = financial_cache.get(ticker, cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get calendar info (includes earnings date)
        calendar = stock.calendar
        info = stock.info
        
        result = {
            'ticker': ticker.upper(),
            'company_name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector'),
            'industry': info.get('industry'),
        }
        
        # Extract earnings date from calendar
        if calendar is not None and not calendar.empty:
            if 'Earnings Date' in calendar.columns:
                earnings_dates = calendar['Earnings Date'].values
                if len(earnings_dates) > 0:
                    # Convert to datetime
                    next_earnings = pd.Timestamp(earnings_dates[0])
                    result['next_earnings_date'] = next_earnings.strftime('%Y-%m-%d')
                    result['days_until_earnings'] = (next_earnings - pd.Timestamp.now()).days
                    
                    if len(earnings_dates) > 1:
                        result['earnings_date_range'] = {
                            'start': pd.Timestamp(earnings_dates[0]).strftime('%Y-%m-%d'),
                            'end': pd.Timestamp(earnings_dates[-1]).strftime('%Y-%m-%d')
                        }
            
            # Extract estimates if available
            if 'Earnings Average' in calendar.index:
                result['eps_estimate'] = safe_round(float(calendar.loc['Earnings Average'].values[0]), 2)
            if 'Revenue Average' in calendar.index:
                result['revenue_estimate'] = clean_value(float(calendar.loc['Revenue Average'].values[0]))
        
        # Get recent price performance
        hist = stock.history(period='1mo')
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            month_ago_price = hist['Close'].iloc[0]
            result['current_price'] = safe_round(current_price, 2)
            result['month_change_pct'] = safe_round(((current_price - month_ago_price) / month_ago_price) * 100, 2)
        
        result['timestamp'] = datetime.now().isoformat()
        
        financial_cache.set(ticker, cache_key, result, Cache.TTL_PRICE)
        return result
        
    except Exception as e:
        return {
            'ticker': ticker.upper(),
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def get_earnings_history(ticker):
    """Get historical earnings data with surprises"""
    cache_key = f'earnings_history_{ticker}'
    cached = financial_cache.get(ticker, cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get earnings history
        earnings_hist = stock.earnings_history
        
        history = []
        
        if earnings_hist is not None and not earnings_hist.empty:
            for idx, row in earnings_hist.iterrows():
                quarter_data = {
                    'quarter': idx.strftime('%Y-Q%q') if hasattr(idx, 'strftime') else str(idx),
                    'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                }
                
                # EPS data
                if 'epsActual' in row:
                    quarter_data['eps_actual'] = safe_round(row['epsActual'], 2)
                if 'epsEstimate' in row:
                    quarter_data['eps_estimate'] = safe_round(row['epsEstimate'], 2)
                if 'epsDifference' in row:
                    quarter_data['eps_surprise'] = safe_round(row['epsDifference'], 2)
                if 'surprisePercent' in row:
                    quarter_data['surprise_pct'] = safe_round(row['surprisePercent'], 2)
                
                # Determine beat/miss
                if 'epsDifference' in row and row['epsDifference'] is not None:
                    if row['epsDifference'] > 0:
                        quarter_data['result'] = 'beat'
                    elif row['epsDifference'] < 0:
                        quarter_data['result'] = 'miss'
                    else:
                        quarter_data['result'] = 'met'
                
                history.append(quarter_data)
        
        # Calculate stats
        beats = sum(1 for h in history if h.get('result') == 'beat')
        misses = sum(1 for h in history if h.get('result') == 'miss')
        total = len([h for h in history if 'result' in h])
        
        result = {
            'ticker': ticker.upper(),
            'company_name': info.get('longName') or info.get('shortName') or ticker,
            'history': history[-12:],  # Last 12 quarters (3 years)
            'stats': {
                'total_quarters': total,
                'beats': beats,
                'misses': misses,
                'beat_rate': safe_round((beats / total) * 100, 1) if total > 0 else None,
                'avg_surprise_pct': safe_round(
                    np.mean([h['surprise_pct'] for h in history if h.get('surprise_pct') is not None]),
                    2
                )
            },
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set(ticker, cache_key, result, Cache.TTL_SENTIMENT)
        return result
        
    except Exception as e:
        return {
            'ticker': ticker.upper(),
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def get_pre_earnings_analysis(ticker):
    """Generate pre-earnings analysis including volatility and options data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data for volatility
        hist = stock.history(period='3mo')
        
        analysis = {
            'ticker': ticker.upper(),
            'company_name': info.get('longName') or info.get('shortName') or ticker,
        }
        
        if not hist.empty:
            # Calculate historical volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            analysis['historical_volatility'] = safe_round(volatility, 2)
            
            # Calculate average volume
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()
            analysis['avg_volume'] = int(avg_volume) if avg_volume else None
            analysis['recent_volume_ratio'] = safe_round(recent_volume / avg_volume, 2) if avg_volume else None
            
            # Price momentum going into earnings
            current_price = hist['Close'].iloc[-1]
            price_20d_ago = hist['Close'].iloc[-20] if len(hist) >= 20 else hist['Close'].iloc[0]
            analysis['momentum_20d'] = safe_round(((current_price - price_20d_ago) / price_20d_ago) * 100, 2)
        
        # Get implied volatility from options if available
        try:
            options_dates = stock.options
            if options_dates:
                nearest_expiry = options_dates[0]
                opt_chain = stock.option_chain(nearest_expiry)
                
                # Get ATM options
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if current_price and not opt_chain.calls.empty:
                    calls = opt_chain.calls
                    atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
                    
                    if 'impliedVolatility' in atm_call.columns:
                        iv = atm_call['impliedVolatility'].values[0]
                        analysis['implied_volatility'] = safe_round(iv * 100, 2)
                        
                        # IV vs HV comparison
                        if 'historical_volatility' in analysis:
                            iv_hv_ratio = (iv * 100) / analysis['historical_volatility']
                            analysis['iv_hv_ratio'] = safe_round(iv_hv_ratio, 2)
                            
                            if iv_hv_ratio > 1.3:
                                analysis['iv_status'] = 'elevated'
                                analysis['iv_interpretation'] = 'Options are pricing in significant earnings move'
                            elif iv_hv_ratio > 1.1:
                                analysis['iv_status'] = 'slightly_elevated'
                                analysis['iv_interpretation'] = 'Options show modest earnings expectations'
                            else:
                                analysis['iv_status'] = 'normal'
                                analysis['iv_interpretation'] = 'Options not showing unusual activity'
        except:
            pass  # Options data not always available
        
        # Get analyst recommendations
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            recent_recs = recommendations.tail(5)
            analysis['recent_analyst_actions'] = len(recent_recs)
        
        analysis['timestamp'] = datetime.now().isoformat()
        
        return analysis
        
    except Exception as e:
        return {
            'ticker': ticker.upper(),
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@earnings_bp.route('/api/earnings/calendar', methods=['GET'])
def earnings_calendar():
    """Get earnings calendar for specified tickers"""
    try:
        tickers_param = request.args.get('tickers', '')
        tickers = [t.strip().upper() for t in tickers_param.split(',') if t.strip()]
        
        if not tickers:
            return jsonify({
                'error': 'No tickers provided. Use ?tickers=AAPL,MSFT',
                'error_type': 'validation'
            }), 400
        
        if len(tickers) > 20:
            return jsonify({
                'error': 'Maximum 20 tickers allowed',
                'error_type': 'validation'
            }), 400
        
        results = []
        for ticker in tickers:
            data = get_earnings_calendar(ticker)
            if 'error' not in data:
                results.append(data)
        
        # Sort by next earnings date
        results_with_date = [r for r in results if r.get('next_earnings_date')]
        results_without_date = [r for r in results if not r.get('next_earnings_date')]
        
        results_with_date.sort(key=lambda x: x.get('next_earnings_date', '9999-12-31'))
        
        return jsonify({
            'earnings': results_with_date + results_without_date,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@earnings_bp.route('/api/earnings/history/<ticker>', methods=['GET'])
def earnings_history(ticker):
    """Get earnings history with surprises for a ticker"""
    try:
        ticker = ticker.strip().upper()
        
        if not ticker:
            return jsonify({
                'error': 'Ticker is required',
                'error_type': 'validation'
            }), 400
        
        result = get_earnings_history(ticker)
        
        if 'error' in result and 'history' not in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@earnings_bp.route('/api/earnings/analysis/<ticker>', methods=['GET'])
def earnings_analysis(ticker):
    """Get pre-earnings analysis for a ticker"""
    try:
        ticker = ticker.strip().upper()
        
        if not ticker:
            return jsonify({
                'error': 'Ticker is required',
                'error_type': 'validation'
            }), 400
        
        # Get both calendar and analysis
        calendar = get_earnings_calendar(ticker)
        analysis = get_pre_earnings_analysis(ticker)
        history = get_earnings_history(ticker)
        
        # Combine results
        result = {
            'ticker': ticker,
            'company_name': calendar.get('company_name') or analysis.get('company_name'),
            'upcoming_earnings': {
                'date': calendar.get('next_earnings_date'),
                'days_until': calendar.get('days_until_earnings'),
                'eps_estimate': calendar.get('eps_estimate'),
                'revenue_estimate': calendar.get('revenue_estimate'),
            } if calendar.get('next_earnings_date') else None,
            'market_context': {
                'current_price': analysis.get('current_price') or calendar.get('current_price'),
                'month_change_pct': calendar.get('month_change_pct'),
                'momentum_20d': analysis.get('momentum_20d'),
                'recent_volume_ratio': analysis.get('recent_volume_ratio'),
            },
            'volatility': {
                'historical_volatility': analysis.get('historical_volatility'),
                'implied_volatility': analysis.get('implied_volatility'),
                'iv_hv_ratio': analysis.get('iv_hv_ratio'),
                'iv_status': analysis.get('iv_status'),
                'iv_interpretation': analysis.get('iv_interpretation'),
            },
            'historical_performance': history.get('stats'),
            'recent_quarters': history.get('history', [])[:4],  # Last 4 quarters
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
