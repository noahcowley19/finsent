from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from cache import financial_cache, Cache

alerts_bp = Blueprint('alerts', __name__)


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


# =============================================================================
# ALERT DETECTION FUNCTIONS
# =============================================================================

def check_volume_spike(ticker):
    """Check for unusual volume activity (2x+ average)"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='3mo')
        
        if hist.empty or len(hist) < 20:
            return None
        
        avg_volume = hist['Volume'].iloc[:-1].mean()  # Exclude today
        today_volume = hist['Volume'].iloc[-1]
        
        if avg_volume == 0:
            return None
        
        ratio = today_volume / avg_volume
        
        if ratio >= 2.0:
            return {
                'type': 'volume_spike',
                'ticker': ticker.upper(),
                'severity': 'high' if ratio >= 3.0 else 'medium',
                'title': f'{ticker.upper()}: Volume Spike',
                'description': f'Trading volume is {ratio:.1f}x the 3-month average',
                'data': {
                    'today_volume': int(today_volume),
                    'avg_volume': int(avg_volume),
                    'ratio': safe_round(ratio, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
        
        return None
        
    except Exception as e:
        return None


def check_price_breakout(ticker):
    """Check if price broke 52-week high or low"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh')
        fifty_two_week_low = info.get('fiftyTwoWeekLow')
        
        if not all([current_price, fifty_two_week_high, fifty_two_week_low]):
            return None
        
        # Check for breakout (within 2% of high)
        high_distance = (fifty_two_week_high - current_price) / fifty_two_week_high
        low_distance = (current_price - fifty_two_week_low) / fifty_two_week_low
        
        if high_distance <= 0.02:  # Within 2% of 52-week high
            return {
                'type': 'price_breakout_high',
                'ticker': ticker.upper(),
                'severity': 'high',
                'title': f'{ticker.upper()}: Near 52-Week High',
                'description': f'Price is within 2% of 52-week high (${fifty_two_week_high:.2f})',
                'data': {
                    'current_price': safe_round(current_price, 2),
                    'fifty_two_week_high': safe_round(fifty_two_week_high, 2),
                    'distance_pct': safe_round(high_distance * 100, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
        
        if low_distance <= 0.05:  # Within 5% of 52-week low
            return {
                'type': 'price_breakout_low',
                'ticker': ticker.upper(),
                'severity': 'high',
                'title': f'{ticker.upper()}: Near 52-Week Low',
                'description': f'Price is within 5% of 52-week low (${fifty_two_week_low:.2f})',
                'data': {
                    'current_price': safe_round(current_price, 2),
                    'fifty_two_week_low': safe_round(fifty_two_week_low, 2),
                    'distance_pct': safe_round(low_distance * 100, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
        
        return None
        
    except Exception as e:
        return None


def check_earnings_approaching(ticker):
    """Check if earnings are within 14 days"""
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        info = stock.info
        
        if calendar is None or calendar.empty:
            return None
        
        if 'Earnings Date' not in calendar.columns:
            return None
        
        earnings_dates = calendar['Earnings Date'].values
        if len(earnings_dates) == 0:
            return None
        
        next_earnings = pd.Timestamp(earnings_dates[0])
        days_until = (next_earnings - pd.Timestamp.now()).days
        
        if 0 <= days_until <= 14:
            return {
                'type': 'earnings_approaching',
                'ticker': ticker.upper(),
                'severity': 'medium' if days_until > 7 else 'high',
                'title': f'{ticker.upper()}: Earnings in {days_until} days',
                'description': f'Earnings report expected on {next_earnings.strftime("%b %d, %Y")}',
                'data': {
                    'earnings_date': next_earnings.strftime('%Y-%m-%d'),
                    'days_until': days_until,
                    'company_name': info.get('longName') or info.get('shortName')
                },
                'timestamp': datetime.now().isoformat()
            }
        
        return None
        
    except Exception as e:
        return None


def check_high_put_call_ratio(ticker):
    """Check for elevated put/call ratio (unusual options activity)"""
    try:
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        
        if not options_dates:
            return None
        
        # Get nearest expiry
        nearest_expiry = options_dates[0]
        opt_chain = stock.option_chain(nearest_expiry)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        if calls.empty or puts.empty:
            return None
        
        # Calculate volumes
        call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
        put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
        
        if call_volume == 0:
            return None
        
        put_call_ratio = put_volume / call_volume
        
        if put_call_ratio >= 1.5:  # High put activity
            return {
                'type': 'high_put_call_ratio',
                'ticker': ticker.upper(),
                'severity': 'high' if put_call_ratio >= 2.0 else 'medium',
                'title': f'{ticker.upper()}: Elevated Put/Call Ratio',
                'description': f'Put/Call ratio of {put_call_ratio:.2f} suggests bearish options activity',
                'data': {
                    'put_call_ratio': safe_round(put_call_ratio, 2),
                    'put_volume': int(put_volume),
                    'call_volume': int(call_volume),
                    'expiry': nearest_expiry
                },
                'timestamp': datetime.now().isoformat()
            }
        
        return None
        
    except Exception as e:
        return None


def check_beta_divergence(ticker, market_ticker='^GSPC'):
    """Check if stock is diverging from expected beta behavior"""
    try:
        stock = yf.Ticker(ticker)
        market = yf.Ticker(market_ticker)
        
        # Get beta
        info = stock.info
        beta = info.get('beta') or 1.0
        
        # Get recent returns
        stock_hist = stock.history(period='5d')
        market_hist = market.history(period='5d')
        
        if stock_hist.empty or market_hist.empty:
            return None
        
        stock_return = (stock_hist['Close'].iloc[-1] - stock_hist['Close'].iloc[0]) / stock_hist['Close'].iloc[0]
        market_return = (market_hist['Close'].iloc[-1] - market_hist['Close'].iloc[0]) / market_hist['Close'].iloc[0]
        
        # Expected return based on beta
        expected_return = beta * market_return
        actual_return = stock_return
        
        divergence = abs(actual_return - expected_return)
        
        # If divergence is significant (more than 3%)
        if divergence > 0.03:
            direction = 'outperforming' if actual_return > expected_return else 'underperforming'
            return {
                'type': 'beta_divergence',
                'ticker': ticker.upper(),
                'severity': 'medium',
                'title': f'{ticker.upper()}: Beta Divergence',
                'description': f'Stock is {direction} expected beta-adjusted return by {divergence*100:.1f}%',
                'data': {
                    'beta': safe_round(beta, 2),
                    'stock_return_5d': safe_round(actual_return * 100, 2),
                    'expected_return': safe_round(expected_return * 100, 2),
                    'market_return_5d': safe_round(market_return * 100, 2),
                    'divergence_pct': safe_round(divergence * 100, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
        
        return None
        
    except Exception as e:
        return None


def check_all_alerts(ticker):
    """Run all alert checks for a ticker"""
    alerts = []
    
    # Run all checks
    checks = [
        check_volume_spike,
        check_price_breakout,
        check_earnings_approaching,
        check_high_put_call_ratio,
        check_beta_divergence,
    ]
    
    for check in checks:
        try:
            result = check(ticker)
            if result:
                alerts.append(result)
        except:
            continue
    
    return alerts


# =============================================================================
# API ENDPOINTS
# =============================================================================

@alerts_bp.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    """Check alerts for specified tickers"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({
                'error': 'No tickers provided',
                'error_type': 'validation'
            }), 400
        
        if len(tickers) > 20:
            return jsonify({
                'error': 'Maximum 20 tickers allowed',
                'error_type': 'validation'
            }), 400
        
        all_alerts = []
        
        for ticker in tickers:
            ticker = ticker.strip().upper()
            ticker_alerts = check_all_alerts(ticker)
            all_alerts.extend(ticker_alerts)
        
        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        all_alerts.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 2))
        
        return jsonify({
            'alerts': all_alerts,
            'count': len(all_alerts),
            'tickers_checked': len(tickers),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@alerts_bp.route('/api/alerts/types', methods=['GET'])
def get_alert_types():
    """Get available alert types with descriptions"""
    try:
        alert_types = [
            {
                'type': 'volume_spike',
                'name': 'Volume Spike',
                'description': 'Detects when trading volume is 2x or more the 3-month average',
                'severity_levels': ['medium (2-3x)', 'high (3x+)']
            },
            {
                'type': 'price_breakout_high',
                'name': '52-Week High Breakout',
                'description': 'Alerts when price is within 2% of 52-week high',
                'severity_levels': ['high']
            },
            {
                'type': 'price_breakout_low',
                'name': '52-Week Low Breakdown',
                'description': 'Alerts when price is within 5% of 52-week low',
                'severity_levels': ['high']
            },
            {
                'type': 'earnings_approaching',
                'name': 'Earnings Approaching',
                'description': 'Notifies when earnings are within 14 days',
                'severity_levels': ['medium (7-14 days)', 'high (0-7 days)']
            },
            {
                'type': 'high_put_call_ratio',
                'name': 'Unusual Options Activity',
                'description': 'Detects elevated put/call ratio suggesting bearish positioning',
                'severity_levels': ['medium (1.5-2x)', 'high (2x+)']
            },
            {
                'type': 'beta_divergence',
                'name': 'Beta Divergence',
                'description': 'Identifies stocks moving significantly different from expected beta behavior',
                'severity_levels': ['medium']
            },
            {
                'type': 'cluster_insider_buying',
                'name': 'Cluster Insider Buying Before Earnings',
                'description': 'Multiple insider purchases within 30 days of earnings (requires insider data integration)',
                'severity_levels': ['high'],
                'note': 'Coming soon - requires integration with insider module'
            },
            {
                'type': 'sentiment_reversal',
                'name': 'Sentiment Reversal',
                'description': 'Significant shift in social/news sentiment (requires sentiment data integration)',
                'severity_levels': ['medium', 'high'],
                'note': 'Coming soon - requires integration with sentiment module'
            }
        ]
        
        return jsonify({
            'alert_types': alert_types,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@alerts_bp.route('/api/alerts/ticker/<ticker>', methods=['GET'])
def get_ticker_alerts(ticker):
    """Get all alerts for a specific ticker"""
    try:
        ticker = ticker.strip().upper()
        
        if not ticker:
            return jsonify({
                'error': 'Ticker is required',
                'error_type': 'validation'
            }), 400
        
        alerts = check_all_alerts(ticker)
        
        return jsonify({
            'ticker': ticker,
            'alerts': alerts,
            'count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
