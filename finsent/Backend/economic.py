from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from cache import financial_cache, Cache

economic_bp = Blueprint('economic', __name__)


# =============================================================================
# STATIC ECONOMIC DATA (Updated periodically)
# =============================================================================

# Last updated: 2024-12 (update these values monthly from official sources)
STATIC_INDICATORS = {
    'fed_funds_rate': {
        'value': 4.50,
        'unit': '%',
        'name': 'Federal Funds Rate',
        'description': 'The interest rate at which banks lend reserve balances to other banks overnight.',
        'simple_explanation': 'This is the main interest rate the Fed uses to control the economy. Higher rates make borrowing more expensive, which can slow down spending and reduce inflation.',
        'source': 'Federal Reserve',
        'last_updated': '2024-12-18',
        'trend': 'stable',  # 'up', 'down', 'stable'
        'impact': 'Higher rates typically pressure stock valuations, especially growth stocks.'
    },
    'inflation_rate': {
        'value': 2.7,
        'unit': '%',
        'name': 'CPI Inflation Rate (YoY)',
        'description': 'The year-over-year change in the Consumer Price Index, measuring average price changes.',
        'simple_explanation': 'How much prices have risen compared to last year. The Fed targets 2% inflation. Higher inflation erodes purchasing power.',
        'source': 'Bureau of Labor Statistics',
        'last_updated': '2024-12-11',
        'trend': 'stable',
        'impact': 'High inflation often leads to higher interest rates, which can hurt stock prices.'
    },
    'unemployment_rate': {
        'value': 4.2,
        'unit': '%',
        'name': 'Unemployment Rate',
        'description': 'The percentage of the labor force that is unemployed and actively seeking work.',
        'simple_explanation': 'The percentage of people who want jobs but cannot find them. Low unemployment is good for workers but can lead to inflation.',
        'source': 'Bureau of Labor Statistics',
        'last_updated': '2024-12-06',
        'trend': 'up',
        'impact': 'Rising unemployment may signal economic weakness, prompting Fed rate cuts.'
    },
    'gdp_growth': {
        'value': 2.8,
        'unit': '%',
        'name': 'GDP Growth Rate (QoQ Annualized)',
        'description': 'The annualized percentage change in Gross Domestic Product from the previous quarter.',
        'simple_explanation': 'How fast the economy is growing. Positive growth means the economy is expanding. Negative growth (recession) is concerning.',
        'source': 'Bureau of Economic Analysis',
        'last_updated': '2024-11-27',
        'trend': 'stable',
        'impact': 'Strong GDP growth typically supports corporate earnings and stock prices.'
    }
}


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


def get_treasury_yields():
    """Fetch current Treasury yields using yfinance"""
    cache_key = 'treasury_yields'
    cached = financial_cache.get('TREASURY', cache_key)
    if cached is not None:
        return cached
    
    try:
        # Treasury yield tickers
        tickers = {
            '3M': '^IRX',   # 13-week Treasury
            '2Y': '^UST2Y', # 2-year (may not work)
            '5Y': '^FVX',   # 5-year Treasury
            '10Y': '^TNX',  # 10-year Treasury
            '30Y': '^TYX',  # 30-year Treasury
        }
        
        yields = {}
        
        for name, ticker in tickers.items():
            try:
                data = yf.Ticker(ticker)
                hist = data.history(period='5d')
                if not hist.empty:
                    # IRX is quoted differently (multiply by 0.01 if needed)
                    price = hist['Close'].iloc[-1]
                    if ticker == '^IRX':
                        # IRX is already in percentage terms
                        yields[name] = safe_round(price, 2)
                    else:
                        yields[name] = safe_round(price, 2)
            except:
                continue
        
        # Calculate curve inversion indicators
        if '10Y' in yields and '3M' in yields:
            yields['10Y_3M_spread'] = safe_round(yields['10Y'] - yields['3M'], 2)
            yields['curve_inverted'] = yields['10Y'] < yields['3M']
        
        if '10Y' in yields and '2Y' in yields:
            yields['10Y_2Y_spread'] = safe_round(yields['10Y'] - yields['2Y'], 2)
        
        result = {
            'yields': yields,
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set('TREASURY', cache_key, result, Cache.TTL_PRICE)
        return result
        
    except Exception as e:
        return {
            'yields': {},
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def get_yield_curve_history(days=365):
    """Get historical yield curve data"""
    cache_key = f'yield_curve_history_{days}'
    cached = financial_cache.get('TREASURY', cache_key)
    if cached is not None:
        return cached
    
    try:
        # Fetch historical data for key maturities
        tickers = {
            '3M': '^IRX',
            '10Y': '^TNX',
            '30Y': '^TYX',
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        history = {}
        
        for name, ticker in tickers.items():
            try:
                data = yf.Ticker(ticker)
                hist = data.history(start=start_date, end=end_date)
                if not hist.empty:
                    # Resample to weekly for performance
                    weekly = hist['Close'].resample('W').last().dropna()
                    history[name] = [
                        {
                            'date': date.strftime('%Y-%m-%d'),
                            'value': safe_round(value, 2)
                        }
                        for date, value in weekly.items()
                    ]
            except:
                continue
        
        # Calculate spread history if we have both
        if '10Y' in history and '3M' in history:
            spread_data = []
            ten_year = {item['date']: item['value'] for item in history['10Y']}
            three_month = {item['date']: item['value'] for item in history['3M']}
            
            for date in set(ten_year.keys()) & set(three_month.keys()):
                if ten_year[date] is not None and three_month[date] is not None:
                    spread_data.append({
                        'date': date,
                        'value': safe_round(ten_year[date] - three_month[date], 2)
                    })
            
            spread_data.sort(key=lambda x: x['date'])
            history['10Y_3M_spread'] = spread_data
        
        result = {
            'history': history,
            'period_days': days,
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set('TREASURY', cache_key, result, Cache.TTL_SENTIMENT)
        return result
        
    except Exception as e:
        return {
            'history': {},
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def get_market_correlation():
    """Calculate correlation between Treasury yields and S&P 500"""
    cache_key = 'market_correlation'
    cached = financial_cache.get('TREASURY', cache_key)
    if cached is not None:
        return cached
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Fetch data
        sp500 = yf.Ticker('^GSPC').history(start=start_date, end=end_date)['Close']
        tnx = yf.Ticker('^TNX').history(start=start_date, end=end_date)['Close']
        
        # Resample to daily and align
        sp500_daily = sp500.resample('D').last().dropna()
        tnx_daily = tnx.resample('D').last().dropna()
        
        # Align dates
        common_dates = sp500_daily.index.intersection(tnx_daily.index)
        sp500_aligned = sp500_daily.loc[common_dates]
        tnx_aligned = tnx_daily.loc[common_dates]
        
        # Calculate returns
        sp500_returns = sp500_aligned.pct_change().dropna()
        tnx_changes = tnx_aligned.diff().dropna()
        
        # Align again after calculating changes
        common_dates = sp500_returns.index.intersection(tnx_changes.index)
        
        # Calculate correlation
        correlation = sp500_returns.loc[common_dates].corr(tnx_changes.loc[common_dates])
        
        # Interpretation
        if correlation < -0.3:
            interpretation = 'Strong negative correlation - yields and stocks moving opposite'
        elif correlation < 0:
            interpretation = 'Weak negative correlation'
        elif correlation < 0.3:
            interpretation = 'Weak positive correlation'
        else:
            interpretation = 'Strong positive correlation - yields and stocks moving together'
        
        result = {
            'sp500_vs_10y_yield': {
                'correlation': safe_round(correlation, 3),
                'interpretation': interpretation,
                'period': '1 year'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set('TREASURY', cache_key, result, Cache.TTL_SENTIMENT)
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@economic_bp.route('/api/economic/indicators', methods=['GET'])
def get_indicators():
    """Get all economic indicators with current values"""
    try:
        # Get live Treasury yields
        treasury = get_treasury_yields()
        
        # Combine static and live data
        indicators = {
            'static': STATIC_INDICATORS,
            'treasury': treasury,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(indicators)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@economic_bp.route('/api/economic/yield-curve', methods=['GET'])
def get_yield_curve():
    """Get yield curve data including history"""
    try:
        current = get_treasury_yields()
        history = get_yield_curve_history(365)
        
        # Build yield curve points
        yields = current.get('yields', {})
        curve_points = []
        
        maturities = ['3M', '2Y', '5Y', '10Y', '30Y']
        for maturity in maturities:
            if maturity in yields:
                curve_points.append({
                    'maturity': maturity,
                    'yield': yields[maturity]
                })
        
        # Determine curve shape
        if yields.get('curve_inverted'):
            shape = 'Inverted'
            shape_explanation = 'Short-term rates exceed long-term rates, often signaling recession risk.'
        elif yields.get('10Y_3M_spread', 0) < 0.5:
            shape = 'Flat'
            shape_explanation = 'Minimal difference between short and long rates, suggesting uncertainty.'
        else:
            shape = 'Normal'
            shape_explanation = 'Long-term rates exceed short-term rates, indicating economic optimism.'
        
        return jsonify({
            'current_curve': curve_points,
            'spreads': {
                '10Y_3M': yields.get('10Y_3M_spread'),
                '10Y_2Y': yields.get('10Y_2Y_spread'),
            },
            'shape': shape,
            'shape_explanation': shape_explanation,
            'inverted': yields.get('curve_inverted', False),
            'history': history.get('history', {}),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@economic_bp.route('/api/economic/correlation', methods=['GET'])
def get_correlation():
    """Get market correlation with economic indicators"""
    try:
        correlation = get_market_correlation()
        
        return jsonify(correlation)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@economic_bp.route('/api/economic/explainers', methods=['GET'])
def get_explainers():
    """Get educational explanations for all indicators"""
    try:
        explainers = [
            {
                'indicator': 'Federal Funds Rate',
                'what_it_is': 'The interest rate banks charge each other for overnight loans.',
                'why_it_matters': 'It influences all other interest rates in the economy - mortgages, car loans, credit cards, and corporate borrowing.',
                'market_impact': 'Higher rates make borrowing expensive, slowing the economy. This typically hurts growth stocks more than value stocks.',
                'current_context': f"The Fed has kept rates elevated at {STATIC_INDICATORS['fed_funds_rate']['value']}% to combat inflation."
            },
            {
                'indicator': 'Inflation (CPI)',
                'what_it_is': 'The rate at which prices for goods and services are rising.',
                'why_it_matters': 'High inflation erodes purchasing power. The Fed targets 2% as the sweet spot for a healthy economy.',
                'market_impact': 'High inflation leads to higher rates, hurting stock valuations. Companies that can raise prices (pricing power) fare better.',
                'current_context': f"Inflation has moderated to {STATIC_INDICATORS['inflation_rate']['value']}%, closer to the Fed's target."
            },
            {
                'indicator': 'Unemployment Rate',
                'what_it_is': 'The percentage of people in the workforce who are jobless and looking for work.',
                'why_it_matters': 'Low unemployment means a strong economy but can fuel inflation through wage growth.',
                'market_impact': 'Rising unemployment can signal recession, often leading to market volatility followed by Fed rate cuts.',
                'current_context': f"Unemployment at {STATIC_INDICATORS['unemployment_rate']['value']}% is near historical lows."
            },
            {
                'indicator': 'GDP Growth',
                'what_it_is': 'The total value of goods and services produced, measuring economic output.',
                'why_it_matters': 'Two consecutive quarters of negative GDP is a recession. Strong growth supports corporate earnings.',
                'market_impact': 'Strong GDP growth typically lifts corporate profits and stock prices, though very high growth can trigger inflation fears.',
                'current_context': f"GDP growth of {STATIC_INDICATORS['gdp_growth']['value']}% indicates a resilient economy."
            },
            {
                'indicator': 'Yield Curve',
                'what_it_is': 'The difference between long-term and short-term interest rates.',
                'why_it_matters': 'An inverted curve (short rates > long rates) has historically preceded recessions.',
                'market_impact': 'Inversion often signals investors expect economic weakness ahead. Banks profit less when the curve is flat or inverted.',
                'current_context': 'Monitor the 10-year vs 3-month Treasury spread for inversion signals.'
            }
        ]
        
        return jsonify({
            'explainers': explainers,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
