from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from cache import financial_cache, Cache

dividends_bp = Blueprint('dividends', __name__)


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


def get_dividend_data(ticker):
    """Get comprehensive dividend data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get dividend history
        dividends = stock.dividends
        
        data = {
            'ticker': ticker.upper(),
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown'),
            'price': safe_round(info.get('currentPrice') or info.get('regularMarketPrice'), 2),
            
            # Current dividend info
            'dividend_rate': safe_round(info.get('dividendRate'), 2),
            'dividend_yield': safe_round((info.get('dividendYield') or 0) * 100, 2),
            'payout_ratio': safe_round((info.get('payoutRatio') or 0) * 100, 2),
            'ex_dividend_date': None,
            
            # Historical data
            'dividend_history': [],
            'annual_dividend': 0,
            'dividend_growth_5y': None,
        }
        
        # Ex-dividend date
        ex_div = info.get('exDividendDate')
        if ex_div:
            data['ex_dividend_date'] = datetime.fromtimestamp(ex_div).strftime('%Y-%m-%d')
            data['days_until_ex_div'] = (datetime.fromtimestamp(ex_div) - datetime.now()).days
        
        # Process dividend history
        if not dividends.empty:
            # Last 3 years of dividends
            three_years_ago = datetime.now() - timedelta(days=365*3)
            recent_divs = dividends[dividends.index >= three_years_ago]
            
            for date, amount in recent_divs.items():
                data['dividend_history'].append({
                    'date': date.strftime('%Y-%m-%d'),
                    'amount': safe_round(amount, 4)
                })
            
            # Annual dividend calculation
            last_year = datetime.now() - timedelta(days=365)
            annual_divs = dividends[dividends.index >= last_year]
            data['annual_dividend'] = safe_round(annual_divs.sum(), 2)
            
            # 5-year dividend growth
            if len(dividends) >= 20:  # At least 5 years of quarterly dividends
                five_years_ago = datetime.now() - timedelta(days=365*5)
                old_divs = dividends[dividends.index < five_years_ago + timedelta(days=365)]
                recent_divs_5y = dividends[dividends.index >= datetime.now() - timedelta(days=365)]
                
                if not old_divs.empty and not recent_divs_5y.empty:
                    old_annual = old_divs.sum()
                    new_annual = recent_divs_5y.sum()
                    if old_annual > 0:
                        growth = ((new_annual / old_annual) ** (1/5) - 1) * 100
                        data['dividend_growth_5y'] = safe_round(growth, 2)
            
            # Next expected dividend
            if len(dividends) >= 4:
                avg_dividend = dividends.tail(4).mean()
                data['expected_next_dividend'] = safe_round(avg_dividend, 4)
        
        # Dividend safety score (simple calculation)
        payout = data['payout_ratio'] or 0
        if payout > 0:
            if payout < 50:
                data['dividend_safety'] = 'Safe'
                data['safety_score'] = 5
            elif payout < 70:
                data['dividend_safety'] = 'Moderate'
                data['safety_score'] = 3
            elif payout < 90:
                data['dividend_safety'] = 'Caution'
                data['safety_score'] = 2
            else:
                data['dividend_safety'] = 'At Risk'
                data['safety_score'] = 1
        
        return data
        
    except:
        return None


def calculate_drip_projection(initial_investment, annual_dividend_yield, years, annual_price_growth=0.05):
    """Calculate DRIP projection"""
    projections = []
    
    shares = initial_investment / 100  # Assume $100 starting price
    price = 100
    total_dividends = 0
    
    for year in range(years + 1):
        portfolio_value = shares * price
        annual_dividend = portfolio_value * (annual_dividend_yield / 100)
        
        projections.append({
            'year': year,
            'shares': safe_round(shares, 2),
            'share_price': safe_round(price, 2),
            'portfolio_value': safe_round(portfolio_value, 2),
            'annual_dividend': safe_round(annual_dividend, 2),
            'total_dividends_received': safe_round(total_dividends, 2),
        })
        
        if year < years:
            # Reinvest dividends quarterly
            quarterly_div = annual_dividend / 4
            for _ in range(4):
                # Buy new shares
                new_shares = quarterly_div / price
                shares += new_shares
                total_dividends += quarterly_div
                # Price appreciation (distributed quarterly)
                price *= (1 + annual_price_growth / 4)
    
    return projections


# =============================================================================
# API ENDPOINTS
# =============================================================================

@dividends_bp.route('/api/dividends/calendar', methods=['GET'])
def get_dividend_calendar():
    """Get upcoming dividends for specified tickers"""
    try:
        tickers_param = request.args.get('tickers', '')
        tickers = [t.strip().upper() for t in tickers_param.split(',') if t.strip()]
        
        if not tickers:
            return jsonify({
                'error': 'No tickers provided. Use ?tickers=AAPL,MSFT',
                'error_type': 'validation'
            }), 400
        
        dividends = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_dividend_data, ticker): ticker for ticker in tickers}
            
            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data and data.get('ex_dividend_date'):
                        dividends.append({
                            'ticker': data['ticker'],
                            'name': data['name'],
                            'ex_date': data['ex_dividend_date'],
                            'days_until': data.get('days_until_ex_div'),
                            'amount': data.get('expected_next_dividend'),
                            'yield': data['dividend_yield'],
                        })
                except:
                    continue
        
        # Sort by ex-date
        dividends.sort(key=lambda x: x.get('ex_date', '9999-12-31'))
        
        # Filter to upcoming (next 60 days)
        today = datetime.now()
        upcoming = [d for d in dividends if d.get('days_until') and d['days_until'] >= -1]
        
        return jsonify({
            'upcoming_dividends': upcoming,
            'count': len(upcoming),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@dividends_bp.route('/api/dividends/analysis/<ticker>', methods=['GET'])
def analyze_dividend(ticker):
    """Get comprehensive dividend analysis for a ticker"""
    try:
        ticker = ticker.strip().upper()
        data = get_dividend_data(ticker)
        
        if data is None:
            return jsonify({
                'error': f'Could not fetch dividend data for {ticker}',
                'error_type': 'data_error'
            }), 400
        
        data['timestamp'] = datetime.now().isoformat()
        return jsonify(data)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@dividends_bp.route('/api/dividends/drip-calculator', methods=['POST'])
def drip_calculator():
    """Calculate DRIP projection"""
    try:
        data = request.get_json()
        initial_investment = float(data.get('initial_investment', 10000))
        dividend_yield = float(data.get('dividend_yield', 3.0))
        years = min(int(data.get('years', 20)), 50)
        price_growth = float(data.get('price_growth', 5.0)) / 100
        
        projections = calculate_drip_projection(
            initial_investment,
            dividend_yield,
            years,
            price_growth
        )
        
        final = projections[-1]
        
        return jsonify({
            'projections': projections,
            'summary': {
                'initial_investment': initial_investment,
                'final_value': final['portfolio_value'],
                'total_return': safe_round(
                    ((final['portfolio_value'] - initial_investment) / initial_investment) * 100, 2
                ),
                'total_dividends': final['total_dividends_received'],
                'final_annual_dividend': final['annual_dividend'],
                'years': years,
                'dividend_yield': dividend_yield,
                'price_growth': price_growth * 100
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@dividends_bp.route('/api/dividends/income-projection', methods=['POST'])
def project_dividend_income():
    """Project dividend income from a portfolio"""
    try:
        data = request.get_json()
        positions = data.get('positions', [])
        
        if not positions:
            return jsonify({
                'error': 'No positions provided',
                'error_type': 'validation'
            }), 400
        
        total_annual_income = 0
        position_details = []
        
        for pos in positions:
            ticker = pos.get('ticker', '').upper()
            shares = float(pos.get('shares', 0))
            
            if not ticker or shares <= 0:
                continue
            
            div_data = get_dividend_data(ticker)
            if div_data and div_data.get('annual_dividend'):
                annual_income = div_data['annual_dividend'] * shares
                total_annual_income += annual_income
                
                position_details.append({
                    'ticker': ticker,
                    'name': div_data['name'],
                    'shares': shares,
                    'annual_dividend_per_share': div_data['annual_dividend'],
                    'annual_income': safe_round(annual_income, 2),
                    'yield': div_data['dividend_yield'],
                    'ex_date': div_data.get('ex_dividend_date'),
                    'safety': div_data.get('dividend_safety'),
                })
        
        # Sort by income contribution
        position_details.sort(key=lambda x: x.get('annual_income', 0), reverse=True)
        
        return jsonify({
            'positions': position_details,
            'summary': {
                'total_annual_income': safe_round(total_annual_income, 2),
                'monthly_income': safe_round(total_annual_income / 12, 2),
                'quarterly_income': safe_round(total_annual_income / 4, 2),
                'positions_with_dividends': len(position_details),
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
