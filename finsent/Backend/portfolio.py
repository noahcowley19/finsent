from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
from cache import financial_cache, Cache

portfolio_bp = Blueprint('portfolio', __name__)


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
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


def format_large_number(value):
    if value is None:
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"${value / 1e3:.1f}K"
    else:
        return f"${value:,.2f}"


def get_stock_data(ticker):
    """Fetch stock data with caching"""
    cached = financial_cache.get(ticker, 'portfolio_data')
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info.get('shortName') and not info.get('longName'):
            return {
                'error': 'Ticker not recognized. Please enter a valid stock symbol.',
                'error_type': 'invalid_ticker'
            }
        
        current_price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
        beta = clean_value(info.get('beta')) or 1.0
        sector = info.get('sector') or 'Unknown'
        industry = info.get('industry') or 'Unknown'
        market_cap = clean_value(info.get('marketCap'))
        
        # Get historical data for calculations
        hist = stock.history(period='1y', interval='1d')
        
        data = {
            'ticker': ticker.upper(),
            'name': info.get('longName') or info.get('shortName') or ticker,
            'current_price': current_price,
            'beta': beta,
            'sector': sector,
            'industry': industry,
            'market_cap': market_cap,
            'historical': hist,
            'info': info,
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set(ticker, 'portfolio_data', data, Cache.TTL_PRICE)
        return data
        
    except Exception as e:
        return {
            'error': f'Unable to fetch data for {ticker}. Please try again.',
            'error_type': 'fetch_error',
            'details': str(e)
        }


def calculate_capm(beta, risk_free_rate=0.02, market_return=0.10):
    """
    Calculate CAPM expected return: E(R) = Rf + β(Rm - Rf)
    Default: Rf = 2% (10-year Treasury), Rm = 10% (S&P 500 historical)
    """
    if beta is None:
        beta = 1.0
    
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    return {
        'expected_return': safe_round(expected_return * 100, 2),
        'beta': safe_round(beta, 2),
        'risk_free_rate': safe_round(risk_free_rate * 100, 2),
        'market_return': safe_round(market_return * 100, 2),
        'risk_premium': safe_round((market_return - risk_free_rate) * 100, 2)
    }


def calculate_stock_returns(historical_data):
    """Calculate stock returns from historical data"""
    if historical_data is None or historical_data.empty:
        return None
    
    prices = historical_data['Close'].values
    returns = np.diff(prices) / prices[:-1]
    return returns


def calculate_volatility(returns):
    """Calculate annualized volatility"""
    if returns is None or len(returns) == 0:
        return None
    
    std_dev = np.std(returns)
    annualized_vol = std_dev * np.sqrt(252)  # 252 trading days
    return safe_round(annualized_vol * 100, 2)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    if returns is None or len(returns) == 0:
        return None
    
    excess_returns = returns - (risk_free_rate / 252)
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    return safe_round(sharpe, 2)


def calculate_portfolio_metrics(positions):
    """Calculate comprehensive portfolio metrics"""
    if not positions:
        return {
            'total_value': 0,
            'total_cost': 0,
            'total_gain_loss': 0,
            'total_gain_loss_percent': 0,
            'positions_count': 0
        }
    
    total_value = 0
    total_cost = 0
    positions_data = []
    
    for pos in positions:
        ticker = pos.get('ticker', '').upper()
        shares = float(pos.get('shares', 0))
        cost_basis = float(pos.get('cost_basis', 0))
        purchase_date = pos.get('purchase_date', '')
        
        stock_data = get_stock_data(ticker)
        if 'error' in stock_data:
            continue
        
        current_price = stock_data.get('current_price', 0)
        if current_price is None:
            current_price = 0
        
        current_value = shares * current_price
        cost_basis_total = shares * cost_basis
        gain_loss = current_value - cost_basis_total
        gain_loss_percent = (gain_loss / cost_basis_total * 100) if cost_basis_total > 0 else 0
        
        total_value += current_value
        total_cost += cost_basis_total
        
        positions_data.append({
            'ticker': ticker,
            'name': stock_data.get('name', ticker),
            'shares': shares,
            'cost_basis': cost_basis,
            'current_price': current_price,
            'current_value': current_value,
            'cost_basis_total': cost_basis_total,
            'gain_loss': gain_loss,
            'gain_loss_percent': gain_loss_percent,
            'purchase_date': purchase_date,
            'beta': stock_data.get('beta', 1.0),
            'sector': stock_data.get('sector', 'Unknown'),
            'industry': stock_data.get('industry', 'Unknown')
        })
    
    total_gain_loss = total_value - total_cost
    total_gain_loss_percent = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'total_value': safe_round(total_value, 2),
        'total_cost': safe_round(total_cost, 2),
        'total_gain_loss': safe_round(total_gain_loss, 2),
        'total_gain_loss_percent': safe_round(total_gain_loss_percent, 2),
        'positions_count': len(positions_data),
        'positions': positions_data
    }


def calculate_portfolio_allocation(positions_data):
    """Calculate sector and industry allocation"""
    sector_allocation = {}
    industry_allocation = {}
    ticker_allocation = {}
    
    total_value = sum(pos.get('current_value', 0) for pos in positions_data)
    
    if total_value == 0:
        return {
            'sector': [],
            'industry': [],
            'ticker': []
        }
    
    for pos in positions_data:
        ticker = pos.get('ticker', '')
        value = pos.get('current_value', 0)
        sector = pos.get('sector', 'Unknown')
        industry = pos.get('industry', 'Unknown')
        
        # Sector allocation
        if sector not in sector_allocation:
            sector_allocation[sector] = 0
        sector_allocation[sector] += value
        
        # Industry allocation
        if industry not in industry_allocation:
            industry_allocation[industry] = 0
        industry_allocation[industry] += value
        
        # Ticker allocation
        ticker_allocation[ticker] = value
    
    # Convert to percentages and format
    sector_list = [
        {
            'name': sector,
            'value': safe_round(value, 2),
            'percentage': safe_round((value / total_value) * 100, 2)
        }
        for sector, value in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True)
    ]
    
    industry_list = [
        {
            'name': industry,
            'value': safe_round(value, 2),
            'percentage': safe_round((value / total_value) * 100, 2)
        }
        for industry, value in sorted(industry_allocation.items(), key=lambda x: x[1], reverse=True)
    ]
    
    ticker_list = [
        {
            'ticker': ticker,
            'value': safe_round(value, 2),
            'percentage': safe_round((value / total_value) * 100, 2)
        }
        for ticker, value in sorted(ticker_allocation.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return {
        'sector': sector_list,
        'industry': industry_list,
        'ticker': ticker_list
    }


def calculate_portfolio_beta(positions_data):
    """Calculate weighted portfolio beta"""
    total_value = sum(pos.get('current_value', 0) for pos in positions_data)
    
    if total_value == 0:
        return 1.0
    
    weighted_beta = sum(
        (pos.get('current_value', 0) / total_value) * pos.get('beta', 1.0)
        for pos in positions_data
    )
    
    return safe_round(weighted_beta, 2)


def calculate_portfolio_risk_metrics(positions_data, risk_free_rate=0.02):
    """Calculate portfolio risk metrics"""
    if not positions_data:
        return {
            'portfolio_beta': 1.0,
            'diversification_score': 0,
            'concentration_risk': 'High'
        }
    
    portfolio_beta = calculate_portfolio_beta(positions_data)
    
    # Calculate diversification score (0-100)
    # Based on number of positions and sector diversity
    num_positions = len(positions_data)
    sectors = set(pos.get('sector', 'Unknown') for pos in positions_data)
    num_sectors = len(sectors)
    
    # Diversification score: positions (50%) + sectors (50%)
    position_score = min(num_positions / 20 * 50, 50)  # Max 50 points for 20+ positions
    sector_score = min(num_sectors / 10 * 50, 50)  # Max 50 points for 10+ sectors
    diversification_score = safe_round(position_score + sector_score, 1)
    
    # Concentration risk
    total_value = sum(pos.get('current_value', 0) for pos in positions_data)
    if total_value > 0:
        max_position_pct = max(
            (pos.get('current_value', 0) / total_value) * 100
            for pos in positions_data
        )
        
        if max_position_pct > 40:
            concentration_risk = 'Very High'
        elif max_position_pct > 25:
            concentration_risk = 'High'
        elif max_position_pct > 15:
            concentration_risk = 'Moderate'
        else:
            concentration_risk = 'Low'
    else:
        concentration_risk = 'N/A'
    
    return {
        'portfolio_beta': portfolio_beta,
        'diversification_score': diversification_score,
        'concentration_risk': concentration_risk,
        'num_positions': num_positions,
        'num_sectors': num_sectors,
        'max_position_pct': safe_round(max_position_pct, 2) if total_value > 0 else 0
    }


def get_individual_stock_analysis(ticker):
    """Get comprehensive analysis for a single stock"""
    stock_data = get_stock_data(ticker)
    
    if 'error' in stock_data:
        return stock_data
    
    current_price = stock_data.get('current_price', 0)
    beta = stock_data.get('beta', 1.0)
    historical = stock_data.get('historical')
    
    # Calculate returns and metrics
    returns = calculate_stock_returns(historical)
    volatility = calculate_volatility(returns)
    sharpe = calculate_sharpe_ratio(returns)
    capm = calculate_capm(beta)
    
    # Get additional metrics from info
    info = stock_data.get('info', {})
    pe = clean_value(info.get('trailingPE'))
    market_cap = clean_value(info.get('marketCap'))
    dividend_yield = clean_value(info.get('dividendYield'))
    if dividend_yield:
        dividend_yield = dividend_yield * 100
    
    return {
        'ticker': ticker.upper(),
        'name': stock_data.get('name', ticker),
        'current_price': safe_round(current_price, 2),
        'beta': safe_round(beta, 2),
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'capm': capm,
        'pe_ratio': safe_round(pe, 2) if pe else None,
        'market_cap': market_cap,
        'market_cap_display': format_large_number(market_cap),
        'dividend_yield': safe_round(dividend_yield, 2) if dividend_yield else None,
        'sector': stock_data.get('sector', 'Unknown'),
        'industry': stock_data.get('industry', 'Unknown')
    }


@portfolio_bp.route('/api/portfolio/analyze', methods=['POST'])
def analyze_portfolio():
    """Main endpoint to analyze portfolio"""
    try:
        data = request.get_json()
        positions = data.get('positions', [])
        risk_free_rate = data.get('risk_free_rate', 0.02)
        market_return = data.get('market_return', 0.10)
        
        if not positions:
            return jsonify({
                'error': 'No positions provided',
                'error_type': 'validation'
            }), 400
        
        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(positions)
        positions_data = portfolio_metrics.get('positions', [])
        
        if not positions_data:
            return jsonify({
                'error': 'Unable to fetch data for any positions',
                'error_type': 'data_error'
            }), 400
        
        # Calculate allocation
        allocation = calculate_portfolio_allocation(positions_data)
        
        # Calculate risk metrics
        risk_metrics = calculate_portfolio_risk_metrics(positions_data, risk_free_rate)
        
        # Calculate portfolio CAPM
        portfolio_beta = risk_metrics.get('portfolio_beta', 1.0)
        portfolio_capm = calculate_capm(portfolio_beta, risk_free_rate, market_return)
        
        # Individual stock analyses
        stock_analyses = []
        for pos in positions_data:
            ticker = pos.get('ticker', '')
            analysis = get_individual_stock_analysis(ticker)
            if 'error' not in analysis:
                # Add position-specific data
                analysis['shares'] = pos.get('shares', 0)
                analysis['cost_basis'] = pos.get('cost_basis', 0)
                analysis['current_value'] = pos.get('current_value', 0)
                analysis['gain_loss'] = pos.get('gain_loss', 0)
                analysis['gain_loss_percent'] = pos.get('gain_loss_percent', 0)
                stock_analyses.append(analysis)
        
        response = {
            'portfolio_metrics': portfolio_metrics,
            'allocation': allocation,
            'risk_metrics': risk_metrics,
            'portfolio_capm': portfolio_capm,
            'stock_analyses': stock_analyses,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@portfolio_bp.route('/api/portfolio/stock', methods=['POST'])
def analyze_stock():
    """Analyze a single stock"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        
        if not ticker:
            return jsonify({
                'error': 'Ticker is required',
                'error_type': 'validation'
            }), 400
        
        analysis = get_individual_stock_analysis(ticker)
        
        if 'error' in analysis:
            return jsonify(analysis), 400
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@portfolio_bp.route('/api/portfolio/capm', methods=['POST'])
def calculate_capm_endpoint():
    """Calculate CAPM for given parameters"""
    try:
        data = request.get_json()
        beta = data.get('beta', 1.0)
        risk_free_rate = data.get('risk_free_rate', 0.02)
        market_return = data.get('market_return', 0.10)
        
        capm_result = calculate_capm(beta, risk_free_rate, market_return)
        
        return jsonify(capm_result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


