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
        previous_close = clean_value(info.get('previousClose')) or clean_value(info.get('regularMarketPreviousClose'))
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
            'previous_close': previous_close,
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
        # Accept total cost basis and calculate per-share cost basis
        total_cost_basis = float(pos.get('total_cost_basis', 0))
        cost_basis_per_share = (total_cost_basis / shares) if shares > 0 else 0
        
        stock_data = get_stock_data(ticker)
        if 'error' in stock_data:
            continue
        
        current_price = stock_data.get('current_price', 0)
        if current_price is None:
            current_price = 0
        
        current_value = shares * current_price
        cost_basis_total = total_cost_basis
        gain_loss = current_value - cost_basis_total
        gain_loss_percent = (gain_loss / cost_basis_total * 100) if cost_basis_total > 0 else 0
        
        total_value += current_value
        total_cost += cost_basis_total
        
        positions_data.append({
            'ticker': ticker,
            'name': stock_data.get('name', ticker),
            'shares': shares,
            'cost_basis': cost_basis_per_share,  # Per share for display
            'cost_basis_total': cost_basis_total,  # Total for calculations
            'current_price': current_price,
            'current_value': current_value,
            'gain_loss': gain_loss,
            'gain_loss_percent': gain_loss_percent,
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
    previous_close = stock_data.get('previous_close')
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
        'previous_close': safe_round(previous_close, 2) if previous_close else None,
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


# =============================================================================
# ADVANCED PORTFOLIO ANALYTICS
# =============================================================================

def get_price_history(tickers, period='1y'):
    """Get historical price data for multiple tickers"""
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                prices[ticker] = hist['Close']
        except:
            continue
    
    if not prices:
        return None
    
    # Align dates
    df = pd.DataFrame(prices)
    df = df.dropna()
    return df


def calculate_correlation_matrix(tickers):
    """Calculate correlation matrix between holdings"""
    prices = get_price_history(tickers)
    
    if prices is None or prices.empty:
        return None
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate correlation
    corr_matrix = returns.corr()
    
    # Convert to serializable format
    result = {
        'tickers': list(corr_matrix.columns),
        'matrix': []
    }
    
    for i, row_ticker in enumerate(corr_matrix.index):
        row = []
        for j, col_ticker in enumerate(corr_matrix.columns):
            value = corr_matrix.iloc[i, j]
            row.append(safe_round(value, 3) if not np.isnan(value) else 0)
        result['matrix'].append(row)
    
    return result


def run_monte_carlo_simulation(positions, num_simulations=1000, time_horizon=252):
    """Run Monte Carlo simulation for portfolio"""
    if not positions:
        return None
    
    tickers = [p.get('ticker', '').upper() for p in positions]
    weights = []
    
    # Get prices and calculate weights
    prices = get_price_history(tickers)
    if prices is None or prices.empty:
        return None
    
    # Calculate portfolio weights based on current values
    total_value = 0
    position_values = {}
    
    for pos in positions:
        ticker = pos.get('ticker', '').upper()
        shares = float(pos.get('shares', 0))
        if ticker in prices.columns:
            current_price = prices[ticker].iloc[-1]
            value = shares * current_price
            position_values[ticker] = value
            total_value += value
    
    if total_value == 0:
        return None
    
    # Calculate weights
    weights = np.array([position_values.get(t, 0) / total_value for t in tickers if t in prices.columns])
    valid_tickers = [t for t in tickers if t in prices.columns]
    
    # Calculate returns
    returns = prices[valid_tickers].pct_change().dropna()
    
    # Calculate mean and covariance
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    # Run simulations
    np.random.seed(42)  # For reproducibility
    simulated_values = []
    
    for _ in range(num_simulations):
        # Generate random returns using multivariate normal
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(random_returns, weights)
        
        # Calculate cumulative value
        cumulative = total_value * np.cumprod(1 + portfolio_returns)
        simulated_values.append(cumulative)
    
    simulated_values = np.array(simulated_values)
    
    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = {}
    
    for p in percentiles:
        values = np.percentile(simulated_values, p, axis=0)
        percentile_values[f'p{p}'] = [safe_round(v, 2) for v in values[::21]]  # Monthly samples
    
    # Final values statistics
    final_values = simulated_values[:, -1]
    
    return {
        'initial_value': safe_round(total_value, 2),
        'simulations': num_simulations,
        'time_horizon_days': time_horizon,
        'percentile_paths': percentile_values,
        'final_value_stats': {
            'mean': safe_round(np.mean(final_values), 2),
            'median': safe_round(np.median(final_values), 2),
            'std': safe_round(np.std(final_values), 2),
            'min': safe_round(np.min(final_values), 2),
            'max': safe_round(np.max(final_values), 2),
            'p5': safe_round(np.percentile(final_values, 5), 2),
            'p95': safe_round(np.percentile(final_values, 95), 2),
        },
        'expected_return': safe_round((np.mean(final_values) / total_value - 1) * 100, 2),
        'worst_case_return': safe_round((np.percentile(final_values, 5) / total_value - 1) * 100, 2),
        'best_case_return': safe_round((np.percentile(final_values, 95) / total_value - 1) * 100, 2),
    }


def calculate_var_cvar(positions, confidence=0.95, time_horizon=1):
    """Calculate Value at Risk and Conditional VaR"""
    if not positions:
        return None
    
    tickers = [p.get('ticker', '').upper() for p in positions]
    prices = get_price_history(tickers)
    
    if prices is None or prices.empty:
        return None
    
    # Calculate weights
    total_value = 0
    position_values = {}
    
    for pos in positions:
        ticker = pos.get('ticker', '').upper()
        shares = float(pos.get('shares', 0))
        if ticker in prices.columns:
            current_price = prices[ticker].iloc[-1]
            value = shares * current_price
            position_values[ticker] = value
            total_value += value
    
    if total_value == 0:
        return None
    
    valid_tickers = [t for t in tickers if t in prices.columns]
    weights = np.array([position_values.get(t, 0) / total_value for t in valid_tickers])
    
    # Calculate portfolio returns
    returns = prices[valid_tickers].pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    
    # Calculate VaR
    var_pct = np.percentile(portfolio_returns, (1 - confidence) * 100)
    var_dollar = total_value * var_pct * np.sqrt(time_horizon)
    
    # Calculate CVaR (Expected Shortfall)
    cvar_returns = portfolio_returns[portfolio_returns <= var_pct]
    cvar_pct = cvar_returns.mean() if len(cvar_returns) > 0 else var_pct
    cvar_dollar = total_value * cvar_pct * np.sqrt(time_horizon)
    
    return {
        'portfolio_value': safe_round(total_value, 2),
        'confidence_level': confidence,
        'time_horizon_days': time_horizon,
        'var': {
            'percentage': safe_round(var_pct * 100, 2),
            'dollar_amount': safe_round(abs(var_dollar), 2),
            'interpretation': f'There is a {(1-confidence)*100:.0f}% chance of losing more than ${abs(var_dollar):,.2f} in {time_horizon} day(s)'
        },
        'cvar': {
            'percentage': safe_round(cvar_pct * 100, 2),
            'dollar_amount': safe_round(abs(cvar_dollar), 2),
            'interpretation': f'If losses exceed VaR, the expected loss is ${abs(cvar_dollar):,.2f}'
        }
    }


def generate_optimization_recommendations(positions):
    """Generate portfolio optimization suggestions"""
    if not positions:
        return None
    
    recommendations = []
    
    # Analyze concentration
    total_value = sum(p.get('current_value', 0) for p in positions)
    if total_value == 0:
        return None
    
    for pos in positions:
        weight = pos.get('current_value', 0) / total_value
        ticker = pos.get('ticker', '')
        
        # Over-concentration warning
        if weight > 0.25:
            recommendations.append({
                'type': 'rebalance',
                'priority': 'high',
                'ticker': ticker,
                'message': f'{ticker} represents {weight*100:.1f}% of portfolio. Consider reducing to below 25%.',
                'current_weight': safe_round(weight * 100, 1),
                'suggested_weight': 25.0
            })
        elif weight > 0.15:
            recommendations.append({
                'type': 'rebalance',
                'priority': 'medium',
                'ticker': ticker,
                'message': f'{ticker} at {weight*100:.1f}% is above recommended 15% single-stock limit.',
                'current_weight': safe_round(weight * 100, 1),
                'suggested_weight': 15.0
            })
    
    # Sector concentration
    sector_weights = {}
    for pos in positions:
        sector = pos.get('sector', 'Unknown')
        sector_weights[sector] = sector_weights.get(sector, 0) + pos.get('current_value', 0) / total_value
    
    for sector, weight in sector_weights.items():
        if weight > 0.40:
            recommendations.append({
                'type': 'diversify',
                'priority': 'high',
                'sector': sector,
                'message': f'{sector} sector at {weight*100:.1f}% of portfolio. Consider diversifying.',
                'current_weight': safe_round(weight * 100, 1),
                'suggested_weight': 40.0
            })
    
    # Low number of holdings
    if len(positions) < 5:
        recommendations.append({
            'type': 'diversify',
            'priority': 'medium',
            'message': f'Portfolio has only {len(positions)} holdings. Consider adding more for diversification.',
            'current_count': len(positions),
            'suggested_count': 10
        })
    
    return {
        'recommendations': recommendations,
        'diversification_score': calculate_portfolio_risk_metrics(positions).get('diversification_score', 0),
        'positions_count': len(positions),
        'sectors_count': len(sector_weights)
    }


def simulate_what_if(current_positions, trades):
    """Simulate what-if portfolio changes"""
    if not current_positions:
        return None
    
    # Deep copy positions
    simulated_positions = []
    for pos in current_positions:
        simulated_positions.append(dict(pos))
    
    # Apply trades
    for trade in trades:
        action = trade.get('action')  # 'buy' or 'sell'
        ticker = trade.get('ticker', '').upper()
        shares = float(trade.get('shares', 0))
        price = float(trade.get('price', 0))
        
        if action == 'buy':
            # Check if position exists
            existing = next((p for p in simulated_positions if p['ticker'] == ticker), None)
            if existing:
                # Add to existing
                old_value = existing['shares'] * existing.get('cost_basis', existing.get('current_price', price))
                new_value = shares * price
                total_shares = existing['shares'] + shares
                existing['shares'] = total_shares
                existing['cost_basis'] = (old_value + new_value) / total_shares
                existing['cost_basis_total'] = existing['shares'] * existing['cost_basis']
            else:
                # New position
                stock_data = get_stock_data(ticker)
                simulated_positions.append({
                    'ticker': ticker,
                    'shares': shares,
                    'cost_basis': price,
                    'cost_basis_total': shares * price,
                    'current_price': stock_data.get('current_price', price) if 'error' not in stock_data else price,
                    'sector': stock_data.get('sector', 'Unknown') if 'error' not in stock_data else 'Unknown'
                })
        
        elif action == 'sell':
            existing = next((p for p in simulated_positions if p['ticker'] == ticker), None)
            if existing:
                if shares >= existing['shares']:
                    # Remove position
                    simulated_positions = [p for p in simulated_positions if p['ticker'] != ticker]
                else:
                    existing['shares'] -= shares
                    existing['cost_basis_total'] = existing['shares'] * existing.get('cost_basis', 0)
    
    # Calculate metrics for both portfolios
    current_metrics = calculate_portfolio_metrics([
        {'ticker': p['ticker'], 'shares': p['shares'], 'total_cost_basis': p.get('cost_basis_total', p.get('shares', 0) * p.get('cost_basis', 0))}
        for p in current_positions
    ])
    
    simulated_metrics = calculate_portfolio_metrics([
        {'ticker': p['ticker'], 'shares': p['shares'], 'total_cost_basis': p.get('cost_basis_total', p.get('shares', 0) * p.get('cost_basis', 0))}
        for p in simulated_positions
    ])
    
    current_risk = calculate_portfolio_risk_metrics(current_metrics.get('positions', []))
    simulated_risk = calculate_portfolio_risk_metrics(simulated_metrics.get('positions', []))
    
    return {
        'current': {
            'total_value': current_metrics.get('total_value', 0),
            'total_positions': current_metrics.get('positions_count', 0),
            'portfolio_beta': current_risk.get('portfolio_beta', 1.0),
            'diversification_score': current_risk.get('diversification_score', 0),
            'concentration_risk': current_risk.get('concentration_risk', 'N/A')
        },
        'simulated': {
            'total_value': simulated_metrics.get('total_value', 0),
            'total_positions': simulated_metrics.get('positions_count', 0),
            'portfolio_beta': simulated_risk.get('portfolio_beta', 1.0),
            'diversification_score': simulated_risk.get('diversification_score', 0),
            'concentration_risk': simulated_risk.get('concentration_risk', 'N/A')
        },
        'changes': {
            'value_change': safe_round(simulated_metrics.get('total_value', 0) - current_metrics.get('total_value', 0), 2),
            'beta_change': safe_round(simulated_risk.get('portfolio_beta', 1.0) - current_risk.get('portfolio_beta', 1.0), 2),
            'diversification_change': safe_round(simulated_risk.get('diversification_score', 0) - current_risk.get('diversification_score', 0), 1),
        },
        'trades_applied': trades,
        'simulated_positions': simulated_metrics.get('positions', [])
    }


# =============================================================================
# ADVANCED PORTFOLIO ENDPOINTS
# =============================================================================

@portfolio_bp.route('/api/portfolio/correlation', methods=['POST'])
def get_correlation():
    """Get correlation matrix for portfolio holdings"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({
                'error': 'No tickers provided',
                'error_type': 'validation'
            }), 400
        
        tickers = [t.upper() for t in tickers]
        result = calculate_correlation_matrix(tickers)
        
        if result is None:
            return jsonify({
                'error': 'Unable to calculate correlation matrix',
                'error_type': 'data_error'
            }), 400
        
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@portfolio_bp.route('/api/portfolio/monte-carlo', methods=['POST'])
def monte_carlo():
    """Run Monte Carlo simulation"""
    try:
        data = request.get_json()
        positions = data.get('positions', [])
        num_simulations = min(data.get('simulations', 1000), 2000)  # Cap at 2000
        time_horizon = min(data.get('time_horizon', 252), 504)  # Cap at 2 years
        
        if not positions:
            return jsonify({
                'error': 'No positions provided',
                'error_type': 'validation'
            }), 400
        
        result = run_monte_carlo_simulation(positions, num_simulations, time_horizon)
        
        if result is None:
            return jsonify({
                'error': 'Unable to run Monte Carlo simulation',
                'error_type': 'data_error'
            }), 400
        
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@portfolio_bp.route('/api/portfolio/var', methods=['POST'])
def get_var():
    """Calculate Value at Risk and CVaR"""
    try:
        data = request.get_json()
        positions = data.get('positions', [])
        confidence = data.get('confidence', 0.95)
        time_horizon = data.get('time_horizon', 1)
        
        if not positions:
            return jsonify({
                'error': 'No positions provided',
                'error_type': 'validation'
            }), 400
        
        # Get position data
        portfolio_metrics = calculate_portfolio_metrics(positions)
        positions_data = portfolio_metrics.get('positions', [])
        
        result = calculate_var_cvar(positions_data, confidence, time_horizon)
        
        if result is None:
            return jsonify({
                'error': 'Unable to calculate VaR',
                'error_type': 'data_error'
            }), 400
        
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@portfolio_bp.route('/api/portfolio/optimize', methods=['POST'])
def optimize():
    """Get portfolio optimization recommendations"""
    try:
        data = request.get_json()
        positions = data.get('positions', [])
        
        if not positions:
            return jsonify({
                'error': 'No positions provided',
                'error_type': 'validation'
            }), 400
        
        # Get position data
        portfolio_metrics = calculate_portfolio_metrics(positions)
        positions_data = portfolio_metrics.get('positions', [])
        
        result = generate_optimization_recommendations(positions_data)
        
        if result is None:
            return jsonify({
                'error': 'Unable to generate recommendations',
                'error_type': 'data_error'
            }), 400
        
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@portfolio_bp.route('/api/portfolio/what-if', methods=['POST'])
def what_if():
    """Simulate what-if portfolio changes"""
    try:
        data = request.get_json()
        positions = data.get('positions', [])
        trades = data.get('trades', [])
        
        if not positions:
            return jsonify({
                'error': 'No positions provided',
                'error_type': 'validation'
            }), 400
        
        if not trades:
            return jsonify({
                'error': 'No trades provided',
                'error_type': 'validation'
            }), 400
        
        # Get position data
        portfolio_metrics = calculate_portfolio_metrics(positions)
        positions_data = portfolio_metrics.get('positions', [])
        
        result = simulate_what_if(positions_data, trades)
        
        if result is None:
            return jsonify({
                'error': 'Unable to simulate trades',
                'error_type': 'data_error'
            }), 400
        
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


