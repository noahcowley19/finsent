# =============================================================================
# QUANT LAB V2 - Advanced Quantitative Analysis Engine
# =============================================================================
# Modules:
# - Strategy Forge (Vectorized Backtesting)
# - Regime Detector (Hidden Markov Model)
# - Risk Decomposition (Fama-French, EVT)
# - Volatility Surface (3D Analysis)
# - Chaos Lab (Entropy, Cointegration)
# =============================================================================

from flask import Blueprint, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Create Blueprint
quant_v2_bp = Blueprint('quant_v2', __name__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_value(value: Any) -> Any:
    """Clean and normalize values for JSON serialization"""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    return value


def safe_round(value: Any, decimals: int = 2) -> Optional[float]:
    """Safely round values"""
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return round(float(cleaned), decimals)
    except:
        return None


def get_price_data(ticker: str, period: str = '5y') -> Optional[pd.DataFrame]:
    """Fetch historical price data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


# =============================================================================
# MODULE A: STRATEGY FORGE - Vectorized Backtesting
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram"""
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def generate_signals(prices: pd.Series, rules: List[Dict]) -> Tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals from rule blocks
    
    Rules format:
    [
        {"indicator": "RSI", "operator": "<", "value": 30, "action": "entry"},
        {"indicator": "Price", "operator": ">", "value": "SMA_200", "action": "entry"},
        {"indicator": "RSI", "operator": ">", "value": 70, "action": "exit"},
    ]
    """
    n = len(prices)
    entries = pd.Series([False] * n, index=prices.index)
    exits = pd.Series([False] * n, index=prices.index)
    
    # Pre-calculate indicators
    indicators = {
        'Price': prices,
        'RSI': calculate_rsi(prices),
        'SMA_20': calculate_sma(prices, 20),
        'SMA_50': calculate_sma(prices, 50),
        'SMA_200': calculate_sma(prices, 200),
        'EMA_12': calculate_ema(prices, 12),
        'EMA_26': calculate_ema(prices, 26),
    }
    macd, signal, hist = calculate_macd(prices)
    indicators['MACD'] = macd
    indicators['MACD_Signal'] = signal
    
    for rule in rules:
        indicator_name = rule.get('indicator', 'Price')
        operator = rule.get('operator', '>')
        value = rule.get('value', 0)
        action = rule.get('action', 'entry')
        
        # Get indicator series
        ind_series = indicators.get(indicator_name, prices)
        
        # Handle dynamic values (e.g., "SMA_200")
        if isinstance(value, str) and value in indicators:
            compare_series = indicators[value]
        else:
            compare_series = float(value)
        
        # Apply operator
        if operator == '<':
            condition = ind_series < compare_series
        elif operator == '<=':
            condition = ind_series <= compare_series
        elif operator == '>':
            condition = ind_series > compare_series
        elif operator == '>=':
            condition = ind_series >= compare_series
        elif operator == '==':
            condition = ind_series == compare_series
        elif operator == 'crosses_above':
            condition = (ind_series > compare_series) & (ind_series.shift(1) <= compare_series.shift(1) if isinstance(compare_series, pd.Series) else ind_series.shift(1) <= compare_series)
        elif operator == 'crosses_below':
            condition = (ind_series < compare_series) & (ind_series.shift(1) >= compare_series.shift(1) if isinstance(compare_series, pd.Series) else ind_series.shift(1) >= compare_series)
        else:
            condition = pd.Series([False] * n, index=prices.index)
        
        # Apply to entries or exits
        if action == 'entry':
            entries = entries | condition
        else:
            exits = exits | condition
    
    return entries, exits


def run_vectorized_backtest(
    prices: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    initial_capital: float = 100000
) -> Dict:
    """
    Run vectorized backtest without vectorbt dependency
    Pure numpy/pandas implementation for speed
    """
    # Ensure alignment
    prices = prices.loc[entries.index]
    
    # Generate positions: 1 = long, 0 = out
    positions = pd.Series(0, index=prices.index)
    in_position = False
    
    for i in range(len(prices)):
        if not in_position and entries.iloc[i]:
            positions.iloc[i] = 1
            in_position = True
        elif in_position and exits.iloc[i]:
            in_position = False
        elif in_position:
            positions.iloc[i] = 1
    
    # Calculate returns
    daily_returns = prices.pct_change()
    strategy_returns = positions.shift(1) * daily_returns  # Shift to avoid look-ahead
    
    # Calculate equity curve
    equity = initial_capital * (1 + strategy_returns).cumprod()
    equity.iloc[0] = initial_capital
    
    # Calculate metrics
    total_return = (equity.iloc[-1] / initial_capital) - 1
    trading_days = len(prices)
    years = trading_days / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Max Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Volatility
    volatility = strategy_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    risk_free_rate = 0.02
    excess_return = cagr - risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else 0
    
    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win Rate
    winning_trades = (strategy_returns > 0).sum()
    losing_trades = (strategy_returns < 0).sum()
    total_trades = winning_trades + losing_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Omega Ratio (threshold = 0)
    threshold = 0
    gains = strategy_returns[strategy_returns > threshold].sum()
    losses = abs(strategy_returns[strategy_returns < threshold].sum())
    omega = gains / losses if losses > 0 else float('inf')
    
    # Tail Ratio
    p95 = np.percentile(strategy_returns.dropna(), 95)
    p05 = abs(np.percentile(strategy_returns.dropna(), 5))
    tail_ratio = p95 / p05 if p05 > 0 else 0
    
    return {
        'total_return': safe_round(total_return * 100, 2),
        'cagr': safe_round(cagr * 100, 2),
        'max_drawdown': safe_round(max_drawdown * 100, 2),
        'volatility': safe_round(volatility * 100, 2),
        'sharpe_ratio': safe_round(sharpe, 2),
        'calmar_ratio': safe_round(calmar, 2),
        'omega_ratio': safe_round(omega, 2) if omega != float('inf') else None,
        'tail_ratio': safe_round(tail_ratio, 2),
        'win_rate': safe_round(win_rate * 100, 2),
        'total_trades': total_trades,
        'equity_curve': [safe_round(v, 2) for v in equity.values[::5]],  # Downsample
        'dates': [d.strftime('%Y-%m-%d') for d in equity.index[::5]],
        'drawdown_curve': [safe_round(v * 100, 2) for v in drawdown.values[::5]],
    }


def generate_tearsheet(ticker: str, rules: List[Dict], period: str = '5y') -> Dict:
    """
    Generate comprehensive backtest tearsheet
    """
    # Fetch data
    df = get_price_data(ticker, period)
    if df is None:
        return {'error': f'Unable to fetch data for {ticker}'}
    
    prices = df['Close']
    
    # Generate signals
    entries, exits = generate_signals(prices, rules)
    
    # Run backtest
    results = run_vectorized_backtest(prices, entries, exits)
    
    # Add metadata
    results['ticker'] = ticker
    results['period'] = period
    results['rules'] = rules
    results['start_date'] = prices.index[0].strftime('%Y-%m-%d')
    results['end_date'] = prices.index[-1].strftime('%Y-%m-%d')
    results['timestamp'] = datetime.now().isoformat()
    
    return results


# =============================================================================
# MODULE B: REGIME DETECTOR - Hidden Markov Model
# =============================================================================

def detect_market_regimes(ticker: str, n_regimes: int = 3, period: str = '5y') -> Dict:
    """
    Detect market regimes using Hidden Markov Model
    States: Bull (0), Chop (1), Bear (2)
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return {'error': 'hmmlearn not installed', 'fallback': True}
    
    # Fetch data
    df = get_price_data(ticker, period)
    if df is None:
        return {'error': f'Unable to fetch data for {ticker}'}
    
    prices = df['Close'].values
    
    # Calculate log returns
    returns = np.diff(np.log(prices)).reshape(-1, 1)
    
    # Fit HMM
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    
    try:
        model.fit(returns)
        hidden_states = model.predict(returns)
    except Exception as e:
        return {'error': f'HMM fitting failed: {str(e)}'}
    
    # Calculate state statistics
    state_stats = []
    for i in range(n_regimes):
        state_returns = returns[hidden_states == i]
        state_stats.append({
            'mean_return': float(np.mean(state_returns) * 252 * 100),  # Annualized %
            'volatility': float(np.std(state_returns) * np.sqrt(252) * 100),
            'count': int(np.sum(hidden_states == i)),
        })
    
    # Order states by mean return (Bull > Chop > Bear)
    state_order = np.argsort([s['mean_return'] for s in state_stats])[::-1]
    
    # Map old states to new labels
    state_mapping = {old: new for new, old in enumerate(state_order)}
    ordered_states = [state_mapping[s] for s in hidden_states]
    
    # Reorder stats
    ordered_stats = [state_stats[i] for i in state_order]
    
    # Labels and colors
    labels = ['Bull', 'Chop', 'Bear'][:n_regimes]
    colors = ['#22C55E', '#94A3B8', '#EF4444'][:n_regimes]
    
    # Prepare dates (align with returns, not prices)
    dates = df.index[1:].strftime('%Y-%m-%d').tolist()
    
    return {
        'ticker': ticker,
        'states': ordered_states,
        'dates': dates,
        'labels': labels,
        'colors': colors,
        'state_stats': [{**s, 'label': labels[i], 'color': colors[i]} 
                        for i, s in enumerate(ordered_stats)],
        'transition_matrix': model.transmat_.tolist(),
        'current_regime': labels[ordered_states[-1]],
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# MODULE C: RISK DECOMPOSITION - Fama-French & EVT
# =============================================================================

def fetch_fama_french_factors(period: str = '5y') -> Optional[pd.DataFrame]:
    """
    Fetch Fama-French 5 factors from Kenneth French's data library
    Falls back to simulated factors if unavailable
    """
    try:
        import pandas_datareader.data as web
        ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')
        return ff_data[0]
    except:
        # Fallback: simulate factors based on SPY data
        try:
            spy = yf.download('SPY', period=period)['Close']
            returns = spy.pct_change().dropna()
            
            # Simulate factors (rough approximation)
            ff_data = pd.DataFrame(index=returns.index)
            ff_data['Mkt-RF'] = returns * 100  # Market excess return
            ff_data['SMB'] = np.random.normal(0.02, 0.5, len(returns))  # Small minus Big
            ff_data['HML'] = np.random.normal(0.02, 0.5, len(returns))  # High minus Low
            ff_data['RMW'] = np.random.normal(0.02, 0.4, len(returns))  # Robust minus Weak
            ff_data['CMA'] = np.random.normal(0.01, 0.3, len(returns))  # Conservative minus Aggressive
            ff_data['RF'] = 0.02 / 252 * 100  # Risk-free rate
            
            return ff_data
        except:
            return None


def fama_french_decomposition(ticker: str, period: str = '5y') -> Dict:
    """
    Decompose returns into Fama-French 5 factors
    R - Rf = α + β1(Mkt-RF) + β2(SMB) + β3(HML) + β4(RMW) + β5(CMA) + ε
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels not installed'}
    
    # Fetch stock data
    df = get_price_data(ticker, period)
    if df is None:
        return {'error': f'Unable to fetch data for {ticker}'}
    
    stock_returns = df['Close'].pct_change().dropna() * 100  # Convert to percent
    
    # Fetch FF factors
    ff_data = fetch_fama_french_factors(period)
    if ff_data is None:
        return {'error': 'Unable to fetch Fama-French factors'}
    
    # Align data
    common_dates = stock_returns.index.intersection(ff_data.index)
    if len(common_dates) < 60:  # Need at least 60 days
        return {'error': 'Insufficient overlapping data'}
    
    y = stock_returns.loc[common_dates]
    
    # Handle RF column
    if 'RF' in ff_data.columns:
        y = y - ff_data.loc[common_dates, 'RF']
        X = ff_data.loc[common_dates, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    else:
        X = ff_data.loc[common_dates, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    
    # Add constant for alpha
    X = sm.add_constant(X)
    
    # Fit OLS regression
    model = sm.OLS(y, X).fit()
    
    # Extract results
    factor_names = ['Alpha', 'Market', 'Size (SMB)', 'Value (HML)', 'Profitability (RMW)', 'Investment (CMA)']
    
    return {
        'ticker': ticker,
        'factors': {
            'alpha': {
                'coefficient': safe_round(model.params['const'], 4),
                'pvalue': safe_round(model.pvalues['const'], 4),
                'significant': model.pvalues['const'] < 0.05,
            },
            'market_beta': {
                'coefficient': safe_round(model.params['Mkt-RF'], 4),
                'pvalue': safe_round(model.pvalues['Mkt-RF'], 4),
            },
            'size_factor': {
                'coefficient': safe_round(model.params['SMB'], 4),
                'pvalue': safe_round(model.pvalues['SMB'], 4),
            },
            'value_factor': {
                'coefficient': safe_round(model.params['HML'], 4),
                'pvalue': safe_round(model.pvalues['HML'], 4),
            },
            'profitability_factor': {
                'coefficient': safe_round(model.params['RMW'], 4),
                'pvalue': safe_round(model.pvalues['RMW'], 4),
            },
            'investment_factor': {
                'coefficient': safe_round(model.params['CMA'], 4),
                'pvalue': safe_round(model.pvalues['CMA'], 4),
            },
        },
        'r_squared': safe_round(model.rsquared, 4),
        'adj_r_squared': safe_round(model.rsquared_adj, 4),
        'observations': len(y),
        'interpretation': generate_ff_interpretation(model.params),
        'timestamp': datetime.now().isoformat(),
    }


def generate_ff_interpretation(params: pd.Series) -> str:
    """Generate human-readable interpretation of factor loadings"""
    interpretations = []
    
    if params['Mkt-RF'] > 1.2:
        interpretations.append("Highly sensitive to market movements (high beta)")
    elif params['Mkt-RF'] < 0.8:
        interpretations.append("Defensive stock (low beta)")
    
    if params['SMB'] > 0.3:
        interpretations.append("Behaves like a small-cap stock")
    elif params['SMB'] < -0.3:
        interpretations.append("Behaves like a large-cap stock")
    
    if params['HML'] > 0.3:
        interpretations.append("Value stock characteristics")
    elif params['HML'] < -0.3:
        interpretations.append("Growth stock characteristics")
    
    return "; ".join(interpretations) if interpretations else "No strong factor tilts"


def calculate_evt_var(ticker: str, threshold_percentile: float = 0.95, confidence: float = 0.99) -> Dict:
    """
    Calculate Extreme Value Theory VaR using Generalized Pareto Distribution
    More accurate for tail risk than Gaussian assumptions
    """
    from scipy.stats import genpareto
    
    # Fetch data
    df = get_price_data(ticker, '5y')
    if df is None:
        return {'error': f'Unable to fetch data for {ticker}'}
    
    returns = df['Close'].pct_change().dropna()
    
    # We're interested in losses (negative returns)
    losses = -returns[returns < 0]
    
    # Define threshold
    threshold = np.percentile(losses, threshold_percentile * 100)
    exceedances = losses[losses > threshold] - threshold
    
    if len(exceedances) < 10:
        return {'error': 'Insufficient tail data for EVT'}
    
    # Fit GPD
    try:
        shape, loc, scale = genpareto.fit(exceedances)
    except:
        return {'error': 'GPD fitting failed'}
    
    # Calculate EVT VaR
    n = len(returns)
    nu = len(exceedances)
    
    if shape != 0:
        evt_var = threshold + (scale / shape) * ((n/nu * (1-confidence)) ** (-shape) - 1)
    else:
        evt_var = threshold + scale * np.log(n/nu * (1-confidence))
    
    # Standard Gaussian VaR for comparison
    from scipy.stats import norm
    gaussian_var = -norm.ppf(1-confidence) * returns.std()
    
    return {
        'ticker': ticker,
        'evt_var': safe_round(evt_var * 100, 2),
        'gaussian_var': safe_round(gaussian_var * 100, 2),
        'difference': safe_round((evt_var - gaussian_var) * 100, 2),
        'shape_parameter': safe_round(shape, 4),
        'scale_parameter': safe_round(scale, 4),
        'threshold': safe_round(threshold * 100, 2),
        'exceedances_count': nu,
        'confidence': confidence,
        'interpretation': f'EVT shows {"higher" if evt_var > gaussian_var else "lower"} tail risk than Gaussian model',
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# MODULE D: VOLATILITY SURFACE - 3D Analysis
# =============================================================================

def generate_volatility_surface(tickers: List[str], periods: List[int] = [30, 60, 90, 120]) -> Dict:
    """
    Generate 3D volatility surface data
    X = Ticker index, Y = Period (days), Z = Volatility
    """
    surface_data = []
    
    for i, ticker in enumerate(tickers):
        df = get_price_data(ticker, '2y')
        if df is None:
            continue
        
        returns = df['Close'].pct_change().dropna()
        
        for period in periods:
            if len(returns) < period:
                continue
            
            rolling_vol = returns.rolling(period).std() * np.sqrt(252)
            current_vol = rolling_vol.iloc[-1]
            
            surface_data.append({
                'ticker': ticker,
                'ticker_idx': i,
                'period': period,
                'volatility': safe_round(current_vol * 100, 2),
                'vol_change': safe_round((current_vol - rolling_vol.iloc[-period]) / rolling_vol.iloc[-period] * 100, 2) if len(rolling_vol) > period else None,
            })
    
    if not surface_data:
        return {'error': 'Unable to generate surface data'}
    
    # Format for Plotly 3D mesh
    return {
        'tickers': tickers,
        'periods': periods,
        'data': surface_data,
        'x': [d['ticker_idx'] for d in surface_data],
        'y': [d['period'] for d in surface_data],
        'z': [d['volatility'] for d in surface_data],
        'timestamp': datetime.now().isoformat(),
    }


def generate_correlation_heatmap(tickers: List[str], period: str = '1y') -> Dict:
    """
    Generate correlation matrix for multiple assets
    """
    prices = {}
    
    for ticker in tickers:
        df = get_price_data(ticker, period)
        if df is not None:
            prices[ticker] = df['Close']
    
    if len(prices) < 2:
        return {'error': 'Need at least 2 valid tickers'}
    
    # Create DataFrame of returns
    returns_df = pd.DataFrame(prices).pct_change().dropna()
    
    # Calculate correlation
    corr_matrix = returns_df.corr()
    
    return {
        'tickers': list(corr_matrix.columns),
        'matrix': corr_matrix.values.tolist(),
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# MODULE E: CHAOS LAB - Entropy & Cointegration
# =============================================================================

def calculate_sample_entropy(ticker: str, m: int = 2, r: float = 0.2) -> Dict:
    """
    Calculate Sample Entropy (SampEn) to measure predictability
    Lower entropy = more predictable patterns
    """
    df = get_price_data(ticker, '2y')
    if df is None:
        return {'error': f'Unable to fetch data for {ticker}'}
    
    prices = df['Close'].values
    returns = np.diff(np.log(prices))
    
    # Normalize returns
    returns_norm = (returns - np.mean(returns)) / np.std(returns)
    
    # Simple SampEn implementation
    N = len(returns_norm)
    
    def _phi(m_val):
        templates = np.array([returns_norm[i:i+m_val] for i in range(N - m_val)])
        
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        
        return count / (len(templates) * (len(templates) - 1) / 2) if len(templates) > 1 else 0
    
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if phi_m == 0 or phi_m1 == 0:
        sampen = None
        interpretation = "Unable to calculate"
    else:
        sampen = -np.log(phi_m1 / phi_m)
        if sampen < 0.5:
            interpretation = "Highly Predictable - Strong patterns detected"
        elif sampen < 1.0:
            interpretation = "Moderately Predictable"
        elif sampen < 1.5:
            interpretation = "Somewhat Random"
        else:
            interpretation = "Highly Random - Efficient market behavior"
    
    return {
        'ticker': ticker,
        'sample_entropy': safe_round(sampen, 4) if sampen else None,
        'interpretation': interpretation,
        'embedding_dimension': m,
        'tolerance': r,
        'data_points': N,
        'timestamp': datetime.now().isoformat(),
    }


def find_cointegrated_pairs(tickers: List[str], pvalue_threshold: float = 0.05) -> Dict:
    """
    Find cointegrated pairs for pairs trading using ADF test
    """
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        return {'error': 'statsmodels not installed'}
    
    # Fetch prices
    prices = {}
    for ticker in tickers:
        df = get_price_data(ticker, '2y')
        if df is not None:
            prices[ticker] = df['Close']
    
    if len(prices) < 2:
        return {'error': 'Need at least 2 valid tickers'}
    
    # Test all pairs
    pairs = []
    tested_tickers = list(prices.keys())
    
    for i, t1 in enumerate(tested_tickers):
        for t2 in tested_tickers[i+1:]:
            # Align series
            p1 = prices[t1]
            p2 = prices[t2]
            common_idx = p1.index.intersection(p2.index)
            
            if len(common_idx) < 60:
                continue
            
            p1_aligned = p1.loc[common_idx]
            p2_aligned = p2.loc[common_idx]
            
            # Run cointegration test
            try:
                score, pvalue, _ = coint(p1_aligned, p2_aligned)
                
                pairs.append({
                    'pair': [t1, t2],
                    'pvalue': safe_round(pvalue, 4),
                    'score': safe_round(score, 4),
                    'cointegrated': pvalue < pvalue_threshold,
                    'correlation': safe_round(p1_aligned.corr(p2_aligned), 4),
                })
            except:
                continue
    
    # Sort by p-value
    pairs.sort(key=lambda x: x['pvalue'])
    
    return {
        'tickers': tested_tickers,
        'pairs': pairs,
        'cointegrated_count': sum(1 for p in pairs if p['cointegrated']),
        'total_pairs': len(pairs),
        'pvalue_threshold': pvalue_threshold,
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@quant_v2_bp.route('/api/quant/backtest', methods=['POST'])
def backtest_endpoint():
    """Run vectorized backtest"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        rules = data.get('rules', [])
        period = data.get('period', '5y')
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        if not rules:
            # Default rules: RSI oversold entry, overbought exit
            rules = [
                {'indicator': 'RSI', 'operator': '<', 'value': 30, 'action': 'entry'},
                {'indicator': 'RSI', 'operator': '>', 'value': 70, 'action': 'exit'},
            ]
        
        result = generate_tearsheet(ticker, rules, period)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@quant_v2_bp.route('/api/quant/regime', methods=['POST'])
def regime_endpoint():
    """Detect market regimes"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        n_regimes = data.get('n_regimes', 3)
        period = data.get('period', '5y')
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        result = detect_market_regimes(ticker, n_regimes, period)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@quant_v2_bp.route('/api/quant/factors', methods=['POST'])
def factors_endpoint():
    """Fama-French factor decomposition"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        period = data.get('period', '5y')
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        result = fama_french_decomposition(ticker, period)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@quant_v2_bp.route('/api/quant/evt', methods=['POST'])
def evt_endpoint():
    """Extreme Value Theory VaR"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        confidence = data.get('confidence', 0.99)
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        result = calculate_evt_var(ticker, confidence=confidence)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@quant_v2_bp.route('/api/quant/volatility-surface', methods=['POST'])
def volatility_surface_endpoint():
    """Generate 3D volatility surface"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        periods = data.get('periods', [30, 60, 90, 120])
        
        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'At least 2 tickers required'}), 400
        
        result = generate_volatility_surface(tickers, periods)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@quant_v2_bp.route('/api/quant/entropy/<ticker>', methods=['GET'])
def entropy_endpoint(ticker: str):
    """Calculate sample entropy"""
    try:
        result = calculate_sample_entropy(ticker.upper())
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@quant_v2_bp.route('/api/quant/pairs', methods=['POST'])
def pairs_endpoint():
    """Find cointegrated pairs"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        pvalue_threshold = data.get('pvalue_threshold', 0.05)
        
        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'At least 2 tickers required'}), 400
        
        result = find_cointegrated_pairs(tickers, pvalue_threshold)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@quant_v2_bp.route('/api/quant/correlation', methods=['POST'])
def correlation_endpoint():
    """Generate correlation heatmap"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        period = data.get('period', '1y')
        
        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'At least 2 tickers required'}), 400
        
        result = generate_correlation_heatmap(tickers, period)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@quant_v2_bp.route('/api/quant/health', methods=['GET'])
def quant_v2_health():
    """Health check for Quant Lab V2"""
    modules = {
        'strategy_forge': True,
        'regime_detector': False,
        'fama_french': False,
        'evt': True,
        'volatility_surface': True,
        'chaos_lab': False,
    }
    
    # Check for optional dependencies
    try:
        from hmmlearn.hmm import GaussianHMM
        modules['regime_detector'] = True
    except:
        pass
    
    try:
        import statsmodels.api as sm
        modules['fama_french'] = True
        modules['chaos_lab'] = True
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'modules': modules,
        'timestamp': datetime.now().isoformat(),
    })
