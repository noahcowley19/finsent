# =============================================================================
# SCREENER V2 - ML-Powered Stock Discovery Engine
# =============================================================================
# Features:
# - Behavioral Clustering (K-Means)
# - Anomaly Detection (Isolation Forest)
# - Find Alike Engine (Nearest Neighbors)
# - Z-Score Filtering
# - MACD Derivatives (Velocity/Acceleration)
# - Piotroski F-Score & Altman Z-Score
# =============================================================================

from flask import Blueprint, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ML imports (graceful fallback)
try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Create Blueprint
screener_v2_bp = Blueprint('screener_v2', __name__)

# =============================================================================
# EXPANDED STOCK UNIVERSE (500+ tickers)
# =============================================================================

# S&P 500 representative sample + popular stocks
STOCK_UNIVERSE = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL',
    'CRM', 'ADBE', 'AMD', 'INTC', 'CSCO', 'QCOM', 'TXN', 'IBM', 'NOW', 'INTU',
    'AMAT', 'MU', 'ADI', 'LRCX', 'SNPS', 'KLAC', 'CDNS', 'MRVL', 'FTNT', 'PANW',
    'CRWD', 'ZS', 'DDOG', 'NET', 'SNOW', 'PLTR', 'DELL', 'HPQ', 'HPE', 'WDAY',
    
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
    'PNC', 'TFC', 'COF', 'BK', 'STT', 'AIG', 'MET', 'PRU', 'ALL', 'TRV',
    'CB', 'AON', 'MMC', 'ICE', 'CME', 'SPGI', 'MCO', 'MSCI', 'FIS', 'FISV',
    'V', 'MA', 'PYPL', 'SQ', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC',
    
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'VRTX', 'REGN', 'MRNA', 'BIIB', 'ISRG', 'MDT', 'SYK', 'BSX',
    'EW', 'ZBH', 'BDX', 'BAX', 'DXCM', 'IDXX', 'IQV', 'A', 'MTD', 'WAT',
    'CVS', 'CI', 'ELV', 'HUM', 'CNC', 'MOH', 'HCA', 'UHS', 'THC', 'GEHC',
    
    # Consumer Discretionary
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'ROST',
    'CMG', 'YUM', 'DPZ', 'ORLY', 'AZO', 'BBY', 'ULTA', 'LULU', 'RCL', 'CCL',
    'MAR', 'HLT', 'WYNN', 'LVS', 'MGM', 'BKNG', 'ABNB', 'EXPE', 'DRI', 'POOL',
    'F', 'GM', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'APTV', 'BWA', 'LEA',
    
    # Consumer Staples
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'MDLZ', 'PM', 'MO', 'CL', 'EL',
    'KMB', 'GIS', 'K', 'SJM', 'CAG', 'HSY', 'MNST', 'KHC', 'KR', 'SYY',
    'STZ', 'BF.B', 'TAP', 'TSN', 'HRL', 'CPB', 'MKC', 'CHD', 'CLX', 'KDP',
    
    # Industrials
    'CAT', 'DE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'UNP',
    'CSX', 'NSC', 'FDX', 'GD', 'NOC', 'LHX', 'TDG', 'IR', 'EMR', 'ETN',
    'PH', 'ROK', 'CMI', 'PCAR', 'FAST', 'GWW', 'WM', 'RSG', 'VRSK', 'CARR',
    'OTIS', 'JCI', 'TT', 'A', 'URI', 'PWR', 'AME', 'CPRT', 'CTAS', 'PAYX',
    
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
    'WMB', 'OKE', 'HAL', 'BKR', 'FANG', 'DVN', 'HES', 'PXD', 'APA', 'MRO',
    
    # Materials
    'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'ECL', 'VMC',
    'MLM', 'PPG', 'ALB', 'CTVA', 'CF', 'MOS', 'FMC', 'IFF', 'EMN', 'CE',
    
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC',
    'PEG', 'ES', 'AWK', 'AEE', 'CMS', 'LNT', 'PPL', 'FE', 'EVRG', 'NI',
    
    # REITs
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'DLR', 'WELL', 'AVB', 'EQR',
    'PSA', 'VTR', 'SLG', 'ARE', 'BXP', 'KIM', 'REG', 'FRT', 'MAA', 'UDR',
    
    # Communication Services
    'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'TTWO', 'PARA',
    'WBD', 'FOX', 'FOXA', 'LYV', 'MTCH', 'RBLX', 'U', 'SPOT', 'PINS', 'SNAP',
    
    # ETFs for benchmarking
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VTV', 'VUG', 'VB', 'VWO',
    'EFA', 'EEM', 'GLD', 'SLV', 'USO', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI',
]

# Remove duplicates
STOCK_UNIVERSE = list(set(STOCK_UNIVERSE))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_value(value: Any) -> Any:
    """Clean NaN and Inf values"""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
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


# =============================================================================
# Z-SCORE CALCULATIONS
# =============================================================================

def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling Z-score"""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore


def calculate_all_zscores(hist: pd.DataFrame) -> Dict:
    """Calculate Z-scores for key metrics"""
    if hist is None or hist.empty or len(hist) < 20:
        return {}
    
    close = hist['Close']
    volume = hist['Volume']
    returns = close.pct_change()
    
    # Rolling volatility
    volatility = returns.rolling(20).std() * np.sqrt(252)
    
    return {
        'volume_zscore': safe_round(calculate_zscore(volume).iloc[-1], 2),
        'return_zscore': safe_round(calculate_zscore(returns).iloc[-1], 2),
        'volatility_zscore': safe_round(calculate_zscore(volatility).iloc[-1], 2) if len(hist) > 40 else None,
    }


# =============================================================================
# MACD DERIVATIVES (Velocity & Acceleration)
# =============================================================================

def calculate_macd_derivatives(hist: pd.DataFrame) -> Dict:
    """Calculate MACD velocity and acceleration"""
    if hist is None or hist.empty or len(hist) < 35:
        return {}
    
    close = hist['Close']
    
    # Calculate MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Derivatives
    velocity = histogram.diff()  # First derivative
    acceleration = velocity.diff()  # Second derivative
    
    return {
        'macd_histogram': safe_round(histogram.iloc[-1], 4),
        'macd_velocity': safe_round(velocity.iloc[-1], 4),
        'macd_acceleration': safe_round(acceleration.iloc[-1], 4),
        'momentum_increasing': bool(velocity.iloc[-1] > 0 and acceleration.iloc[-1] > 0) if velocity.iloc[-1] and acceleration.iloc[-1] else None,
        'momentum_signal': 'accelerating_bullish' if (velocity.iloc[-1] > 0 and acceleration.iloc[-1] > 0) else
                          'decelerating_bullish' if (velocity.iloc[-1] > 0 and acceleration.iloc[-1] < 0) else
                          'accelerating_bearish' if (velocity.iloc[-1] < 0 and acceleration.iloc[-1] < 0) else
                          'decelerating_bearish',
    }


# =============================================================================
# PIOTROSKI F-SCORE (0-9)
# =============================================================================

def calculate_piotroski_score(info: Dict, hist: pd.DataFrame = None) -> Dict:
    """
    Calculate Piotroski F-Score (0-9)
    Higher = stronger financial position
    """
    score = 0
    details = {}
    
    # 1. Positive ROA
    roa = info.get('returnOnAssets', 0) or 0
    if roa > 0:
        score += 1
        details['positive_roa'] = True
    else:
        details['positive_roa'] = False
    
    # 2. Positive Operating Cash Flow
    ocf = info.get('operatingCashflow', 0) or 0
    if ocf > 0:
        score += 1
        details['positive_ocf'] = True
    else:
        details['positive_ocf'] = False
    
    # 3. ROA Improving (simplified - compare to sector average)
    # Using trailing vs forward earnings as proxy
    trailing_eps = info.get('trailingEps', 0) or 0
    forward_eps = info.get('forwardEps', 0) or 0
    if forward_eps > trailing_eps:
        score += 1
        details['roa_improving'] = True
    else:
        details['roa_improving'] = False
    
    # 4. Cash Flow > Net Income (Accruals check)
    net_income = info.get('netIncomeToCommon', 0) or 0
    if ocf > net_income:
        score += 1
        details['quality_earnings'] = True
    else:
        details['quality_earnings'] = False
    
    # 5. Long-term Debt Decreasing (use D/E as proxy)
    debt_to_equity = info.get('debtToEquity', 100) or 100
    if debt_to_equity < 50:  # Low debt
        score += 1
        details['low_debt'] = True
    else:
        details['low_debt'] = False
    
    # 6. Current Ratio > 1 (Liquidity)
    current_ratio = info.get('currentRatio', 0) or 0
    if current_ratio > 1:
        score += 1
        details['good_liquidity'] = True
    else:
        details['good_liquidity'] = False
    
    # 7. No Share Dilution
    shares_outstanding = info.get('sharesOutstanding', 0) or 0
    float_shares = info.get('floatShares', 0) or 0
    if shares_outstanding > 0 and float_shares > 0:
        dilution = (shares_outstanding - float_shares) / shares_outstanding
        if dilution < 0.1:  # Less than 10% non-float
            score += 1
            details['no_dilution'] = True
        else:
            details['no_dilution'] = False
    else:
        details['no_dilution'] = None
    
    # 8. Gross Margin (using profit margin as proxy)
    profit_margin = info.get('profitMargins', 0) or 0
    if profit_margin > 0.1:  # > 10%
        score += 1
        details['good_margins'] = True
    else:
        details['good_margins'] = False
    
    # 9. Asset Turnover (Revenue/Assets efficiency)
    revenue = info.get('totalRevenue', 0) or 0
    assets = info.get('totalAssets', 1) or 1
    turnover = revenue / assets if assets else 0
    if turnover > 0.5:
        score += 1
        details['good_turnover'] = True
    else:
        details['good_turnover'] = False
    
    return {
        'piotroski_score': score,
        'piotroski_grade': 'Strong' if score >= 7 else 'Moderate' if score >= 4 else 'Weak',
        'piotroski_details': details,
    }


# =============================================================================
# ALTMAN Z-SCORE (Bankruptcy Prediction)
# =============================================================================

def calculate_altman_zscore(info: Dict) -> Dict:
    """
    Calculate Altman Z-Score for bankruptcy prediction
    Z > 2.99 = Safe Zone
    1.81 < Z < 2.99 = Grey Zone
    Z < 1.81 = Distress Zone
    """
    # Get financial data
    market_cap = info.get('marketCap', 0) or 0
    total_assets = info.get('totalAssets', 1) or 1
    total_liabilities = info.get('totalDebt', 0) or 0
    current_assets = info.get('totalCurrentAssets', 0) or 0
    current_liabilities = info.get('totalCurrentLiabilities', 0) or 0
    retained_earnings = info.get('retainedEarnings', 0) or 0
    ebit = info.get('ebitda', 0) or 0  # Using EBITDA as proxy
    revenue = info.get('totalRevenue', 0) or 0
    
    # Calculate ratios
    working_capital = current_assets - current_liabilities
    
    x1 = working_capital / total_assets if total_assets else 0
    x2 = retained_earnings / total_assets if total_assets else 0
    x3 = ebit / total_assets if total_assets else 0
    x4 = market_cap / total_liabilities if total_liabilities else 5  # High if no debt
    x5 = revenue / total_assets if total_assets else 0
    
    # Altman Z-Score formula
    z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
    
    if z_score > 2.99:
        zone = 'Safe'
    elif z_score > 1.81:
        zone = 'Grey'
    else:
        zone = 'Distress'
    
    return {
        'altman_zscore': safe_round(z_score, 2),
        'altman_zone': zone,
        'altman_components': {
            'working_capital_ratio': safe_round(x1, 4),
            'retained_earnings_ratio': safe_round(x2, 4),
            'ebit_ratio': safe_round(x3, 4),
            'market_value_ratio': safe_round(x4, 4),
            'asset_turnover': safe_round(x5, 4),
        }
    }


# =============================================================================
# ENHANCED STOCK METRICS
# =============================================================================

def get_enhanced_metrics(ticker: str) -> Optional[Dict]:
    """Get comprehensive metrics including Z-scores and fundamental scores"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            return None
        
        # Get historical data
        hist = stock.history(period='6mo')
        
        # Basic metrics
        metrics = {
            'ticker': ticker.upper(),
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            
            # Price
            'price': safe_round(info.get('currentPrice') or info.get('regularMarketPrice'), 2),
            'change_pct': safe_round(info.get('regularMarketChangePercent'), 2),
            'volume': info.get('regularMarketVolume'),
            'avg_volume': info.get('averageVolume'),
            
            # Valuation
            'market_cap': info.get('marketCap'),
            'pe_ratio': safe_round(info.get('trailingPE'), 2),
            'forward_pe': safe_round(info.get('forwardPE'), 2),
            'peg_ratio': safe_round(info.get('pegRatio'), 2),
            'pb_ratio': safe_round(info.get('priceToBook'), 2),
            
            # Fundamentals
            'roe': safe_round((info.get('returnOnEquity') or 0) * 100, 2),
            'profit_margin': safe_round((info.get('profitMargins') or 0) * 100, 2),
            'debt_to_equity': safe_round(info.get('debtToEquity'), 2),
            'current_ratio': safe_round(info.get('currentRatio'), 2),
            'dividend_yield': safe_round((info.get('dividendYield') or 0) * 100, 2),
            'beta': safe_round(info.get('beta'), 2),
            
            # 52-week range
            'fifty_two_week_high': safe_round(info.get('fiftyTwoWeekHigh'), 2),
            'fifty_two_week_low': safe_round(info.get('fiftyTwoWeekLow'), 2),
        }
        
        # Technical indicators
        if not hist.empty and len(hist) >= 14:
            close = hist['Close']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            metrics['rsi'] = safe_round(rsi.iloc[-1], 2)
            
            # Moving averages
            metrics['sma_20'] = safe_round(close.rolling(20).mean().iloc[-1], 2)
            metrics['sma_50'] = safe_round(close.rolling(50).mean().iloc[-1], 2) if len(close) >= 50 else None
            
            # Returns
            metrics['return_5d'] = safe_round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 2) if len(close) >= 5 else None
            metrics['return_20d'] = safe_round((close.iloc[-1] / close.iloc[-20] - 1) * 100, 2) if len(close) >= 20 else None
            
            # Volatility
            returns = close.pct_change()
            metrics['volatility_20d'] = safe_round(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100, 2)
        
        # Z-Scores
        zscores = calculate_all_zscores(hist)
        metrics.update(zscores)
        
        # MACD Derivatives
        macd = calculate_macd_derivatives(hist)
        metrics.update(macd)
        
        # Piotroski F-Score
        piotroski = calculate_piotroski_score(info, hist)
        metrics.update(piotroski)
        
        # Altman Z-Score
        altman = calculate_altman_zscore(info)
        metrics.update(altman)
        
        return metrics
        
    except Exception as e:
        return None


# =============================================================================
# ML DISCOVERY ENGINE
# =============================================================================

def extract_behavior_features(stocks_data: List[Dict]) -> np.ndarray:
    """Extract feature matrix for ML models"""
    features = []
    
    for stock in stocks_data:
        if stock is None:
            continue
        
        feature_vector = [
            stock.get('return_5d', 0) or 0,
            stock.get('return_20d', 0) or 0,
            stock.get('volatility_20d', 0) or 0,
            stock.get('volume_zscore', 0) or 0,
            stock.get('rsi', 50) or 50,
            stock.get('macd_velocity', 0) or 0,
        ]
        features.append(feature_vector)
    
    return np.array(features)


def run_clustering(stocks_data: List[Dict], n_clusters: int = 5) -> Dict:
    """Run K-Means clustering on stock behavior"""
    if not ML_AVAILABLE:
        return {'error': 'scikit-learn not installed'}
    
    if len(stocks_data) < n_clusters:
        return {'error': 'Not enough stocks for clustering'}
    
    # Extract features
    features = extract_behavior_features(stocks_data)
    
    if len(features) < n_clusters:
        return {'error': 'Not enough valid data'}
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Analyze clusters
    cluster_info = []
    for i in range(n_clusters):
        mask = labels == i
        cluster_stocks = [s for j, s in enumerate(stocks_data) if mask[j]]
        
        avg_return = np.mean([s.get('return_5d', 0) or 0 for s in cluster_stocks])
        avg_vol = np.mean([s.get('volatility_20d', 0) or 0 for s in cluster_stocks])
        avg_volume_z = np.mean([s.get('volume_zscore', 0) or 0 for s in cluster_stocks])
        
        # Name cluster based on characteristics
        if avg_return > 5 and avg_volume_z > 1:
            name = "Momentum Ignition"
            color = "#22C55E"
        elif avg_return > 2 and avg_vol < 20:
            name = "Slow Accumulation"
            color = "#3B82F6"
        elif avg_return < -5:
            name = "Freefall"
            color = "#EF4444"
        elif abs(avg_return) < 2 and avg_vol < 15:
            name = "Sideways Consolidation"
            color = "#94A3B8"
        else:
            name = f"Cluster {i+1}"
            color = "#6366F1"
        
        cluster_info.append({
            'id': i,
            'name': name,
            'color': color,
            'count': len(cluster_stocks),
            'avg_return_5d': safe_round(avg_return, 2),
            'avg_volatility': safe_round(avg_vol, 2),
            'avg_volume_zscore': safe_round(avg_volume_z, 2),
            'tickers': [s['ticker'] for s in cluster_stocks[:10]],  # Top 10
        })
    
    # Assign labels to stocks
    for i, stock in enumerate(stocks_data):
        if i < len(labels):
            stock['cluster_id'] = int(labels[i])
            stock['cluster_name'] = cluster_info[labels[i]]['name']
    
    return {
        'clusters': cluster_info,
        'stocks': stocks_data,
    }


def detect_anomalies(stocks_data: List[Dict], contamination: float = 0.05) -> Dict:
    """Detect anomalous stocks using Isolation Forest"""
    if not ML_AVAILABLE:
        return {'error': 'scikit-learn not installed'}
    
    features = extract_behavior_features(stocks_data)
    
    if len(features) < 10:
        return {'error': 'Not enough data for anomaly detection'}
    
    # Detect anomalies
    iso = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso.fit_predict(features)
    
    # Mark anomalies
    anomalies = []
    for i, stock in enumerate(stocks_data):
        if i < len(predictions):
            is_anomaly = predictions[i] == -1
            stock['is_anomaly'] = is_anomaly
            if is_anomaly:
                anomalies.append(stock)
    
    return {
        'anomaly_count': len(anomalies),
        'anomalies': anomalies,
        'all_stocks': stocks_data,
    }


def find_similar_stocks(target_ticker: str, stocks_data: List[Dict], n_neighbors: int = 10) -> Dict:
    """Find stocks similar to target using Nearest Neighbors"""
    if not ML_AVAILABLE:
        return {'error': 'scikit-learn not installed'}
    
    # Find target
    target_idx = None
    for i, stock in enumerate(stocks_data):
        if stock.get('ticker', '').upper() == target_ticker.upper():
            target_idx = i
            break
    
    if target_idx is None:
        return {'error': f'{target_ticker} not found in universe'}
    
    features = extract_behavior_features(stocks_data)
    
    if len(features) < n_neighbors + 1:
        return {'error': 'Not enough data'}
    
    # Fit NN
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(features)))
    nn.fit(X)
    
    distances, indices = nn.kneighbors([X[target_idx]])
    
    # Get similar stocks (exclude self)
    similar = []
    for i, idx in enumerate(indices[0][1:]):
        if idx < len(stocks_data):
            stock = stocks_data[idx].copy()
            stock['similarity_rank'] = i + 1
            stock['distance'] = safe_round(distances[0][i + 1], 4)
            similar.append(stock)
    
    return {
        'target': stocks_data[target_idx],
        'similar': similar,
    }


# =============================================================================
# FILTERING ENGINE
# =============================================================================

def apply_advanced_filters(stocks: List[Dict], filters: Dict) -> List[Dict]:
    """Apply advanced filters including Z-scores and ML clusters"""
    filtered = []
    
    for stock in stocks:
        if stock is None:
            continue
        
        passes = True
        
        for key, condition in filters.items():
            value = stock.get(key)
            
            if value is None:
                if condition.get('required', False):
                    passes = False
                    break
                continue
            
            # Handle different filter types
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


# =============================================================================
# API ENDPOINTS
# =============================================================================

@screener_v2_bp.route('/api/screener/v2/scan', methods=['POST'])
def scan_v2():
    """Advanced screener with ML and statistical filters"""
    try:
        data = request.get_json() or {}
        filters = data.get('filters', {})
        sort_by = data.get('sort_by', 'market_cap')
        ascending = data.get('ascending', False)
        limit = min(data.get('limit', 50), 200)
        include_clustering = data.get('include_clustering', False)
        include_anomalies = data.get('include_anomalies', False)
        
        # Fetch metrics in parallel
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(get_enhanced_metrics, ticker): ticker 
                      for ticker in STOCK_UNIVERSE[:200]}  # Limit for performance
            
            for future in as_completed(futures):
                try:
                    metrics = future.result(timeout=30)
                    if metrics:
                        all_metrics.append(metrics)
                except:
                    continue
        
        # Apply filters
        filtered = apply_advanced_filters(all_metrics, filters)
        
        # Optional: Run clustering
        if include_clustering and ML_AVAILABLE:
            cluster_result = run_clustering(filtered)
            if 'stocks' in cluster_result:
                filtered = cluster_result['stocks']
        
        # Optional: Flag anomalies
        if include_anomalies and ML_AVAILABLE:
            anomaly_result = detect_anomalies(filtered)
            if 'all_stocks' in anomaly_result:
                filtered = anomaly_result['all_stocks']
        
        # Sort
        sorted_stocks = sorted(
            [s for s in filtered if s.get(sort_by) is not None],
            key=lambda x: x.get(sort_by, 0),
            reverse=not ascending
        )
        
        # Limit
        results = sorted_stocks[:limit]
        
        return jsonify({
            'results': results,
            'count': len(results),
            'total_scanned': len(all_metrics),
            'ml_available': ML_AVAILABLE,
            'timestamp': datetime.now().isoformat(),
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@screener_v2_bp.route('/api/screener/v2/clusters', methods=['GET'])
def get_clusters():
    """Get behavioral clusters for the universe"""
    try:
        # Fetch data
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(get_enhanced_metrics, ticker): ticker 
                      for ticker in STOCK_UNIVERSE[:150]}
            
            for future in as_completed(futures):
                try:
                    metrics = future.result(timeout=30)
                    if metrics:
                        all_metrics.append(metrics)
                except:
                    continue
        
        # Run clustering
        result = run_clustering(all_metrics)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'clusters': result.get('clusters', []),
            'total_stocks': len(all_metrics),
            'timestamp': datetime.now().isoformat(),
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@screener_v2_bp.route('/api/screener/v2/anomalies', methods=['GET'])
def get_anomalies():
    """Get anomaly-flagged stocks"""
    try:
        # Fetch data
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(get_enhanced_metrics, ticker): ticker 
                      for ticker in STOCK_UNIVERSE[:150]}
            
            for future in as_completed(futures):
                try:
                    metrics = future.result(timeout=30)
                    if metrics:
                        all_metrics.append(metrics)
                except:
                    continue
        
        # Detect anomalies
        result = detect_anomalies(all_metrics)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'anomalies': result.get('anomalies', []),
            'anomaly_count': result.get('anomaly_count', 0),
            'total_scanned': len(all_metrics),
            'timestamp': datetime.now().isoformat(),
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@screener_v2_bp.route('/api/screener/v2/similar/<ticker>', methods=['GET'])
def get_similar(ticker: str):
    """Find stocks similar to target"""
    try:
        n_neighbors = request.args.get('n', 10, type=int)
        
        # Fetch data
        all_metrics = []
        
        # Make sure target is in universe
        universe = list(set(STOCK_UNIVERSE[:150] + [ticker.upper()]))
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(get_enhanced_metrics, t): t for t in universe}
            
            for future in as_completed(futures):
                try:
                    metrics = future.result(timeout=30)
                    if metrics:
                        all_metrics.append(metrics)
                except:
                    continue
        
        # Find similar
        result = find_similar_stocks(ticker, all_metrics, n_neighbors)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'target': result.get('target'),
            'similar': result.get('similar', []),
            'timestamp': datetime.now().isoformat(),
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@screener_v2_bp.route('/api/screener/v2/metrics', methods=['GET'])
def get_v2_metrics():
    """Get available filter metrics"""
    return jsonify({
        'metrics': [
            # Basic
            {'key': 'price', 'name': 'Price', 'type': 'number'},
            {'key': 'market_cap', 'name': 'Market Cap', 'type': 'number'},
            {'key': 'change_pct', 'name': 'Change %', 'type': 'number'},
            
            # Valuation
            {'key': 'pe_ratio', 'name': 'P/E Ratio', 'type': 'number'},
            {'key': 'peg_ratio', 'name': 'PEG Ratio', 'type': 'number'},
            {'key': 'pb_ratio', 'name': 'P/B Ratio', 'type': 'number'},
            
            # Technical
            {'key': 'rsi', 'name': 'RSI', 'type': 'number'},
            {'key': 'return_5d', 'name': '5-Day Return %', 'type': 'number'},
            {'key': 'return_20d', 'name': '20-Day Return %', 'type': 'number'},
            {'key': 'volatility_20d', 'name': '20-Day Volatility', 'type': 'number'},
            
            # Z-Scores
            {'key': 'volume_zscore', 'name': 'Volume Z-Score', 'type': 'number', 'advanced': True},
            {'key': 'return_zscore', 'name': 'Return Z-Score', 'type': 'number', 'advanced': True},
            {'key': 'volatility_zscore', 'name': 'Volatility Z-Score', 'type': 'number', 'advanced': True},
            
            # MACD Derivatives
            {'key': 'macd_velocity', 'name': 'MACD Velocity', 'type': 'number', 'advanced': True},
            {'key': 'macd_acceleration', 'name': 'MACD Acceleration', 'type': 'number', 'advanced': True},
            {'key': 'momentum_increasing', 'name': 'Momentum Increasing', 'type': 'boolean', 'advanced': True},
            
            # Fundamental Scores
            {'key': 'piotroski_score', 'name': 'Piotroski F-Score (0-9)', 'type': 'number', 'advanced': True},
            {'key': 'altman_zscore', 'name': 'Altman Z-Score', 'type': 'number', 'advanced': True},
            
            # ML
            {'key': 'cluster_id', 'name': 'Cluster ID', 'type': 'number', 'ml': True},
            {'key': 'is_anomaly', 'name': 'Is Anomaly', 'type': 'boolean', 'ml': True},
        ],
        'timestamp': datetime.now().isoformat(),
    })


@screener_v2_bp.route('/api/screener/v2/health', methods=['GET'])
def screener_v2_health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'ml_available': ML_AVAILABLE,
        'universe_size': len(STOCK_UNIVERSE),
        'features': {
            'z_scores': True,
            'macd_derivatives': True,
            'piotroski_score': True,
            'altman_zscore': True,
            'clustering': ML_AVAILABLE,
            'anomaly_detection': ML_AVAILABLE,
            'find_alike': ML_AVAILABLE,
        },
        'timestamp': datetime.now().isoformat(),
    })
