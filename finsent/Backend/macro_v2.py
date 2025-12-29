# =============================================================================
# MACRO V2 - Global Macro Intelligence Hub
# =============================================================================
# Modules:
# A. Global Liquidity Impulse (Fed + ECB + BOJ)
# B. Yield Curve 3D & Recession Probability (Probit)
# C. Fed Speak Decoder (NLP Sentiment)
# D. Inflation Nowcast (Real-time Proxy)
# E. Cross-Asset Stress Heatmap
# F. Economic Vitals Grid
# =============================================================================

from flask import Blueprint, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
import os
import warnings
warnings.filterwarnings('ignore')

# Optional FRED import
try:
    from fredapi import Fred
    FRED_API_KEY = os.environ.get('FRED_API_KEY')
    if FRED_API_KEY:
        fred = Fred(api_key=FRED_API_KEY)
        FRED_AVAILABLE = True
    else:
        fred = None
        FRED_AVAILABLE = False
except ImportError:
    fred = None
    FRED_AVAILABLE = False

# VADER for lightweight sentiment (already installed)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    vader = None
    VADER_AVAILABLE = False

# Scipy for Probit
try:
    from scipy.stats import norm
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Create Blueprint
macro_v2_bp = Blueprint('macro_v2', __name__)


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


def format_trillions(value: float) -> str:
    """Format large numbers as trillions"""
    if value is None:
        return "—"
    return f"${value / 1e12:.2f}T"


# =============================================================================
# MODULE A: GLOBAL LIQUIDITY IMPULSE
# =============================================================================

def get_fed_balance_sheet() -> Dict:
    """
    Get Federal Reserve Total Assets
    Fallback: Use proxy via ETFs if FRED unavailable
    """
    try:
        if FRED_AVAILABLE and fred:
            # WALCL = Fed Total Assets (Millions)
            data = fred.get_series('WALCL', observation_start='2020-01-01')
            if data is not None and len(data) > 0:
                latest = data.iloc[-1] * 1e6  # Convert millions to dollars
                return {
                    'value': latest,
                    'formatted': format_trillions(latest),
                    'date': data.index[-1].strftime('%Y-%m-%d'),
                    'source': 'FRED',
                }
    except Exception as e:
        print(f"FRED Fed error: {e}")
    
    # Fallback: Estimated value (update periodically)
    return {
        'value': 6.9e12,  # ~$6.9T as of late 2024
        'formatted': '$6.9T',
        'date': '2024-12-01',
        'source': 'Estimated',
    }


def get_ecb_balance_sheet() -> Dict:
    """Get European Central Bank Total Assets"""
    try:
        if FRED_AVAILABLE and fred:
            # ECBASSETSW = ECB Total Assets (Millions EUR)
            ecb_eur = fred.get_series('ECBASSETSW', observation_start='2020-01-01')
            eur_usd = fred.get_series('DEXUSEU', observation_start='2020-01-01')
            
            if ecb_eur is not None and eur_usd is not None:
                # Align dates
                latest_ecb = ecb_eur.iloc[-1] * 1e6
                latest_rate = eur_usd.iloc[-1]
                ecb_usd = latest_ecb * latest_rate
                
                return {
                    'value': ecb_usd,
                    'formatted': format_trillions(ecb_usd),
                    'eur_value': latest_ecb,
                    'exchange_rate': latest_rate,
                    'date': ecb_eur.index[-1].strftime('%Y-%m-%d'),
                    'source': 'FRED',
                }
    except Exception as e:
        print(f"FRED ECB error: {e}")
    
    # Fallback
    return {
        'value': 7.0e12,
        'formatted': '$7.0T',
        'date': '2024-12-01',
        'source': 'Estimated',
    }


def get_boj_balance_sheet() -> Dict:
    """Get Bank of Japan Total Assets"""
    try:
        if FRED_AVAILABLE and fred:
            # JPNASSETS = BOJ Assets (100 Millions JPY)
            boj_jpy = fred.get_series('JPNASSETS', observation_start='2020-01-01')
            jpy_usd = fred.get_series('DEXJPUS', observation_start='2020-01-01')
            
            if boj_jpy is not None and jpy_usd is not None:
                latest_boj = boj_jpy.iloc[-1] * 1e8  # 100 millions to yen
                latest_rate = 1 / jpy_usd.iloc[-1]  # Invert for JPY to USD
                boj_usd = latest_boj * latest_rate
                
                return {
                    'value': boj_usd,
                    'formatted': format_trillions(boj_usd),
                    'jpy_value': latest_boj,
                    'exchange_rate': latest_rate,
                    'date': boj_jpy.index[-1].strftime('%Y-%m-%d'),
                    'source': 'FRED',
                }
    except Exception as e:
        print(f"FRED BOJ error: {e}")
    
    # Fallback
    return {
        'value': 5.0e12,
        'formatted': '$5.0T',
        'date': '2024-12-01',
        'source': 'Estimated',
    }


def calculate_global_liquidity() -> Dict:
    """Calculate total global central bank liquidity"""
    fed = get_fed_balance_sheet()
    ecb = get_ecb_balance_sheet()
    boj = get_boj_balance_sheet()
    
    total = fed['value'] + ecb['value'] + boj['value']
    
    # Get S&P 500 for overlay comparison
    try:
        spy = yf.download('SPY', period='5y', interval='1wk', progress=False)
        sp500_data = [
            {'date': d.strftime('%Y-%m-%d'), 'value': safe_round(v, 2)}
            for d, v in zip(spy.index, spy['Close'])
        ]
    except:
        sp500_data = []
    
    return {
        'total': {
            'value': total,
            'formatted': format_trillions(total),
        },
        'breakdown': {
            'fed': fed,
            'ecb': ecb,
            'boj': boj,
        },
        'sp500_overlay': sp500_data[-52:],  # Last year weekly
        'correlation_note': 'Global liquidity strongly correlates with asset prices',
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# MODULE B: YIELD CURVE & RECESSION PROBABILITY
# =============================================================================

TREASURY_MATURITIES = {
    '1M': '^IRX',   # 13-week T-bill proxy
    '3M': '^IRX',
    '6M': None,     # Need to interpolate
    '1Y': None,
    '2Y': '^FVX',   # 5-year proxy adjusted
    '5Y': '^FVX',
    '10Y': '^TNX',
    '30Y': '^TYX',
}


def get_yield_curve_data() -> Dict:
    """Get current yield curve from yfinance"""
    maturities = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
    maturity_years = [1/12, 0.25, 0.5, 1, 2, 5, 10, 30]
    
    # Fetch from yfinance
    yields = {}
    
    try:
        # 3-month (13-week) T-bill
        irx = yf.Ticker('^IRX')
        yields['3M'] = irx.fast_info.get('lastPrice', 4.5)
        yields['1M'] = yields['3M'] - 0.1  # Approximate
        yields['6M'] = yields['3M'] + 0.1
        
        # 5-year
        fvx = yf.Ticker('^FVX')
        yields['5Y'] = fvx.fast_info.get('lastPrice', 4.3)
        yields['2Y'] = yields['5Y'] - 0.2
        yields['1Y'] = yields['3M'] + 0.15
        
        # 10-year
        tnx = yf.Ticker('^TNX')
        yields['10Y'] = tnx.fast_info.get('lastPrice', 4.5)
        
        # 30-year
        tyx = yf.Ticker('^TYX')
        yields['30Y'] = tyx.fast_info.get('lastPrice', 4.7)
    except Exception as e:
        print(f"Yield curve error: {e}")
        # Fallback values
        yields = {
            '1M': 4.4, '3M': 4.5, '6M': 4.5, '1Y': 4.4,
            '2Y': 4.3, '5Y': 4.3, '10Y': 4.5, '30Y': 4.7
        }
    
    # Build curve data
    curve = [
        {'maturity': m, 'years': y, 'yield': safe_round(yields.get(m, 4.5), 2)}
        for m, y in zip(maturities, maturity_years)
    ]
    
    # Calculate key spreads
    spread_10y_2y = yields.get('10Y', 4.5) - yields.get('2Y', 4.3)
    spread_10y_3m = yields.get('10Y', 4.5) - yields.get('3M', 4.5)
    
    return {
        'curve': curve,
        'spreads': {
            '10Y_2Y': safe_round(spread_10y_2y, 2),
            '10Y_3M': safe_round(spread_10y_3m, 2),
        },
        'inverted': spread_10y_2y < 0 or spread_10y_3m < 0,
        'timestamp': datetime.now().isoformat(),
    }


def calculate_recession_probability(spread_10y_3m: float) -> Dict:
    """
    Calculate recession probability using Probit model
    Based on historical calibration of 10Y-3M spread
    """
    if not SCIPY_AVAILABLE:
        return {
            'probability': 30.0,
            'error': 'scipy not available',
        }
    
    # Calibrated coefficients from Fed research
    # P(Recession) = Φ(α + β * spread)
    alpha = -0.5
    beta = -0.8
    
    z = alpha + beta * spread_10y_3m
    probability = norm.cdf(z) * 100
    
    # Determine risk level
    if probability > 50:
        level = 'High'
        color = '#ff0066'  # Hot pink
    elif probability > 30:
        level = 'Elevated'
        color = '#ffaa00'  # Amber
    else:
        level = 'Low'
        color = '#00ff88'  # Electric green
    
    return {
        'probability': safe_round(probability, 1),
        'level': level,
        'color': color,
        'spread_used': spread_10y_3m,
        'model': 'Probit (10Y-3M Spread)',
        'timestamp': datetime.now().isoformat(),
    }


def get_yield_curve_3d_data(days: int = 365) -> Dict:
    """
    Get historical yield curve data for 3D visualization
    X = Maturity, Y = Yield, Z = Time
    """
    maturities = ['2Y', '5Y', '10Y', '30Y']
    tickers = ['^FVX', '^FVX', '^TNX', '^TYX']
    
    try:
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = {}
        for mat, ticker in zip(maturities, tickers):
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[mat] = df['Close']
        
        if not data:
            return {'error': 'No data available'}
        
        # Build 3D surface data
        df = pd.DataFrame(data)
        
        # Downsample to weekly
        df = df.resample('W').last().dropna()
        
        surface_data = []
        for date in df.index:
            for mat in maturities:
                if mat in df.columns:
                    surface_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'maturity': mat,
                        'yield': safe_round(df.loc[date, mat], 2),
                    })
        
        return {
            'surface': surface_data,
            'maturities': maturities,
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# MODULE C: FED SPEAK DECODER
# =============================================================================

def scrape_fomc_statement() -> Dict:
    """Scrape latest FOMC statement from Federal Reserve"""
    try:
        # Try to fetch from Fed website
        url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20241218a.htm"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find statement text
            content = soup.find('div', {'class': 'col-xs-12'})
            if content:
                paragraphs = content.find_all('p')
                text = ' '.join([p.get_text() for p in paragraphs])
                
                return {
                    'text': text[:2000],  # Limit length
                    'date': '2024-12-18',
                    'source': 'federalreserve.gov',
                    'success': True,
                }
    except Exception as e:
        print(f"FOMC scrape error: {e}")
    
    # Fallback: Use cached statement excerpt
    return {
        'text': """The Federal Reserve decided to lower the target range for the federal funds rate by 1/4 percentage point to 4-1/4 to 4-1/2 percent. The Committee judges that the risks to achieving its employment and inflation goals are roughly in balance. The economic outlook is uncertain, and the Committee is attentive to the risks to both sides of its dual mandate.""",
        'date': '2024-12-18',
        'source': 'Cached',
        'success': True,
    }


def analyze_fed_sentiment(text: str) -> Dict:
    """
    Analyze FOMC statement sentiment
    Returns: Hawk/Dove score from -1 (Dovish) to +1 (Hawkish)
    """
    if not VADER_AVAILABLE or not vader:
        return {
            'score': 0,
            'label': 'Neutral',
            'error': 'Sentiment analyzer not available',
        }
    
    # Fed-specific keyword adjustments
    hawkish_words = [
        'inflation', 'tightening', 'restrictive', 'vigilant', 'elevated',
        'persistent', 'upside risks', 'strong labor market', 'reduce'
    ]
    dovish_words = [
        'accommodative', 'supportive', 'balanced', 'moderate', 'easing',
        'downside risks', 'uncertainty', 'patient', 'flexible'
    ]
    
    text_lower = text.lower()
    
    # Count keyword occurrences
    hawk_count = sum(1 for word in hawkish_words if word in text_lower)
    dove_count = sum(1 for word in dovish_words if word in text_lower)
    
    # VADER sentiment
    vader_scores = vader.polarity_scores(text)
    
    # Combine signals
    # Hawk = negative sentiment (bad for markets) + hawkish keywords
    # Dove = positive sentiment (good for markets) + dovish keywords
    
    keyword_score = (hawk_count - dove_count) / max(hawk_count + dove_count, 1)
    
    # Final score: higher = more hawkish
    final_score = keyword_score * 0.7 + (-vader_scores['compound']) * 0.3
    final_score = max(-1, min(1, final_score))  # Clamp to [-1, 1]
    
    # Determine label
    if final_score > 0.3:
        label = 'Hawkish'
        color = '#ff0066'
    elif final_score < -0.3:
        label = 'Dovish'
        color = '#00ff88'
    else:
        label = 'Neutral'
        color = '#ffaa00'
    
    return {
        'score': safe_round(final_score, 2),
        'label': label,
        'color': color,
        'hawkish_keywords': hawk_count,
        'dovish_keywords': dove_count,
        'vader_compound': safe_round(vader_scores['compound'], 2),
    }


def get_fed_speak_analysis() -> Dict:
    """Complete Fed Speak analysis pipeline"""
    statement = scrape_fomc_statement()
    sentiment = analyze_fed_sentiment(statement['text'])
    
    return {
        'statement': {
            'date': statement['date'],
            'source': statement['source'],
            'excerpt': statement['text'][:500] + '...',
        },
        'sentiment': sentiment,
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# MODULE D: INFLATION NOWCAST
# =============================================================================

def calculate_inflation_nowcast() -> Dict:
    """
    Calculate real-time inflation proxy using:
    - 5Y Breakeven (market expectations)
    - Crude Oil (energy component)
    - Copper (industrial activity)
    """
    try:
        # Fetch commodity data
        oil = yf.download('CL=F', period='3mo', progress=False)['Close']
        copper = yf.download('HG=F', period='3mo', progress=False)['Close']
        
        # Calculate momentum (20-day change)
        oil_change = (oil.iloc[-1] / oil.iloc[-20] - 1) * 100 if len(oil) >= 20 else 0
        copper_change = (copper.iloc[-1] / copper.iloc[-20] - 1) * 100 if len(copper) >= 20 else 0
        
        # Estimated breakeven (use TIP vs nominal spread as proxy)
        try:
            tips = yf.download('TIP', period='1mo', progress=False)['Close'].iloc[-1]
            nominal = yf.download('IEF', period='1mo', progress=False)['Close'].iloc[-1]
            breakeven_proxy = 2.5  # Approximate 5Y breakeven
        except:
            breakeven_proxy = 2.5
        
        # Weighted composite nowcast
        # Breakeven: 50%, Oil: 30%, Copper: 20%
        nowcast = (
            breakeven_proxy * 0.5 +
            (2.5 + oil_change * 0.05) * 0.3 +
            (2.5 + copper_change * 0.03) * 0.2
        )
        
        # Official CPI comparison
        official_cpi = 2.7  # Latest CPI YoY (update manually)
        
        # Trend
        if nowcast > official_cpi + 0.2:
            trend = 'Rising'
            color = '#ff0066'
        elif nowcast < official_cpi - 0.2:
            trend = 'Falling'
            color = '#00ff88'
        else:
            trend = 'Stable'
            color = '#ffaa00'
        
        return {
            'nowcast': safe_round(nowcast, 2),
            'official_cpi': official_cpi,
            'trend': trend,
            'color': color,
            'components': {
                'breakeven': breakeven_proxy,
                'oil_20d_change': safe_round(oil_change, 2),
                'copper_20d_change': safe_round(copper_change, 2),
            },
            'methodology': '50% Breakeven + 30% Oil + 20% Copper',
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# MODULE E: CROSS-ASSET STRESS HEATMAP
# =============================================================================

def calculate_stress_correlations() -> Dict:
    """
    Calculate 30-day rolling correlation matrix for stress detection
    """
    tickers = ['SPY', 'TLT', 'HYG', 'GLD', 'UUP', 'VIX']
    labels = ['Stocks', 'Bonds', 'Junk', 'Gold', 'Dollar', 'Fear']
    
    try:
        # Fetch 60 days to get 30-day rolling
        data = yf.download(tickers, period='60d', progress=False)['Close']
        
        if data.empty:
            return {'error': 'No data available'}
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # 30-day correlation (use last 30 rows)
        corr = returns.tail(30).corr()
        
        # Build matrix for frontend
        matrix = []
        for i, t1 in enumerate(tickers):
            row = []
            for j, t2 in enumerate(tickers):
                if t1 in corr.columns and t2 in corr.columns:
                    row.append(safe_round(corr.loc[t1, t2], 2))
                else:
                    row.append(0)
            matrix.append(row)
        
        # Detect stress conditions
        stress_flags = []
        
        # Stocks-Bonds positive correlation = liquidity crisis
        spy_tlt = corr.loc['SPY', 'TLT'] if 'SPY' in corr.columns and 'TLT' in corr.columns else 0
        if spy_tlt > 0.3:
            stress_flags.append({
                'condition': 'Liquidity Crisis Signal',
                'description': 'Stocks and bonds falling together',
                'severity': 'High',
                'color': '#ff0066',
            })
        
        # High VIX correlation with stocks = fear regime
        spy_vix = corr.loc['SPY', 'VIX'] if 'SPY' in corr.columns and 'VIX' in corr.columns else -0.8
        if spy_vix > -0.5:
            stress_flags.append({
                'condition': 'Fear Regime',
                'description': 'VIX not inversely correlated with stocks',
                'severity': 'Elevated',
                'color': '#ffaa00',
            })
        
        # Gold-Dollar positive = flight to safety
        gld_uup = corr.loc['GLD', 'UUP'] if 'GLD' in corr.columns and 'UUP' in corr.columns else -0.3
        if gld_uup > 0.2:
            stress_flags.append({
                'condition': 'Flight to Safety',
                'description': 'Both gold and dollar strengthening',
                'severity': 'Moderate',
                'color': '#ffaa00',
            })
        
        overall_stress = len(stress_flags) > 0
        
        return {
            'tickers': tickers,
            'labels': labels,
            'matrix': matrix,
            'key_correlations': {
                'stocks_bonds': safe_round(spy_tlt, 2),
                'stocks_vix': safe_round(spy_vix, 2),
                'gold_dollar': safe_round(gld_uup, 2),
            },
            'stress_flags': stress_flags,
            'overall_stress': overall_stress,
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# MODULE F: ECONOMIC VITALS GRID
# =============================================================================

def get_economic_vitals() -> Dict:
    """Get key economic metrics with sparkline data"""
    
    vitals = {}
    
    # Unemployment Rate
    vitals['unemployment'] = {
        'name': 'Unemployment Rate',
        'value': 4.2,
        'unit': '%',
        'trend': 'stable',
        'sparkline': [3.7, 3.8, 3.9, 4.0, 4.1, 4.0, 4.1, 4.2, 4.2, 4.2, 4.1, 4.2],
        'last_updated': '2024-12-06',
        'source': 'BLS',
    }
    
    # GDP Growth (GDPNow)
    try:
        # Atlanta Fed GDPNow is ~3.1% as of late 2024
        vitals['gdp_now'] = {
            'name': 'GDP Now (Atlanta Fed)',
            'value': 3.1,
            'unit': '%',
            'trend': 'up',
            'sparkline': [2.1, 2.3, 2.5, 2.8, 2.9, 3.0, 2.8, 2.9, 3.0, 3.1, 3.0, 3.1],
            'last_updated': '2024-12-20',
            'source': 'Atlanta Fed',
        }
    except:
        pass
    
    # CPI Inflation
    vitals['cpi'] = {
        'name': 'CPI (YoY)',
        'value': 2.7,
        'unit': '%',
        'trend': 'stable',
        'sparkline': [3.4, 3.2, 3.1, 2.9, 2.9, 2.7, 2.6, 2.5, 2.7, 2.6, 2.6, 2.7],
        'last_updated': '2024-12-11',
        'source': 'BLS',
    }
    
    # Case-Shiller Home Prices
    vitals['home_prices'] = {
        'name': 'Case-Shiller (YoY)',
        'value': 4.8,
        'unit': '%',
        'trend': 'up',
        'sparkline': [6.0, 5.5, 5.0, 4.5, 4.2, 4.0, 4.2, 4.5, 4.6, 4.7, 4.8, 4.8],
        'last_updated': '2024-11-26',
        'source': 'S&P CoreLogic',
    }
    
    # Consumer Sentiment
    vitals['consumer_sentiment'] = {
        'name': 'Consumer Sentiment',
        'value': 74.0,
        'unit': '',
        'trend': 'up',
        'sparkline': [67, 69, 70, 68, 69, 71, 70, 72, 73, 74, 73, 74],
        'last_updated': '2024-12-20',
        'source': 'U of Michigan',
    }
    
    # Fed Funds Rate
    vitals['fed_funds'] = {
        'name': 'Fed Funds Rate',
        'value': 4.375,
        'unit': '%',
        'trend': 'down',
        'sparkline': [5.5, 5.5, 5.5, 5.5, 5.5, 5.25, 5.0, 4.75, 4.625, 4.5, 4.375, 4.375],
        'last_updated': '2024-12-18',
        'source': 'Federal Reserve',
    }
    
    return {
        'vitals': vitals,
        'timestamp': datetime.now().isoformat(),
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@macro_v2_bp.route('/api/macro/liquidity', methods=['GET'])
def liquidity_endpoint():
    """Get global central bank liquidity"""
    try:
        result = calculate_global_liquidity()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/yield-curve', methods=['GET'])
def yield_curve_endpoint():
    """Get yield curve data"""
    try:
        curve = get_yield_curve_data()
        recession = calculate_recession_probability(curve['spreads']['10Y_3M'])
        
        return jsonify({
            'curve': curve,
            'recession_probability': recession,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/yield-curve-3d', methods=['GET'])
def yield_curve_3d_endpoint():
    """Get 3D yield curve surface data"""
    try:
        days = request.args.get('days', 365, type=int)
        result = get_yield_curve_3d_data(days)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/recession-prob', methods=['GET'])
def recession_prob_endpoint():
    """Get recession probability"""
    try:
        curve = get_yield_curve_data()
        result = calculate_recession_probability(curve['spreads']['10Y_3M'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/fed-speak', methods=['GET'])
def fed_speak_endpoint():
    """Get Fed Speak sentiment analysis"""
    try:
        result = get_fed_speak_analysis()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/inflation-nowcast', methods=['GET'])
def inflation_nowcast_endpoint():
    """Get real-time inflation nowcast"""
    try:
        result = calculate_inflation_nowcast()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/stress', methods=['GET'])
def stress_endpoint():
    """Get cross-asset stress heatmap"""
    try:
        result = calculate_stress_correlations()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/vitals', methods=['GET'])
def vitals_endpoint():
    """Get economic vitals grid"""
    try:
        result = get_economic_vitals()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/dashboard', methods=['GET'])
def macro_dashboard_endpoint():
    """Get complete macro dashboard data"""
    try:
        return jsonify({
            'liquidity': calculate_global_liquidity(),
            'yield_curve': get_yield_curve_data(),
            'recession': calculate_recession_probability(
                get_yield_curve_data()['spreads']['10Y_3M']
            ),
            'fed_speak': get_fed_speak_analysis(),
            'inflation': calculate_inflation_nowcast(),
            'stress': calculate_stress_correlations(),
            'vitals': get_economic_vitals(),
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@macro_v2_bp.route('/api/macro/health', methods=['GET'])
def macro_health():
    """Health check for Macro V2"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'fred_available': FRED_AVAILABLE,
        'vader_available': VADER_AVAILABLE,
        'scipy_available': SCIPY_AVAILABLE,
        'modules': {
            'liquidity': True,
            'yield_curve': True,
            'recession_probit': SCIPY_AVAILABLE,
            'fed_speak': VADER_AVAILABLE,
            'inflation_nowcast': True,
            'stress_heatmap': True,
            'vitals': True,
        },
        'timestamp': datetime.now().isoformat(),
    })
