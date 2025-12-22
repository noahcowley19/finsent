
# Quant Lab - Multi-Factor Intelligence & Prediction Engine
# Created for Caveray by Noah Cowley with Claude

from flask import Blueprint, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import math
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor

quant_lab_bp = Blueprint('quant_lab', __name__)

# Cache for expensive computations
_quant_cache = {}
_quant_cache_time = {}
QUANT_CACHE_TTL = 600  # 10 minutes

# FinBERT for sentiment
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')


def clean_value(value):
    """Clean NaN and Inf values"""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
    return float(value) if isinstance(value, (np.floating, np.integer)) else value


def safe_get(data, key, default=None):
    """Safely get a value from dict or object"""
    try:
        if isinstance(data, dict):
            val = data.get(key, default)
        else:
            val = getattr(data, key, default)
        return clean_value(val)
    except:
        return default


def calculate_percentile(value, values):
    """Calculate percentile rank of a value in a distribution"""
    if value is None or not values:
        return None
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None
    return stats.percentileofscore(valid_values, value)


# ============================================================================
# MULTI-FACTOR COMPOSITE SCORE
# ============================================================================

def calculate_momentum_factors(hist, info):
    """Calculate momentum-based factors"""
    factors = {}
    
    if hist is None or hist.empty or len(hist) < 5:
        return {'momentum_score': None, 'factors': factors}
    
    try:
        close = hist['Close']
        
        # Calculate returns over different periods
        returns_1m = ((close.iloc[-1] / close.iloc[-min(21, len(close))]) - 1) * 100 if len(close) >= 21 else None
        returns_3m = ((close.iloc[-1] / close.iloc[-min(63, len(close))]) - 1) * 100 if len(close) >= 63 else None
        returns_6m = ((close.iloc[-1] / close.iloc[-min(126, len(close))]) - 1) * 100 if len(close) >= 126 else None
        returns_12m = ((close.iloc[-1] / close.iloc[-min(252, len(close))]) - 1) * 100 if len(close) >= 252 else None
        
        factors['returns_1m'] = clean_value(returns_1m)
        factors['returns_3m'] = clean_value(returns_3m)
        factors['returns_6m'] = clean_value(returns_6m)
        factors['returns_12m'] = clean_value(returns_12m)
        
        # RSI (14-day)
        if len(close) >= 15:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            factors['rsi'] = clean_value(rsi.iloc[-1])
        
        # Rate of change
        if len(close) >= 10:
            roc = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100
            factors['roc_10d'] = clean_value(roc)
        
        # Relative Strength vs MA50
        if len(close) >= 50:
            ma50 = close.rolling(50).mean().iloc[-1]
            rs_ma50 = ((close.iloc[-1] / ma50) - 1) * 100
            factors['rs_vs_ma50'] = clean_value(rs_ma50)
        
        # Volume trend
        if 'Volume' in hist.columns and len(hist) >= 20:
            vol = hist['Volume']
            avg_vol_20 = vol.rolling(20).mean().iloc[-1]
            current_vol = vol.iloc[-1]
            vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1
            factors['volume_ratio'] = clean_value(vol_ratio)
        
        # Calculate momentum score (0-100)
        scores = []
        
        # Returns scoring (higher is better)
        for ret, weight in [(returns_1m, 1.5), (returns_3m, 1.2), (returns_6m, 1.0), (returns_12m, 0.8)]:
            if ret is not None:
                # Normalize: -20% to +40% maps to 0-100
                score = max(0, min(100, (ret + 20) / 60 * 100))
                scores.append((score, weight))
        
        # RSI scoring (50-70 is ideal, extremes are bad)
        if factors.get('rsi') is not None:
            rsi_val = factors['rsi']
            if 50 <= rsi_val <= 70:
                rsi_score = 80 + (20 - abs(60 - rsi_val))  # Peak at 60
            elif rsi_val < 30:
                rsi_score = 70  # Oversold can be opportunity
            elif rsi_val > 70:
                rsi_score = 40  # Overbought is risky
            else:
                rsi_score = 60
            scores.append((rsi_score, 1.0))
        
        if scores:
            total_weight = sum(w for _, w in scores)
            momentum_score = sum(s * w for s, w in scores) / total_weight
        else:
            momentum_score = None
        
        return {'momentum_score': clean_value(momentum_score), 'factors': factors}
        
    except Exception as e:
        print(f"Momentum calculation error: {e}")
        return {'momentum_score': None, 'factors': factors}


def calculate_value_factors(info):
    """Calculate value-based factors"""
    factors = {}
    
    try:
        pe = safe_get(info, 'trailingPE')
        forward_pe = safe_get(info, 'forwardPE')
        pb = safe_get(info, 'priceToBook')
        ps = safe_get(info, 'priceToSalesTrailing12Months')
        ev_ebitda = safe_get(info, 'enterpriseToEbitda')
        peg = safe_get(info, 'pegRatio')
        
        factors['pe'] = pe
        factors['forward_pe'] = forward_pe
        factors['pb'] = pb
        factors['ps'] = ps
        factors['ev_ebitda'] = ev_ebitda
        factors['peg'] = peg
        
        # Value score calculation (lower valuations = higher score, with bounds)
        scores = []
        
        # P/E scoring (5-30 is reasonable range)
        if pe is not None and pe > 0:
            if pe < 10:
                pe_score = 90
            elif pe < 15:
                pe_score = 80
            elif pe < 20:
                pe_score = 65
            elif pe < 25:
                pe_score = 50
            elif pe < 35:
                pe_score = 35
            else:
                pe_score = 20
            scores.append((pe_score, 2.0))
        
        # P/B scoring
        if pb is not None and pb > 0:
            if pb < 1:
                pb_score = 85
            elif pb < 2:
                pb_score = 70
            elif pb < 4:
                pb_score = 55
            elif pb < 6:
                pb_score = 40
            else:
                pb_score = 25
            scores.append((pb_score, 1.5))
        
        # PEG scoring (lower is better, 1 is fair)
        if peg is not None and peg > 0:
            if peg < 0.5:
                peg_score = 90
            elif peg < 1:
                peg_score = 75
            elif peg < 1.5:
                peg_score = 55
            elif peg < 2:
                peg_score = 40
            else:
                peg_score = 25
            scores.append((peg_score, 1.5))
        
        if scores:
            total_weight = sum(w for _, w in scores)
            value_score = sum(s * w for s, w in scores) / total_weight
        else:
            value_score = None
        
        return {'value_score': clean_value(value_score), 'factors': factors}
        
    except Exception as e:
        print(f"Value calculation error: {e}")
        return {'value_score': None, 'factors': factors}


def calculate_quality_factors(info):
    """Calculate quality-based factors"""
    factors = {}
    
    try:
        roe = safe_get(info, 'returnOnEquity')
        roa = safe_get(info, 'returnOnAssets')
        gross_margin = safe_get(info, 'grossMargins')
        operating_margin = safe_get(info, 'operatingMargins')
        profit_margin = safe_get(info, 'profitMargins')
        current_ratio = safe_get(info, 'currentRatio')
        debt_equity = safe_get(info, 'debtToEquity')
        
        # Convert to percentages
        if roe is not None: roe *= 100
        if roa is not None: roa *= 100
        if gross_margin is not None: gross_margin *= 100
        if operating_margin is not None: operating_margin *= 100
        if profit_margin is not None: profit_margin *= 100
        
        factors['roe'] = clean_value(roe)
        factors['roa'] = clean_value(roa)
        factors['gross_margin'] = clean_value(gross_margin)
        factors['operating_margin'] = clean_value(operating_margin)
        factors['profit_margin'] = clean_value(profit_margin)
        factors['current_ratio'] = clean_value(current_ratio)
        factors['debt_equity'] = clean_value(debt_equity)
        
        scores = []
        
        # ROE scoring (higher is better, 15%+ is good)
        if roe is not None:
            if roe > 25:
                roe_score = 90
            elif roe > 20:
                roe_score = 80
            elif roe > 15:
                roe_score = 70
            elif roe > 10:
                roe_score = 55
            elif roe > 5:
                roe_score = 40
            elif roe > 0:
                roe_score = 30
            else:
                roe_score = 15
            scores.append((roe_score, 2.0))
        
        # Profit margin scoring
        if profit_margin is not None:
            if profit_margin > 20:
                margin_score = 90
            elif profit_margin > 15:
                margin_score = 75
            elif profit_margin > 10:
                margin_score = 60
            elif profit_margin > 5:
                margin_score = 45
            elif profit_margin > 0:
                margin_score = 30
            else:
                margin_score = 15
            scores.append((margin_score, 1.5))
        
        # Debt/Equity scoring (lower is better)
        if debt_equity is not None:
            if debt_equity < 30:
                de_score = 85
            elif debt_equity < 50:
                de_score = 70
            elif debt_equity < 100:
                de_score = 55
            elif debt_equity < 150:
                de_score = 40
            else:
                de_score = 25
            scores.append((de_score, 1.0))
        
        # Current ratio scoring
        if current_ratio is not None:
            if current_ratio > 2:
                cr_score = 80
            elif current_ratio > 1.5:
                cr_score = 70
            elif current_ratio > 1:
                cr_score = 55
            else:
                cr_score = 30
            scores.append((cr_score, 0.8))
        
        if scores:
            total_weight = sum(w for _, w in scores)
            quality_score = sum(s * w for s, w in scores) / total_weight
        else:
            quality_score = None
        
        return {'quality_score': clean_value(quality_score), 'factors': factors}
        
    except Exception as e:
        print(f"Quality calculation error: {e}")
        return {'quality_score': None, 'factors': factors}


def calculate_growth_factors(info):
    """Calculate growth-based factors"""
    factors = {}
    
    try:
        revenue_growth = safe_get(info, 'revenueGrowth')
        earnings_growth = safe_get(info, 'earningsGrowth')
        earnings_quarterly_growth = safe_get(info, 'earningsQuarterlyGrowth')
        
        if revenue_growth is not None: revenue_growth *= 100
        if earnings_growth is not None: earnings_growth *= 100
        if earnings_quarterly_growth is not None: earnings_quarterly_growth *= 100
        
        factors['revenue_growth'] = clean_value(revenue_growth)
        factors['earnings_growth'] = clean_value(earnings_growth)
        factors['earnings_quarterly_growth'] = clean_value(earnings_quarterly_growth)
        
        scores = []
        
        # Revenue growth scoring
        if revenue_growth is not None:
            if revenue_growth > 30:
                rev_score = 90
            elif revenue_growth > 20:
                rev_score = 80
            elif revenue_growth > 10:
                rev_score = 65
            elif revenue_growth > 5:
                rev_score = 50
            elif revenue_growth > 0:
                rev_score = 40
            else:
                rev_score = 20
            scores.append((rev_score, 1.5))
        
        # Earnings growth scoring
        if earnings_growth is not None:
            if earnings_growth > 30:
                earn_score = 90
            elif earnings_growth > 20:
                earn_score = 80
            elif earnings_growth > 10:
                earn_score = 65
            elif earnings_growth > 0:
                earn_score = 45
            else:
                earn_score = 20
            scores.append((earn_score, 1.5))
        
        if scores:
            total_weight = sum(w for _, w in scores)
            growth_score = sum(s * w for s, w in scores) / total_weight
        else:
            growth_score = None
        
        return {'growth_score': clean_value(growth_score), 'factors': factors}
        
    except Exception as e:
        print(f"Growth calculation error: {e}")
        return {'growth_score': None, 'factors': factors}


def calculate_volatility_factors(hist, info):
    """Calculate volatility and risk factors"""
    factors = {}
    
    try:
        beta = safe_get(info, 'beta')
        factors['beta'] = beta
        
        if hist is not None and not hist.empty and len(hist) >= 20:
            close = hist['Close']
            returns = close.pct_change().dropna()
            
            # Historical volatility (annualized)
            hist_vol = returns.std() * np.sqrt(252) * 100
            factors['historical_volatility'] = clean_value(hist_vol)
            
            # Maximum drawdown
            rolling_max = close.expanding().max()
            drawdown = (close - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            factors['max_drawdown'] = clean_value(max_drawdown)
            
            # Average true range (volatility measure)
            if 'High' in hist.columns and 'Low' in hist.columns:
                high = hist['High']
                low = hist['Low']
                atr = (high - low).rolling(14).mean().iloc[-1]
                atr_percent = (atr / close.iloc[-1]) * 100
                factors['atr_percent'] = clean_value(atr_percent)
        
        scores = []
        
        # Beta scoring (1 is neutral, lower volatility is better for score)
        if beta is not None:
            if beta < 0.5:
                beta_score = 85  # Very low vol
            elif beta < 0.8:
                beta_score = 75
            elif beta < 1.1:
                beta_score = 65  # Market-like
            elif beta < 1.3:
                beta_score = 50
            elif beta < 1.6:
                beta_score = 35
            else:
                beta_score = 20  # Very high vol
            scores.append((beta_score, 1.5))
        
        # Historical vol scoring
        if factors.get('historical_volatility') is not None:
            vol = factors['historical_volatility']
            if vol < 15:
                vol_score = 85
            elif vol < 25:
                vol_score = 70
            elif vol < 35:
                vol_score = 55
            elif vol < 50:
                vol_score = 40
            else:
                vol_score = 25
            scores.append((vol_score, 1.0))
        
        # Drawdown scoring
        if factors.get('max_drawdown') is not None:
            dd = abs(factors['max_drawdown'])
            if dd < 10:
                dd_score = 85
            elif dd < 20:
                dd_score = 70
            elif dd < 30:
                dd_score = 55
            elif dd < 40:
                dd_score = 40
            else:
                dd_score = 25
            scores.append((dd_score, 1.0))
        
        if scores:
            total_weight = sum(w for _, w in scores)
            volatility_score = sum(s * w for s, w in scores) / total_weight
        else:
            volatility_score = None
        
        return {'volatility_score': clean_value(volatility_score), 'factors': factors}
        
    except Exception as e:
        print(f"Volatility calculation error: {e}")
        return {'volatility_score': None, 'factors': factors}


def calculate_technical_factors(hist, info):
    """Calculate technical analysis factors"""
    factors = {}
    
    try:
        if hist is None or hist.empty or len(hist) < 50:
            return {'technical_score': None, 'factors': factors}
        
        close = hist['Close']
        current_price = close.iloc[-1]
        
        # Moving averages
        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        
        factors['ma20'] = clean_value(ma20)
        factors['ma50'] = clean_value(ma50)
        factors['ma200'] = clean_value(ma200)
        factors['above_ma20'] = current_price > ma20 if ma20 else None
        factors['above_ma50'] = current_price > ma50 if ma50 else None
        factors['above_ma200'] = current_price > ma200 if ma200 else None
        
        # 52-week high/low position
        high_52w = safe_get(info, 'fiftyTwoWeekHigh')
        low_52w = safe_get(info, 'fiftyTwoWeekLow')
        if high_52w and low_52w and high_52w > low_52w:
            range_position = (current_price - low_52w) / (high_52w - low_52w) * 100
            factors['range_position'] = clean_value(range_position)
            factors['distance_from_high'] = clean_value(((high_52w - current_price) / high_52w) * 100)
        
        # MACD
        if len(close) >= 26:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            factors['macd'] = clean_value(macd_line.iloc[-1])
            factors['macd_signal'] = clean_value(signal_line.iloc[-1])
            factors['macd_histogram'] = clean_value(macd_hist.iloc[-1])
            factors['macd_bullish'] = macd_line.iloc[-1] > signal_line.iloc[-1]
        
        # Trend strength (ADX-like calculation)
        if len(hist) >= 14 and 'High' in hist.columns and 'Low' in hist.columns:
            high = hist['High']
            low = hist['Low']
            
            # Simplified trend strength
            price_range = high.rolling(14).max() - low.rolling(14).min()
            price_movement = abs(close - close.shift(14))
            trend_strength = (price_movement / price_range * 100).iloc[-1] if price_range.iloc[-1] > 0 else 50
            factors['trend_strength'] = clean_value(min(100, trend_strength))
        
        scores = []
        
        # MA position scoring
        ma_score = 50
        if factors.get('above_ma20'): ma_score += 10
        if factors.get('above_ma50'): ma_score += 15
        if factors.get('above_ma200'): ma_score += 15
        
        # Golden/death cross bonus
        if ma50 and ma200:
            if ma50 > ma200:
                ma_score += 10  # Golden cross territory
            else:
                ma_score -= 10  # Death cross territory
        
        scores.append((min(100, max(0, ma_score)), 2.0))
        
        # Range position scoring
        if factors.get('range_position') is not None:
            rp = factors['range_position']
            if rp > 80:
                rp_score = 50  # Near highs, risky
            elif rp > 60:
                rp_score = 70  # Healthy uptrend
            elif rp > 40:
                rp_score = 60  # Middle
            elif rp > 20:
                rp_score = 70  # Potential bounce
            else:
                rp_score = 55  # Near lows, could be value or falling knife
            scores.append((rp_score, 1.0))
        
        # MACD scoring
        if factors.get('macd_bullish') is not None:
            macd_score = 70 if factors['macd_bullish'] else 40
            if factors.get('macd_histogram'):
                if factors['macd_histogram'] > 0:
                    macd_score += 10
            scores.append((macd_score, 1.5))
        
        if scores:
            total_weight = sum(w for _, w in scores)
            technical_score = sum(s * w for s, w in scores) / total_weight
        else:
            technical_score = None
        
        return {'technical_score': clean_value(technical_score), 'factors': factors}
        
    except Exception as e:
        print(f"Technical calculation error: {e}")
        return {'technical_score': None, 'factors': factors}


def calculate_composite_alpha_score(momentum, value, quality, growth, volatility, technical):
    """Calculate the overall Alpha Score from all factor scores"""
    
    # Factor weights (sum to 1)
    weights = {
        'momentum': 0.20,
        'value': 0.15,
        'quality': 0.20,
        'growth': 0.15,
        'volatility': 0.10,  # Risk-adjusted
        'technical': 0.20,
    }
    
    scores_with_weights = []
    factor_contributions = {}
    
    for factor_name, score in [
        ('momentum', momentum.get('momentum_score')),
        ('value', value.get('value_score')),
        ('quality', quality.get('quality_score')),
        ('growth', growth.get('growth_score')),
        ('volatility', volatility.get('volatility_score')),
        ('technical', technical.get('technical_score')),
    ]:
        if score is not None:
            weight = weights[factor_name]
            scores_with_weights.append((score, weight))
            contribution = score * weight
            factor_contributions[factor_name] = {
                'score': round(score, 1),
                'weight': weight,
                'contribution': round(contribution, 2),
                'status': 'positive' if score >= 60 else 'negative' if score < 40 else 'neutral'
            }
    
    if not scores_with_weights:
        return None, factor_contributions
    
    # Normalize weights for available factors
    total_weight = sum(w for _, w in scores_with_weights)
    alpha_score = sum(s * w for s, w in scores_with_weights) / total_weight
    
    return round(alpha_score, 1), factor_contributions


# ============================================================================
# PRICE FORECAST ENGINE
# ============================================================================

def generate_price_forecast(hist, info, periods=[30, 90, 252]):
    """
    Generate probabilistic price forecasts using Monte Carlo simulation
    and statistical methods
    """
    forecasts = {}
    
    try:
        if hist is None or hist.empty or len(hist) < 60:
            return {'has_forecast': False, 'forecasts': {}, 'scenarios': {}}
        
        close = hist['Close']
        current_price = close.iloc[-1]
        
        # Calculate historical statistics
        returns = close.pct_change().dropna()
        daily_mean = returns.mean()
        daily_std = returns.std()
        
        # Annualized statistics
        annual_mean = daily_mean * 252
        annual_vol = daily_std * np.sqrt(252)
        
        # Monte Carlo simulation parameters
        num_simulations = 1000
        
        for period_days in periods:
            period_label = f"{period_days}d" if period_days < 252 else "1y"
            
            # Generate simulated price paths
            simulated_returns = np.random.normal(
                daily_mean, 
                daily_std, 
                (num_simulations, period_days)
            )
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
            final_prices = current_price * cumulative_returns[:, -1]
            
            # Calculate percentiles
            percentiles = np.percentile(final_prices, [10, 25, 50, 75, 90])
            
            forecasts[period_label] = {
                'days': period_days,
                'current_price': round(current_price, 2),
                'p10': round(percentiles[0], 2),  # 10th percentile (bear case)
                'p25': round(percentiles[1], 2),
                'median': round(percentiles[2], 2),  # 50th percentile (base case)
                'p75': round(percentiles[3], 2),
                'p90': round(percentiles[4], 2),  # 90th percentile (bull case)
                'expected_return': round(((percentiles[2] / current_price) - 1) * 100, 2),
                'upside_potential': round(((percentiles[4] / current_price) - 1) * 100, 2),
                'downside_risk': round(((percentiles[0] / current_price) - 1) * 100, 2),
            }
        
        # Generate scenarios
        scenarios = {
            'bull': {
                'label': 'Bull Case',
                'description': 'Favorable market conditions, positive catalysts',
                'probability': '20%',
                'price_30d': forecasts.get('30d', {}).get('p90'),
                'price_90d': forecasts.get('90d', {}).get('p90'),
                'price_1y': forecasts.get('1y', {}).get('p90'),
            },
            'base': {
                'label': 'Base Case',
                'description': 'Normal market conditions, no major surprises',
                'probability': '60%',
                'price_30d': forecasts.get('30d', {}).get('median'),
                'price_90d': forecasts.get('90d', {}).get('median'),
                'price_1y': forecasts.get('1y', {}).get('median'),
            },
            'bear': {
                'label': 'Bear Case',
                'description': 'Adverse conditions, potential headwinds',
                'probability': '20%',
                'price_30d': forecasts.get('30d', {}).get('p10'),
                'price_90d': forecasts.get('90d', {}).get('p10'),
                'price_1y': forecasts.get('1y', {}).get('p10'),
            }
        }
        
        # Trend analysis
        trend_data = {
            'annual_expected_return': round(annual_mean * 100, 2),
            'annual_volatility': round(annual_vol * 100, 2),
            'sharpe_estimate': round((annual_mean - 0.02) / annual_vol, 2) if annual_vol > 0 else None,
            'trend_direction': 'bullish' if daily_mean > 0 else 'bearish',
            'volatility_regime': 'high' if annual_vol > 0.30 else 'low' if annual_vol < 0.15 else 'normal'
        }
        
        return {
            'has_forecast': True,
            'forecasts': forecasts,
            'scenarios': scenarios,
            'trend': trend_data,
            'methodology': 'Monte Carlo simulation with 1000 paths based on historical return distribution'
        }
        
    except Exception as e:
        print(f"Forecast error: {e}")
        return {'has_forecast': False, 'forecasts': {}, 'scenarios': {}}


# ============================================================================
# RISK ANALYSIS
# ============================================================================

def calculate_risk_metrics(hist, info):
    """Calculate comprehensive risk metrics"""
    
    try:
        if hist is None or hist.empty or len(hist) < 60:
            return {'has_risk_data': False}
        
        close = hist['Close']
        returns = close.pct_change().dropna()
        
        # Basic statistics
        daily_mean = returns.mean()
        daily_std = returns.std()
        annual_vol = daily_std * np.sqrt(252) * 100
        
        # Value at Risk (VaR)
        var_95_1d = np.percentile(returns, 5) * 100
        var_99_1d = np.percentile(returns, 1) * 100
        var_95_30d = var_95_1d * np.sqrt(30)
        var_99_30d = var_99_1d * np.sqrt(30)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Drawdown analysis
        rolling_max = close.expanding().max()
        drawdowns = (close - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        current_drawdown = drawdowns.iloc[-1]
        
        # Find drawdown duration
        in_drawdown = drawdowns < 0
        if in_drawdown.iloc[-1]:
            drawdown_start_idx = in_drawdown[::-1].idxmin()
            days_in_drawdown = (close.index[-1] - drawdown_start_idx).days
        else:
            days_in_drawdown = 0
        
        # Tail risk (kurtosis)
        kurtosis = returns.kurtosis()
        skewness = returns.skew()
        
        # Upside/Downside beta calculation
        beta = safe_get(info, 'beta')
        
        # Calculate upside/downside capture
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        upside_volatility = positive_returns.std() * np.sqrt(252) * 100 if len(positive_returns) > 0 else None
        downside_volatility = negative_returns.std() * np.sqrt(252) * 100 if len(negative_returns) > 0 else None
        
        # Risk score (0-100, lower risk = higher score)
        risk_components = []
        
        if annual_vol:
            if annual_vol < 15:
                vol_score = 90
            elif annual_vol < 25:
                vol_score = 70
            elif annual_vol < 40:
                vol_score = 50
            else:
                vol_score = 30
            risk_components.append((vol_score, 1.5))
        
        if max_drawdown:
            dd = abs(max_drawdown)
            if dd < 15:
                dd_score = 90
            elif dd < 25:
                dd_score = 70
            elif dd < 40:
                dd_score = 50
            else:
                dd_score = 30
            risk_components.append((dd_score, 1.5))
        
        if beta:
            if beta < 0.7:
                beta_score = 85
            elif beta < 1.0:
                beta_score = 70
            elif beta < 1.3:
                beta_score = 55
            else:
                beta_score = 35
            risk_components.append((beta_score, 1.0))
        
        if risk_components:
            total_weight = sum(w for _, w in risk_components)
            risk_score = sum(s * w for s, w in risk_components) / total_weight
        else:
            risk_score = 50
        
        # Risk level determination
        if risk_score >= 75:
            risk_level = 'Low'
            risk_status = 'positive'
        elif risk_score >= 50:
            risk_level = 'Moderate'
            risk_status = 'neutral'
        elif risk_score >= 30:
            risk_level = 'High'
            risk_status = 'warning'
        else:
            risk_level = 'Extreme'
            risk_status = 'negative'
        
        return {
            'has_risk_data': True,
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'risk_status': risk_status,
            'volatility': {
                'annual': round(annual_vol, 2),
                'daily': round(daily_std * 100, 2),
                'upside': round(upside_volatility, 2) if upside_volatility else None,
                'downside': round(downside_volatility, 2) if downside_volatility else None,
            },
            'var': {
                'var_95_1d': round(var_95_1d, 2),
                'var_99_1d': round(var_99_1d, 2),
                'var_95_30d': round(var_95_30d, 2),
                'var_99_30d': round(var_99_30d, 2),
                'cvar_95': round(cvar_95, 2),
            },
            'drawdown': {
                'max_drawdown': round(max_drawdown, 2),
                'current_drawdown': round(current_drawdown, 2),
                'days_in_drawdown': days_in_drawdown,
            },
            'distribution': {
                'skewness': round(skewness, 3),
                'kurtosis': round(kurtosis, 3),
                'fat_tails': kurtosis > 3,
                'skew_direction': 'left' if skewness < -0.5 else 'right' if skewness > 0.5 else 'symmetric'
            },
            'beta': round(beta, 2) if beta else None,
        }
        
    except Exception as e:
        print(f"Risk calculation error: {e}")
        return {'has_risk_data': False}


# ============================================================================
# TECHNICAL PATTERN RECOGNITION
# ============================================================================

def detect_chart_patterns(hist):
    """Detect common chart patterns using price action analysis"""
    
    patterns = []
    
    try:
        if hist is None or hist.empty or len(hist) < 60:
            return {'patterns': [], 'support_resistance': {}}
        
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        
        # Get recent data for pattern detection
        recent_close = close.tail(60).values
        recent_high = high.tail(60).values
        recent_low = low.tail(60).values
        
        current_price = close.iloc[-1]
        
        # --- Support and Resistance Detection ---
        # Find local maxima and minima
        def find_peaks(data, order=5):
            peaks = []
            for i in range(order, len(data) - order):
                if all(data[i] >= data[i-j] for j in range(1, order+1)) and \
                   all(data[i] >= data[i+j] for j in range(1, order+1)):
                    peaks.append((i, data[i]))
            return peaks
        
        def find_troughs(data, order=5):
            troughs = []
            for i in range(order, len(data) - order):
                if all(data[i] <= data[i-j] for j in range(1, order+1)) and \
                   all(data[i] <= data[i+j] for j in range(1, order+1)):
                    troughs.append((i, data[i]))
            return troughs
        
        resistance_levels = find_peaks(recent_high, order=3)
        support_levels = find_troughs(recent_low, order=3)
        
        # Cluster nearby levels
        def cluster_levels(levels, threshold=0.02):
            if not levels:
                return []
            levels = sorted([l[1] for l in levels])
            clusters = [[levels[0]]]
            for level in levels[1:]:
                if (level - clusters[-1][-1]) / clusters[-1][-1] < threshold:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            return [sum(c) / len(c) for c in clusters]
        
        resistance_prices = cluster_levels(resistance_levels)[-3:] if resistance_levels else []
        support_prices = cluster_levels(support_levels)[:3] if support_levels else []
        
        # Find nearest support/resistance
        nearest_resistance = min([r for r in resistance_prices if r > current_price], default=None)
        nearest_support = max([s for s in support_prices if s < current_price], default=None)
        
        # --- Pattern Detection ---
        
        # Double Bottom Detection
        if len(support_levels) >= 2:
            last_two_troughs = sorted(support_levels, key=lambda x: x[0])[-2:]
            if len(last_two_troughs) == 2:
                trough1, trough2 = last_two_troughs
                price_diff = abs(trough1[1] - trough2[1]) / trough1[1]
                if price_diff < 0.03:  # Within 3% of each other
                    neckline = max(recent_high[trough1[0]:trough2[0]])
                    if current_price > trough2[1] * 1.02:  # Price bounced
                        patterns.append({
                            'name': 'Double Bottom',
                            'type': 'bullish_reversal',
                            'confidence': 70 if current_price > neckline else 55,
                            'status': 'positive',
                            'description': 'Bullish reversal pattern - price found support at similar level twice',
                            'target': round(neckline + (neckline - trough1[1]), 2),
                            'stop_loss': round(min(trough1[1], trough2[1]) * 0.98, 2)
                        })
        
        # Double Top Detection
        if len(resistance_levels) >= 2:
            last_two_peaks = sorted(resistance_levels, key=lambda x: x[0])[-2:]
            if len(last_two_peaks) == 2:
                peak1, peak2 = last_two_peaks
                price_diff = abs(peak1[1] - peak2[1]) / peak1[1]
                if price_diff < 0.03:
                    neckline = min(recent_low[peak1[0]:peak2[0]])
                    if current_price < peak2[1] * 0.98:
                        patterns.append({
                            'name': 'Double Top',
                            'type': 'bearish_reversal',
                            'confidence': 70 if current_price < neckline else 55,
                            'status': 'negative',
                            'description': 'Bearish reversal pattern - price failed to break resistance twice',
                            'target': round(neckline - (peak1[1] - neckline), 2),
                            'stop_loss': round(max(peak1[1], peak2[1]) * 1.02, 2)
                        })
        
        # Trend Detection
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        
        # Golden/Death Cross
        if ma200:
            ma50_prev = close.rolling(50).mean().iloc[-10]
            ma200_prev = close.rolling(200).mean().iloc[-10]
            
            if ma50 > ma200 and ma50_prev <= ma200_prev:
                patterns.append({
                    'name': 'Golden Cross',
                    'type': 'bullish_trend',
                    'confidence': 75,
                    'status': 'positive',
                    'description': '50-day MA crossed above 200-day MA - strong bullish signal'
                })
            elif ma50 < ma200 and ma50_prev >= ma200_prev:
                patterns.append({
                    'name': 'Death Cross',
                    'type': 'bearish_trend',
                    'confidence': 75,
                    'status': 'negative',
                    'description': '50-day MA crossed below 200-day MA - strong bearish signal'
                })
        
        # Uptrend/Downtrend
        if current_price > ma20 > ma50:
            patterns.append({
                'name': 'Uptrend',
                'type': 'bullish_trend',
                'confidence': 65,
                'status': 'positive',
                'description': 'Price above rising moving averages - bullish trend intact'
            })
        elif current_price < ma20 < ma50:
            patterns.append({
                'name': 'Downtrend',
                'type': 'bearish_trend',
                'confidence': 65,
                'status': 'negative',
                'description': 'Price below declining moving averages - bearish trend intact'
            })
        
        # Breakout Detection
        if nearest_resistance and current_price > nearest_resistance * 0.99:
            patterns.append({
                'name': 'Resistance Breakout',
                'type': 'bullish_breakout',
                'confidence': 60,
                'status': 'positive',
                'description': f'Price breaking above resistance at ${nearest_resistance:.2f}'
            })
        
        if nearest_support and current_price < nearest_support * 1.01:
            patterns.append({
                'name': 'Support Breakdown',
                'type': 'bearish_breakdown',
                'confidence': 60,
                'status': 'negative',
                'description': f'Price breaking below support at ${nearest_support:.2f}'
            })
        
        return {
            'patterns': patterns,
            'support_resistance': {
                'resistance_levels': [round(r, 2) for r in resistance_prices],
                'support_levels': [round(s, 2) for s in support_prices],
                'nearest_resistance': round(nearest_resistance, 2) if nearest_resistance else None,
                'nearest_support': round(nearest_support, 2) if nearest_support else None,
                'distance_to_resistance': round(((nearest_resistance / current_price) - 1) * 100, 2) if nearest_resistance else None,
                'distance_to_support': round(((nearest_support / current_price) - 1) * 100, 2) if nearest_support else None,
            }
        }
        
    except Exception as e:
        print(f"Pattern detection error: {e}")
        return {'patterns': [], 'support_resistance': {}}


# ============================================================================
# SIGNAL SYNTHESIS & RECOMMENDATION
# ============================================================================

def synthesize_signals(alpha_score, factor_contributions, forecast, risk, patterns, info):
    """Synthesize all signals into a coherent recommendation"""
    
    try:
        current_price = safe_get(info, 'currentPrice') or safe_get(info, 'regularMarketPrice')
        
        # Determine signal strength
        if alpha_score is None:
            return {
                'action': 'Hold',
                'confidence': 'Low',
                'status': 'neutral',
                'thesis': ['Insufficient data to generate recommendation'],
                'bull_case': [],
                'bear_case': [],
            }
        
        # Calculate action based on alpha score
        if alpha_score >= 75:
            action = 'Strong Buy'
            action_status = 'positive'
        elif alpha_score >= 60:
            action = 'Buy'
            action_status = 'positive'
        elif alpha_score >= 45:
            action = 'Hold'
            action_status = 'neutral'
        elif alpha_score >= 30:
            action = 'Sell'
            action_status = 'negative'
        else:
            action = 'Strong Sell'
            action_status = 'negative'
        
        # Determine confidence
        available_factors = sum(1 for f in factor_contributions.values() if f['score'] is not None)
        factor_agreement = sum(1 for f in factor_contributions.values() 
                             if f.get('score') is not None and 
                             ((alpha_score >= 50 and f['score'] >= 50) or 
                              (alpha_score < 50 and f['score'] < 50)))
        
        if available_factors >= 5 and factor_agreement >= 4:
            confidence = 'High'
        elif available_factors >= 4 and factor_agreement >= 3:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        # Generate thesis points
        thesis = []
        bull_case = []
        bear_case = []
        
        # Add factor-based thesis points
        for factor_name, factor_data in factor_contributions.items():
            score = factor_data.get('score')
            if score is None:
                continue
                
            if score >= 70:
                point = f"Strong {factor_name.replace('_', ' ')} profile (Score: {score})"
                thesis.append(point)
                bull_case.append(point)
            elif score >= 55:
                bull_case.append(f"Positive {factor_name.replace('_', ' ')} indicators")
            elif score <= 30:
                point = f"Weak {factor_name.replace('_', ' ')} metrics (Score: {score})"
                thesis.append(point)
                bear_case.append(point)
            elif score <= 45:
                bear_case.append(f"Concerning {factor_name.replace('_', ' ')} signals")
        
        # Add pattern-based insights
        if patterns and patterns.get('patterns'):
            for pattern in patterns['patterns'][:2]:
                if pattern['status'] == 'positive':
                    bull_case.append(f"{pattern['name']}: {pattern['description']}")
                else:
                    bear_case.append(f"{pattern['name']}: {pattern['description']}")
        
        # Add risk-based insights
        if risk and risk.get('has_risk_data'):
            if risk['risk_level'] == 'Low':
                bull_case.append(f"Low risk profile with {risk['volatility']['annual']:.1f}% annual volatility")
            elif risk['risk_level'] in ['High', 'Extreme']:
                bear_case.append(f"Elevated risk with {risk['volatility']['annual']:.1f}% volatility")
        
        # Add forecast-based insights
        if forecast and forecast.get('has_forecast'):
            year_forecast = forecast.get('forecasts', {}).get('1y', {})
            if year_forecast:
                expected_return = year_forecast.get('expected_return')
                if expected_return and expected_return > 15:
                    bull_case.append(f"Monte Carlo projects {expected_return:.1f}% median return over 1 year")
                elif expected_return and expected_return < -5:
                    bear_case.append(f"Monte Carlo projects {expected_return:.1f}% median return over 1 year")
        
        # Limit thesis to top 4 points
        thesis = thesis[:4]
        bull_case = bull_case[:4]
        bear_case = bear_case[:4]
        
        # Generate position guidance
        position_guidance = {}
        if current_price and risk and risk.get('has_risk_data'):
            # Stop loss based on volatility
            daily_vol = risk['volatility'].get('daily', 2)
            atr_stop = current_price * (1 - daily_vol * 2 / 100)
            
            # Support-based stop
            if patterns and patterns.get('support_resistance', {}).get('nearest_support'):
                support_stop = patterns['support_resistance']['nearest_support'] * 0.98
                stop_loss = max(atr_stop, support_stop)
            else:
                stop_loss = atr_stop
            
            # Target prices based on forecast
            if forecast and forecast.get('forecasts'):
                target_30d = forecast['forecasts'].get('30d', {}).get('p75')
                target_90d = forecast['forecasts'].get('90d', {}).get('p75')
                target_1y = forecast['forecasts'].get('1y', {}).get('p75')
            else:
                target_30d = current_price * 1.05
                target_90d = current_price * 1.10
                target_1y = current_price * 1.15
            
            position_guidance = {
                'stop_loss': round(stop_loss, 2),
                'target_short': round(target_30d, 2) if target_30d else None,
                'target_medium': round(target_90d, 2) if target_90d else None,
                'target_long': round(target_1y, 2) if target_1y else None,
                'risk_reward_ratio': round((target_90d - current_price) / (current_price - stop_loss), 2) if target_90d and stop_loss < current_price else None,
            }
        
        return {
            'action': action,
            'action_status': action_status,
            'confidence': confidence,
            'thesis': thesis,
            'bull_case': bull_case,
            'bear_case': bear_case,
            'position_guidance': position_guidance,
        }
        
    except Exception as e:
        print(f"Signal synthesis error: {e}")
        return {
            'action': 'Hold',
            'confidence': 'Low',
            'status': 'neutral',
            'thesis': ['Error generating recommendation'],
            'bull_case': [],
            'bear_case': [],
        }


# ============================================================================
# MAIN API ENDPOINT
# ============================================================================

@quant_lab_bp.route('/api/quant-lab', methods=['POST'])
def quant_lab_analysis():
    """Main Quant Lab analysis endpoint"""
    
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400
        
        # Check cache
        cache_key = f"quant_{ticker}"
        current_time = time.time()
        
        if cache_key in _quant_cache:
            if current_time - _quant_cache_time.get(cache_key, 0) < QUANT_CACHE_TTL:
                return jsonify(_quant_cache[cache_key])
        
        # Fetch data
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or info.get('regularMarketPrice') is None:
            return jsonify({'error': f'No data found for ticker {ticker}'}), 404
        
        # Get historical data (2 years for pattern detection and statistics)
        hist = stock.history(period='2y', interval='1d')
        
        # Company info
        company = {
            'name': info.get('longName') or info.get('shortName') or ticker,
            'ticker': ticker,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'price': clean_value(info.get('currentPrice') or info.get('regularMarketPrice')),
            'price_display': f"${clean_value(info.get('currentPrice') or info.get('regularMarketPrice')):.2f}" if info.get('currentPrice') or info.get('regularMarketPrice') else 'N/A',
            'change_percent': clean_value(info.get('regularMarketChangePercent', 0) * 100) if info.get('regularMarketChangePercent') else None,
            'market_cap': clean_value(info.get('marketCap')),
            'market_cap_display': format_market_cap(info.get('marketCap')),
        }
        
        # Calculate all factor scores
        momentum = calculate_momentum_factors(hist, info)
        value = calculate_value_factors(info)
        quality = calculate_quality_factors(info)
        growth = calculate_growth_factors(info)
        volatility = calculate_volatility_factors(hist, info)
        technical = calculate_technical_factors(hist, info)
        
        # Calculate composite alpha score
        alpha_score, factor_contributions = calculate_composite_alpha_score(
            momentum, value, quality, growth, volatility, technical
        )
        
        # Generate price forecast
        forecast = generate_price_forecast(hist, info)
        
        # Calculate risk metrics
        risk = calculate_risk_metrics(hist, info)
        
        # Detect technical patterns
        patterns = detect_chart_patterns(hist)
        
        # Synthesize all signals
        signal = synthesize_signals(alpha_score, factor_contributions, forecast, risk, patterns, info)
        
        # Build response
        response = {
            'ticker': ticker,
            'company': company,
            'alpha_score': {
                'score': alpha_score,
                'display': f"{alpha_score}" if alpha_score else 'N/A',
                'status': 'positive' if alpha_score and alpha_score >= 60 else 'negative' if alpha_score and alpha_score < 40 else 'neutral',
                'percentile': None,  # Could compare to market
            },
            'factor_scores': {
                'momentum': {
                    'score': momentum.get('momentum_score'),
                    'factors': momentum.get('factors', {}),
                    'status': 'positive' if momentum.get('momentum_score') and momentum['momentum_score'] >= 60 else 'negative' if momentum.get('momentum_score') and momentum['momentum_score'] < 40 else 'neutral'
                },
                'value': {
                    'score': value.get('value_score'),
                    'factors': value.get('factors', {}),
                    'status': 'positive' if value.get('value_score') and value['value_score'] >= 60 else 'negative' if value.get('value_score') and value['value_score'] < 40 else 'neutral'
                },
                'quality': {
                    'score': quality.get('quality_score'),
                    'factors': quality.get('factors', {}),
                    'status': 'positive' if quality.get('quality_score') and quality['quality_score'] >= 60 else 'negative' if quality.get('quality_score') and quality['quality_score'] < 40 else 'neutral'
                },
                'growth': {
                    'score': growth.get('growth_score'),
                    'factors': growth.get('factors', {}),
                    'status': 'positive' if growth.get('growth_score') and growth['growth_score'] >= 60 else 'negative' if growth.get('growth_score') and growth['growth_score'] < 40 else 'neutral'
                },
                'volatility': {
                    'score': volatility.get('volatility_score'),
                    'factors': volatility.get('factors', {}),
                    'status': 'positive' if volatility.get('volatility_score') and volatility['volatility_score'] >= 60 else 'negative' if volatility.get('volatility_score') and volatility['volatility_score'] < 40 else 'neutral'
                },
                'technical': {
                    'score': technical.get('technical_score'),
                    'factors': technical.get('factors', {}),
                    'status': 'positive' if technical.get('technical_score') and technical['technical_score'] >= 60 else 'negative' if technical.get('technical_score') and technical['technical_score'] < 40 else 'neutral'
                },
            },
            'factor_contributions': factor_contributions,
            'price_forecast': forecast,
            'risk_analysis': risk,
            'technical_analysis': {
                'patterns': patterns.get('patterns', []),
                'support_resistance': patterns.get('support_resistance', {}),
            },
            'signal': signal,
            'timestamp': datetime.now().isoformat(),
            'disclaimer': 'This analysis is for informational purposes only and should not be considered financial advice. Past performance does not guarantee future results.'
        }
        
        # Cache response
        _quant_cache[cache_key] = response
        _quant_cache_time[cache_key] = current_time
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Quant Lab error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def format_market_cap(value):
    """Format market cap for display"""
    if value is None:
        return 'N/A'
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    if value >= 1e6:
        return f"${value/1e6:.2f}M"
    return f"${value:,.0f}"


@quant_lab_bp.route('/api/quant-lab/health', methods=['GET'])
def quant_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'features': [
            'multi_factor_scoring',
            'price_forecast',
            'risk_analysis',
            'technical_patterns',
            'signal_synthesis'
        ],
        'cache_ttl': QUANT_CACHE_TTL
    })
