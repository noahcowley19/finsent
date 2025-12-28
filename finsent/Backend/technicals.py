from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from cache import financial_cache, Cache

technicals_bp = Blueprint('technicals', __name__)


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


# =============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# =============================================================================

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal, and Histogram"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_pivot_points(high, low, close):
    """Calculate Pivot Points"""
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


def detect_trend(prices, short_period=20, long_period=50):
    """Detect current trend"""
    sma_short = prices.rolling(short_period).mean()
    sma_long = prices.rolling(long_period).mean()
    
    current_price = prices.iloc[-1]
    short_ma = sma_short.iloc[-1]
    long_ma = sma_long.iloc[-1]
    
    if current_price > short_ma > long_ma:
        return 'Uptrend', 'bullish'
    elif current_price < short_ma < long_ma:
        return 'Downtrend', 'bearish'
    elif abs(current_price - short_ma) / short_ma < 0.02:
        return 'Consolidation', 'neutral'
    else:
        return 'Mixed', 'neutral'


def generate_signals(indicators):
    """Generate buy/sell signals based on indicators"""
    signals = []
    
    # RSI signals
    rsi = indicators.get('rsi')
    if rsi:
        if rsi < 30:
            signals.append({'type': 'buy', 'indicator': 'RSI', 'reason': f'RSI oversold at {rsi:.1f}'})
        elif rsi > 70:
            signals.append({'type': 'sell', 'indicator': 'RSI', 'reason': f'RSI overbought at {rsi:.1f}'})
    
    # MACD signals
    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')
    macd_hist = indicators.get('macd_histogram')
    if macd_hist:
        if macd_hist > 0 and indicators.get('macd_histogram_prev', 0) <= 0:
            signals.append({'type': 'buy', 'indicator': 'MACD', 'reason': 'MACD crossed above signal line'})
        elif macd_hist < 0 and indicators.get('macd_histogram_prev', 0) >= 0:
            signals.append({'type': 'sell', 'indicator': 'MACD', 'reason': 'MACD crossed below signal line'})
    
    # Bollinger Band signals
    bb_position = indicators.get('bb_position')
    if bb_position:
        if bb_position < 0:
            signals.append({'type': 'buy', 'indicator': 'Bollinger Bands', 'reason': 'Price below lower band'})
        elif bb_position > 100:
            signals.append({'type': 'sell', 'indicator': 'Bollinger Bands', 'reason': 'Price above upper band'})
    
    # Stochastic signals
    stoch_k = indicators.get('stochastic_k')
    stoch_d = indicators.get('stochastic_d')
    if stoch_k and stoch_d:
        if stoch_k < 20 and stoch_d < 20:
            signals.append({'type': 'buy', 'indicator': 'Stochastic', 'reason': f'Stochastic oversold ({stoch_k:.1f})'})
        elif stoch_k > 80 and stoch_d > 80:
            signals.append({'type': 'sell', 'indicator': 'Stochastic', 'reason': f'Stochastic overbought ({stoch_k:.1f})'})
    
    # Moving Average signals
    if indicators.get('above_sma_20') and indicators.get('above_sma_50'):
        signals.append({'type': 'hold', 'indicator': 'Moving Averages', 'reason': 'Price above all major MAs - bullish'})
    elif not indicators.get('above_sma_20') and not indicators.get('above_sma_50'):
        signals.append({'type': 'hold', 'indicator': 'Moving Averages', 'reason': 'Price below all major MAs - bearish'})
    
    return signals


# =============================================================================
# API ENDPOINTS
# =============================================================================

@technicals_bp.route('/api/technicals/<ticker>', methods=['GET'])
def get_technical_analysis(ticker):
    """Get comprehensive technical analysis for a ticker"""
    try:
        ticker = ticker.strip().upper()
        
        cache_key = f'technicals_{ticker}'
        cached = financial_cache.get(ticker, cache_key)
        if cached is not None:
            return jsonify(cached)
        
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='1y')
        
        if hist.empty:
            return jsonify({
                'error': f'No data available for {ticker}',
                'error_type': 'data_error'
            }), 400
        
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        
        current_price = close.iloc[-1]
        
        # Calculate all indicators
        indicators = {
            'ticker': ticker,
            'name': info.get('longName') or info.get('shortName') or ticker,
            'current_price': safe_round(current_price, 2),
            'change_pct': safe_round(info.get('regularMarketChangePercent'), 2),
        }
        
        # RSI
        rsi = calculate_rsi(close)
        indicators['rsi'] = safe_round(rsi.iloc[-1], 2)
        indicators['rsi_interpretation'] = (
            'Oversold' if indicators['rsi'] < 30 else
            'Overbought' if indicators['rsi'] > 70 else
            'Neutral'
        )
        
        # MACD
        macd, signal, histogram = calculate_macd(close)
        indicators['macd'] = safe_round(macd.iloc[-1], 2)
        indicators['macd_signal'] = safe_round(signal.iloc[-1], 2)
        indicators['macd_histogram'] = safe_round(histogram.iloc[-1], 2)
        indicators['macd_histogram_prev'] = safe_round(histogram.iloc[-2], 2) if len(histogram) > 1 else None
        indicators['macd_interpretation'] = (
            'Bullish' if indicators['macd_histogram'] > 0 else 'Bearish'
        )
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(close)
        indicators['bb_upper'] = safe_round(upper.iloc[-1], 2)
        indicators['bb_middle'] = safe_round(middle.iloc[-1], 2)
        indicators['bb_lower'] = safe_round(lower.iloc[-1], 2)
        
        # BB Position (0-100 scale, <0 or >100 means outside bands)
        bb_range = upper.iloc[-1] - lower.iloc[-1]
        if bb_range > 0:
            bb_position = ((current_price - lower.iloc[-1]) / bb_range) * 100
            indicators['bb_position'] = safe_round(bb_position, 1)
        
        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        indicators['stochastic_k'] = safe_round(stoch_k.iloc[-1], 2)
        indicators['stochastic_d'] = safe_round(stoch_d.iloc[-1], 2)
        indicators['stochastic_interpretation'] = (
            'Oversold' if indicators['stochastic_k'] < 20 else
            'Overbought' if indicators['stochastic_k'] > 80 else
            'Neutral'
        )
        
        # Moving Averages
        indicators['sma_20'] = safe_round(close.rolling(20).mean().iloc[-1], 2)
        indicators['sma_50'] = safe_round(close.rolling(50).mean().iloc[-1], 2) if len(close) >= 50 else None
        indicators['sma_200'] = safe_round(close.rolling(200).mean().iloc[-1], 2) if len(close) >= 200 else None
        indicators['ema_12'] = safe_round(close.ewm(span=12, adjust=False).mean().iloc[-1], 2)
        indicators['ema_26'] = safe_round(close.ewm(span=26, adjust=False).mean().iloc[-1], 2)
        
        # MA positions
        indicators['above_sma_20'] = current_price > indicators['sma_20'] if indicators['sma_20'] else None
        indicators['above_sma_50'] = current_price > indicators['sma_50'] if indicators['sma_50'] else None
        indicators['above_sma_200'] = current_price > indicators['sma_200'] if indicators['sma_200'] else None
        
        # ATR (volatility)
        atr = calculate_atr(high, low, close)
        indicators['atr'] = safe_round(atr.iloc[-1], 2)
        indicators['atr_percent'] = safe_round((atr.iloc[-1] / current_price) * 100, 2)
        
        # Volume analysis
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        indicators['volume'] = int(current_volume)
        indicators['avg_volume'] = int(avg_volume)
        indicators['volume_ratio'] = safe_round(current_volume / avg_volume, 2) if avg_volume else None
        
        # OBV trend
        obv = calculate_obv(close, volume)
        obv_sma = obv.rolling(20).mean()
        indicators['obv_trend'] = 'Bullish' if obv.iloc[-1] > obv_sma.iloc[-1] else 'Bearish'
        
        # Pivot points (using yesterday's data)
        if len(hist) > 1:
            pivots = calculate_pivot_points(high.iloc[-2], low.iloc[-2], close.iloc[-2])
            indicators['pivot'] = safe_round(pivots['pivot'], 2)
            indicators['r1'] = safe_round(pivots['r1'], 2)
            indicators['r2'] = safe_round(pivots['r2'], 2)
            indicators['s1'] = safe_round(pivots['s1'], 2)
            indicators['s2'] = safe_round(pivots['s2'], 2)
        
        # Trend detection
        trend, sentiment = detect_trend(close)
        indicators['trend'] = trend
        indicators['trend_sentiment'] = sentiment
        
        # 52-week range
        indicators['fifty_two_week_high'] = safe_round(high.max(), 2)
        indicators['fifty_two_week_low'] = safe_round(low.min(), 2)
        range_size = indicators['fifty_two_week_high'] - indicators['fifty_two_week_low']
        if range_size > 0:
            indicators['range_position'] = safe_round(
                ((current_price - indicators['fifty_two_week_low']) / range_size) * 100, 1
            )
        
        # Generate signals
        signals = generate_signals(indicators)
        indicators['signals'] = signals
        indicators['signal_summary'] = {
            'buy_count': len([s for s in signals if s['type'] == 'buy']),
            'sell_count': len([s for s in signals if s['type'] == 'sell']),
            'hold_count': len([s for s in signals if s['type'] == 'hold']),
        }
        
        # Overall sentiment
        buy_count = indicators['signal_summary']['buy_count']
        sell_count = indicators['signal_summary']['sell_count']
        if buy_count > sell_count + 1:
            indicators['overall_sentiment'] = 'Bullish'
        elif sell_count > buy_count + 1:
            indicators['overall_sentiment'] = 'Bearish'
        else:
            indicators['overall_sentiment'] = 'Neutral'
        
        indicators['timestamp'] = datetime.now().isoformat()
        
        financial_cache.set(ticker, cache_key, indicators, Cache.TTL_PRICE)
        return jsonify(indicators)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@technicals_bp.route('/api/technicals/<ticker>/signals', methods=['GET'])
def get_trading_signals(ticker):
    """Get just the trading signals for a ticker"""
    try:
        ticker = ticker.strip().upper()
        
        # Use the full analysis
        stock = yf.Ticker(ticker)
        hist = stock.history(period='6mo')
        
        if hist.empty:
            return jsonify({
                'error': f'No data available for {ticker}',
                'error_type': 'data_error'
            }), 400
        
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        
        current_price = close.iloc[-1]
        
        # Calculate key indicators for signals
        rsi = calculate_rsi(close)
        macd, signal, histogram = calculate_macd(close)
        upper, middle, lower = calculate_bollinger_bands(close)
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        
        bb_range = upper.iloc[-1] - lower.iloc[-1]
        
        indicators = {
            'rsi': safe_round(rsi.iloc[-1], 2),
            'macd_histogram': safe_round(histogram.iloc[-1], 2),
            'macd_histogram_prev': safe_round(histogram.iloc[-2], 2) if len(histogram) > 1 else 0,
            'bb_position': safe_round(((current_price - lower.iloc[-1]) / bb_range) * 100, 1) if bb_range > 0 else 50,
            'stochastic_k': safe_round(stoch_k.iloc[-1], 2),
            'stochastic_d': safe_round(stoch_d.iloc[-1], 2),
            'above_sma_20': current_price > close.rolling(20).mean().iloc[-1],
            'above_sma_50': current_price > close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None,
        }
        
        signals = generate_signals(indicators)
        
        buy_count = len([s for s in signals if s['type'] == 'buy'])
        sell_count = len([s for s in signals if s['type'] == 'sell'])
        
        return jsonify({
            'ticker': ticker,
            'current_price': safe_round(current_price, 2),
            'signals': signals,
            'summary': {
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'recommendation': (
                    'Buy' if buy_count > sell_count + 1 else
                    'Sell' if sell_count > buy_count + 1 else
                    'Hold'
                )
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500


@technicals_bp.route('/api/technicals/<ticker>/summary', methods=['GET'])
def get_technical_summary(ticker):
    """Get a quick technical summary for dashboard display"""
    try:
        ticker = ticker.strip().upper()
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period='3mo')
        
        if hist.empty:
            return jsonify({
                'error': f'No data available for {ticker}',
                'error_type': 'data_error'
            }), 400
        
        close = hist['Close']
        current_price = close.iloc[-1]
        
        # Quick indicators
        rsi = calculate_rsi(close)
        
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        
        trend, sentiment = detect_trend(close)
        
        return jsonify({
            'ticker': ticker,
            'price': safe_round(current_price, 2),
            'rsi': safe_round(rsi.iloc[-1], 1),
            'trend': trend,
            'sentiment': sentiment,
            'above_sma_20': current_price > sma_20,
            'above_sma_50': current_price > sma_50 if sma_50 else None,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': 'server_error'
        }), 500
