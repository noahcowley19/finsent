# ML Forecaster Module for Portfolio Predictions
# Uses Prophet for time-series forecasting with confidence intervals

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf

# Try to import Prophet, fall back to simple statistical forecast if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed. Using statistical fallback forecaster.")


def safe_round(value, decimals=2):
    """Safely round values, handling None and NaN"""
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    try:
        return round(value, decimals)
    except:
        return None


def get_historical_prices(ticker: str, period: str = '2y') -> Optional[pd.DataFrame]:
    """Fetch historical price data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def prepare_prophet_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Prophet format (ds, y columns)"""
    df = pd.DataFrame()
    df['ds'] = prices.index.tz_localize(None) if prices.index.tz is not None else prices.index
    df['y'] = prices['Close'].values
    return df.reset_index(drop=True)


def statistical_forecast(prices: pd.DataFrame, periods: int = 30) -> Dict:
    """
    Simple statistical forecast using drift model
    Used as fallback when Prophet is not available
    """
    close_prices = prices['Close'].values
    
    # Calculate daily returns
    returns = np.diff(close_prices) / close_prices[:-1]
    
    # Calculate drift (mean return) and volatility
    mean_return = np.mean(returns)
    volatility = np.std(returns)
    
    # Generate future dates
    last_date = prices.index[-1]
    if hasattr(last_date, 'tz_localize'):
        last_date = last_date.tz_localize(None) if last_date.tz is not None else last_date
    
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=periods,
        freq='B'  # Business days
    )
    
    # Geometric Brownian Motion simulation
    last_price = close_prices[-1]
    
    # Generate multiple paths for confidence intervals
    n_simulations = 1000
    all_paths = np.zeros((n_simulations, periods))
    
    np.random.seed(42)
    for i in range(n_simulations):
        path = [last_price]
        for _ in range(periods):
            drift = mean_return
            shock = volatility * np.random.normal()
            new_price = path[-1] * (1 + drift + shock)
            path.append(new_price)
        all_paths[i] = path[1:]
    
    # Calculate percentiles
    forecast_mean = np.mean(all_paths, axis=0)
    forecast_lower_80 = np.percentile(all_paths, 10, axis=0)
    forecast_upper_80 = np.percentile(all_paths, 90, axis=0)
    forecast_lower_95 = np.percentile(all_paths, 2.5, axis=0)
    forecast_upper_95 = np.percentile(all_paths, 97.5, axis=0)
    
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
        'forecast': [safe_round(v, 2) for v in forecast_mean],
        'lower_80': [safe_round(v, 2) for v in forecast_lower_80],
        'upper_80': [safe_round(v, 2) for v in forecast_upper_80],
        'lower_95': [safe_round(v, 2) for v in forecast_lower_95],
        'upper_95': [safe_round(v, 2) for v in forecast_upper_95],
        'method': 'statistical_gbm',
        'current_price': safe_round(last_price, 2),
        'predicted_30d_return': safe_round((forecast_mean[-1] / last_price - 1) * 100, 2),
    }


def prophet_forecast(prices: pd.DataFrame, periods: int = 30) -> Dict:
    """
    Prophet-based time series forecast with confidence intervals
    """
    if not PROPHET_AVAILABLE:
        return statistical_forecast(prices, periods)
    
    try:
        # Prepare data
        df = prepare_prophet_data(prices)
        
        # Initialize and fit Prophet
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95,
            changepoint_prior_scale=0.05
        )
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='B')
        
        # Predict
        forecast = model.predict(future)
        
        # Extract forecast for future dates only
        future_forecast = forecast.tail(periods)
        
        # Get 80% confidence intervals by adjusting
        yhat = future_forecast['yhat'].values
        yhat_lower_95 = future_forecast['yhat_lower'].values
        yhat_upper_95 = future_forecast['yhat_upper'].values
        
        # Estimate 80% intervals (narrower than 95%)
        interval_width_95 = (yhat_upper_95 - yhat_lower_95) / 2
        interval_width_80 = interval_width_95 * 0.7  # Approximate 80% as 70% of 95%
        
        last_price = df['y'].iloc[-1]
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in future_forecast['ds']],
            'forecast': [safe_round(v, 2) for v in yhat],
            'lower_80': [safe_round(v, 2) for v in (yhat - interval_width_80)],
            'upper_80': [safe_round(v, 2) for v in (yhat + interval_width_80)],
            'lower_95': [safe_round(v, 2) for v in yhat_lower_95],
            'upper_95': [safe_round(v, 2) for v in yhat_upper_95],
            'method': 'prophet',
            'current_price': safe_round(last_price, 2),
            'predicted_30d_return': safe_round((yhat[-1] / last_price - 1) * 100, 2),
        }
        
    except Exception as e:
        print(f"Prophet forecast failed: {e}. Falling back to statistical method.")
        return statistical_forecast(prices, periods)


def forecast_ticker(ticker: str, periods: int = 30) -> Optional[Dict]:
    """
    Generate forecast for a single ticker
    """
    prices = get_historical_prices(ticker)
    if prices is None or len(prices) < 60:  # Need at least 60 days of data
        return None
    
    forecast = prophet_forecast(prices, periods)
    forecast['ticker'] = ticker
    
    # Add trend classification
    pred_return = forecast.get('predicted_30d_return', 0) or 0
    if pred_return > 5:
        forecast['trend'] = 'Bullish'
    elif pred_return > 0:
        forecast['trend'] = 'Slightly Bullish'
    elif pred_return > -5:
        forecast['trend'] = 'Slightly Bearish'
    else:
        forecast['trend'] = 'Bearish'
    
    return forecast


def forecast_portfolio(positions: List[Dict], periods: int = 30) -> Dict:
    """
    Generate forecasts for all holdings and aggregate portfolio forecast
    """
    forecasts = {}
    portfolio_forecast = None
    total_value = 0
    
    for pos in positions:
        ticker = pos.get('ticker', '').upper()
        shares = float(pos.get('shares', 0))
        current_value = float(pos.get('current_value', 0))
        total_value += current_value
        
        forecast = forecast_ticker(ticker, periods)
        if forecast:
            forecast['shares'] = shares
            forecast['current_value'] = current_value
            forecasts[ticker] = forecast
    
    # Calculate weighted portfolio forecast
    if forecasts and total_value > 0:
        dates = list(forecasts.values())[0]['dates']
        portfolio_values = np.zeros(len(dates))
        portfolio_lower_95 = np.zeros(len(dates))
        portfolio_upper_95 = np.zeros(len(dates))
        
        for ticker, forecast in forecasts.items():
            shares = forecast['shares']
            weight = forecast['current_value'] / total_value
            
            for i, price in enumerate(forecast['forecast']):
                portfolio_values[i] += shares * price
            for i, price in enumerate(forecast['lower_95']):
                portfolio_lower_95[i] += shares * price
            for i, price in enumerate(forecast['upper_95']):
                portfolio_upper_95[i] += shares * price
        
        portfolio_forecast = {
            'dates': dates,
            'forecast': [safe_round(v, 2) for v in portfolio_values],
            'lower_95': [safe_round(v, 2) for v in portfolio_lower_95],
            'upper_95': [safe_round(v, 2) for v in portfolio_upper_95],
            'current_value': safe_round(total_value, 2),
            'predicted_30d_value': safe_round(portfolio_values[-1], 2),
            'predicted_30d_return': safe_round((portfolio_values[-1] / total_value - 1) * 100, 2),
        }
    
    return {
        'individual_forecasts': forecasts,
        'portfolio_forecast': portfolio_forecast,
        'method': 'prophet' if PROPHET_AVAILABLE else 'statistical_gbm',
        'periods': periods,
    }
