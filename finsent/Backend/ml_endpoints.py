# ML Endpoints Blueprint for Portfolio
# Provides API routes for forecasting and sentiment analysis

from flask import Blueprint, request, jsonify
from datetime import datetime

# Create Blueprint
ml_bp = Blueprint('ml', __name__)

# Try to import ML modules
try:
    from ml.forecaster import forecast_portfolio, forecast_ticker
    from ml.sentiment import analyze_portfolio_sentiment, analyze_ticker_sentiment
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML modules not available: {e}")
    ML_AVAILABLE = False

# Import portfolio metrics function
try:
    from portfolio import calculate_portfolio_metrics
except ImportError:
    calculate_portfolio_metrics = None


def safe_round(value, decimals=2):
    """Safely round values"""
    import numpy as np
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    try:
        return round(value, decimals)
    except:
        return None


# =============================================================================
# FORECAST ENDPOINTS
# =============================================================================

@ml_bp.route('/api/ml/forecast', methods=['POST'])
def get_portfolio_forecast():
    """
    Get ML-powered price forecasts for portfolio holdings
    Uses Prophet or statistical fallback for 30-day predictions
    """
    try:
        if not ML_AVAILABLE:
            return jsonify({
                'error': 'ML modules not available',
                'error_type': 'dependency_error'
            }), 503
        
        data = request.get_json()
        positions = data.get('positions', [])
        periods = data.get('periods', 30)
        
        if not positions:
            return jsonify({
                'error': 'No positions provided',
                'error_type': 'validation'
            }), 400
        
        # Get position data if calculate_portfolio_metrics is available
        if calculate_portfolio_metrics:
            portfolio_metrics = calculate_portfolio_metrics(positions)
            positions_data = portfolio_metrics.get('positions', [])
        else:
            positions_data = positions
        
        if not positions_data:
            return jsonify({
                'error': 'Unable to fetch data for positions',
                'error_type': 'data_error'
            }), 400
        
        # Generate forecasts
        result = forecast_portfolio(positions_data, periods)
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@ml_bp.route('/api/ml/forecast/<ticker>', methods=['GET'])
def get_ticker_forecast(ticker: str):
    """Get forecast for a single ticker"""
    try:
        if not ML_AVAILABLE:
            return jsonify({
                'error': 'ML modules not available',
                'error_type': 'dependency_error'
            }), 503
        
        periods = request.args.get('periods', 30, type=int)
        result = forecast_ticker(ticker.upper(), periods)
        
        if result is None:
            return jsonify({
                'error': f'Unable to generate forecast for {ticker}',
                'error_type': 'data_error'
            }), 400
        
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


# =============================================================================
# SENTIMENT ENDPOINTS
# =============================================================================

@ml_bp.route('/api/ml/sentiment', methods=['POST'])
def get_portfolio_sentiment():
    """
    Get sentiment analysis for portfolio holdings
    Uses VADER on news headlines to generate sentiment scores
    """
    try:
        if not ML_AVAILABLE:
            return jsonify({
                'error': 'ML modules not available',
                'error_type': 'dependency_error'
            }), 503
        
        data = request.get_json()
        positions = data.get('positions', [])
        
        if not positions:
            return jsonify({
                'error': 'No positions provided',
                'error_type': 'validation'
            }), 400
        
        # Get position data if calculate_portfolio_metrics is available
        if calculate_portfolio_metrics:
            portfolio_metrics = calculate_portfolio_metrics(positions)
            positions_data = portfolio_metrics.get('positions', [])
        else:
            positions_data = positions
        
        if not positions_data:
            return jsonify({
                'error': 'Unable to fetch data for positions',
                'error_type': 'data_error'
            }), 400
        
        # Analyze sentiment
        result = analyze_portfolio_sentiment(positions_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@ml_bp.route('/api/ml/sentiment/<ticker>', methods=['GET'])
def get_ticker_sentiment(ticker: str):
    """Get sentiment for a single ticker"""
    try:
        if not ML_AVAILABLE:
            return jsonify({
                'error': 'ML modules not available',
                'error_type': 'dependency_error'
            }), 503
        
        result = analyze_ticker_sentiment(ticker.upper())
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@ml_bp.route('/api/ml/health', methods=['GET'])
def ml_health():
    """Check ML module availability"""
    from ml.forecaster import PROPHET_AVAILABLE
    from ml.sentiment import VADER_AVAILABLE
    
    return jsonify({
        'ml_available': ML_AVAILABLE,
        'prophet_available': PROPHET_AVAILABLE,
        'vader_available': VADER_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })
