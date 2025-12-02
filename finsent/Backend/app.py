from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os

from sentiment import sentiment_bp
from financials import financials_bp
from insider import insider_bp
from search import search_bp
from portfolio import portfolio_bp

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address



app = Flask(__name__)

CORS(app, origins=["*"])

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["15 per minute"] 
)

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if hasattr(e, 'code'):
        return jsonify({
            'error': str(e),
            'error_type': 'http_error'
        }), e.code

    if request.path.startswith('/api/'):
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500
    raise e

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Frontend')


app.register_blueprint(sentiment_bp)
app.register_blueprint(financials_bp)
app.register_blueprint(insider_bp)
app.register_blueprint(search_bp)
app.register_blueprint(portfolio_bp)

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@limiter.limit("5 per minute)
@app.route('/sentiment')
def sentiment():
    return send_from_directory(FRONTEND_DIR, 'sentiment.html')


@app.route('/financials')
def financials():
    return send_from_directory(FRONTEND_DIR, 'financials.html')


@app.route('/insider')
def insider():
    return send_from_directory(FRONTEND_DIR, 'insider.html')


@app.route('/search')
def search():
    return send_from_directory(FRONTEND_DIR, 'search.html')


@app.route('/portfolio')
def portfolio():
    return send_from_directory(FRONTEND_DIR, 'portfolio.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Finsent API is running'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)





