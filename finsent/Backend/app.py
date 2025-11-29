from flask import Flask, jsonify
from flask_cors import CORS
import os

from sentiment import sentiment_bp
from financials import financials_bp
from insider import insider_bp
from search import search_bp
from dashboard import dashboard_bp


app = Flask(__name__)

CORS(app, origins=["*"])


app.register_blueprint(sentiment_bp)
app.register_blueprint(financials_bp)
app.register_blueprint(insider_bp)
app.register_blueprint(search_bp)
app.register_blueprint(dashboard_bp)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Finsent API is running'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)
