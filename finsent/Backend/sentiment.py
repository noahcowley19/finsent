# Made by Noah C, debugged by Claude.ai
# Finsent Sentiment Analysis Backend

from flask import Blueprint, request, jsonify
import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import yfinance as yf
import math

sentiment_bp = Blueprint('sentiment', __name__)

# ==================== CONFIGURATION ====================

# Hugging Face Inference API for FinBERT
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')

# Initialize VADER for sentiment analysis
vader_analyzer = SentimentIntensityAnalyzer()

# Default tickers for social screening
DEFAULT_TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA']

# Company name mappings for common tickers
COMPANY_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'GOOG': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corp.',
    'META': 'Meta Platforms',
    'TSLA': 'Tesla Inc.',
    'AMD': 'AMD Inc.',
    'INTC': 'Intel Corp.',
    'NFLX': 'Netflix Inc.',
    'BA': 'Boeing Co.',
    'DIS': 'Walt Disney Co.',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.',
    'JPM': 'JPMorgan Chase',
    'WMT': 'Walmart Inc.',
    'COIN': 'Coinbase Global',
    'PLTR': 'Palantir Tech.',
    'F': 'Ford Motor Co.',
    'GM': 'General Motors',
    'AAL': 'American Airlines',
    'BAC': 'Bank of America',
    'C': 'Citigroup Inc.',
    'PYPL': 'PayPal Holdings',
    'UBER': 'Uber Technologies',
    'LYFT': 'Lyft Inc.',
    'SNAP': 'Snap Inc.',
    'TWTR': 'Twitter Inc.',
    'SQ': 'Block Inc.',
    'SHOP': 'Shopify Inc.',
    'ROKU': 'Roku Inc.',
    'ZM': 'Zoom Video',
    'DOCU': 'DocuSign Inc.',
    'CRM': 'Salesforce Inc.',
    'ORCL': 'Oracle Corp.',
    'IBM': 'IBM Corp.',
    'CSCO': 'Cisco Systems',
    'QCOM': 'Qualcomm Inc.',
    'AVGO': 'Broadcom Inc.',
    'TXN': 'Texas Instruments',
    'MU': 'Micron Technology',
    'LRCX': 'Lam Research',
    'AMAT': 'Applied Materials',
    'KLAC': 'KLA Corp.',
    'ADI': 'Analog Devices',
    'MRVL': 'Marvell Technology',
    'ON': 'ON Semiconductor',
    'SWKS': 'Skyworks Solutions',
    'QRVO': 'Qorvo Inc.',
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ Trust',
    'IWM': 'iShares Russell 2000',
    'DIA': 'SPDR Dow Jones ETF',
    'VTI': 'Vanguard Total Stock',
    'VOO': 'Vanguard S&P 500',
    'ARKK': 'ARK Innovation ETF',
    'XLF': 'Financial Select SPDR',
    'XLE': 'Energy Select SPDR',
    'XLK': 'Technology Select SPDR',
}

# ==================== CACHING ====================

# Track FinBERT availability
_finbert_available = None
_finbert_last_check = 0

# Cache for social screening data
_screening_cache = {}
_screening_cache_ttl = 300  # 5 minutes

# Cache for Sentdex data
_sentdex_cache = {
    'data': None,
    'timestamp': 0,
    'ttl': 600  # 10 minutes
}


def get_cache_key(tickers):
    """Generate a cache key from ticker list."""
    return ':'.join(sorted([t.upper() for t in tickers]))


# ==================== FINBERT FUNCTIONS ====================

def check_finbert_availability():
    """Check if FinBERT API is responding."""
    global _finbert_available, _finbert_last_check
    
    if _finbert_available is not None and (time.time() - _finbert_last_check) < 300:
        return _finbert_available
    
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": "Stock price increased today."},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result and isinstance(result, list):
                _finbert_available = True
                _finbert_last_check = time.time()
                return True
        elif response.status_code == 503:
            data = response.json()
            wait_time = min(data.get('estimated_time', 20), 25)
            print(f"FinBERT: Cold start, waiting {wait_time}s...")
            time.sleep(wait_time)
            
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": "Stock price increased today."},
                timeout=30
            )
            if response.status_code == 200:
                _finbert_available = True
                _finbert_last_check = time.time()
                return True
        
        _finbert_available = False
        _finbert_last_check = time.time()
        return False
        
    except Exception as e:
        print(f"FinBERT: Error - {e}")
        _finbert_available = False
        _finbert_last_check = time.time()
        return False


def query_finbert(text):
    """Query FinBERT model via Hugging Face Inference API."""
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text[:1000]},
            timeout=20
        )
        
        if response.status_code != 200:
            return None
            
        result = response.json()
        
        if result and isinstance(result, list) and len(result) > 0:
            predictions = result[0] if isinstance(result[0], list) else result
            best = max(predictions, key=lambda x: x['score'])
            label = best['label'].lower()
            score = best['score']
            
            if label == 'positive':
                polarity = score
            elif label == 'negative':
                polarity = -score
            else:
                polarity = 0.0
            
            return polarity, label.capitalize()
            
    except Exception as e:
        print(f"FinBERT query error: {e}")
        return None
    
    return None


def query_vader(text):
    """Analyze sentiment using VADER."""
    scores = vader_analyzer.polarity_scores(text)
    polarity = scores['compound']
    
    if polarity > 0.05:
        sentiment = 'Positive'
    elif polarity < -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return polarity, sentiment


def analyze_sentiment(text, use_finbert=True):
    """Analyze sentiment - tries FinBERT first, falls back to VADER."""
    if use_finbert:
        result = query_finbert(text)
        if result is not None:
            return result[0], result[1], 'FinBERT'
    
    polarity, sentiment = query_vader(text)
    return polarity, sentiment, 'VADER'


# ==================== NEWS FETCHING ====================

def fetch_article_content(url):
    """Fetch and extract article text content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=3, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content[:50000], 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        paragraphs = soup.find_all('p')[:5]
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if len(text) > 50:
            return text[:1000]
        return ""
    except:
        return ""


def fetch_news_batch(queries, num_articles_per_query):
    """Fetch news articles from Google News RSS."""
    all_articles = []
    
    for query in queries:
        try:
            rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            for item in feed.entries[:num_articles_per_query]:
                all_articles.append({
                    "title": item.title,
                    "link": item.link,
                    "published": item.get('published', ''),
                    "summary": (item.get('summary', '') or item.get('description', ''))[:500]
                })
        except:
            continue
    
    # Remove duplicates
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title_lower = article['title'].lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(article)
    
    unique_articles = unique_articles[:50]
    
    def scrape_content(article):
        content = fetch_article_content(article['link'])
        if not content:
            content = article['summary']
        return {
            "title": article['title'],
            "link": article['link'],
            "published": article['published'],
            "content": content[:1000]
        }
    
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(scrape_content, article): article for article in unique_articles}
        
        for future in as_completed(futures, timeout=15):
            try:
                result = future.result(timeout=2)
                results.append(result)
            except:
                article = futures[future]
                results.append({
                    "title": article['title'],
                    "link": article['link'],
                    "published": article['published'],
                    "content": article['summary'][:500]
                })
    
    return results


def analyze_ticker_sentiment(ticker, num_articles_per_query=10):
    """Main function to analyze sentiment for a ticker/asset."""
    use_finbert = check_finbert_availability()
    
    queries = [
        f"{ticker} news",
        f"{ticker} market",
        f"{ticker} financial",
        f"{ticker} analysis",
        f"{ticker} trend",
    ]

    articles = fetch_news_batch(queries, num_articles_per_query)
    
    if not articles:
        return {
            'ticker': ticker,
            'articles': [],
            'summary': {
                'Positive': {'count': 0, 'percentage': 0},
                'Negative': {'count': 0, 'percentage': 0},
                'Neutral': {'count': 0, 'percentage': 0}
            },
            'total_articles': 0,
            'model': 'N/A'
        }

    analyzed_articles = []
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    models_used = set()
    
    for article in articles:
        text_to_analyze = f"{article['title']} {article['content']}"
        polarity, sentiment, model = analyze_sentiment(text_to_analyze, use_finbert)
        
        analyzed_articles.append({
            'title': article['title'],
            'link': article['link'],
            'published': article['published'],
            'polarity': round(polarity, 3),
            'sentiment': sentiment
        })
        summary[sentiment] += 1
        models_used.add(model)

    total = len(analyzed_articles)
    summary_percent = {
        sentiment: {
            'count': count,
            'percentage': round((count / total) * 100, 2) if total > 0 else 0
        }
        for sentiment, count in summary.items()
    }
    
    if 'FinBERT' in models_used and 'VADER' in models_used:
        model_display = 'FinBERT + VADER (hybrid)'
    elif 'FinBERT' in models_used:
        model_display = 'FinBERT'
    else:
        model_display = 'VADER'

    return {
        'ticker': ticker,
        'articles': analyzed_articles,
        'summary': summary_percent,
        'total_articles': total,
        'model': model_display
    }


# ==================== SOCIAL SCREENING FUNCTIONS ====================

def clean_value(value):
    """Clean a value, handling NaN and Inf."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def get_stock_data_yfinance(tickers):
    """
    Fetch stock data using yfinance for specific tickers.
    Returns dict with stock data for each ticker.
    """
    results = []
    
    if not tickers:
        return results
    
    # Clean and validate tickers
    clean_tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not clean_tickers:
        return results
    
    print(f"Fetching data for tickers: {clean_tickers}")
    
    try:
        # Download data for all tickers at once
        data = yf.download(
            clean_tickers,
            period='5d',
            group_by='ticker',
            progress=False,
            threads=True,
            timeout=30
        )
        
        for ticker in clean_tickers:
            try:
                # Handle single vs multiple ticker data structure
                if len(clean_tickers) == 1:
                    ticker_data = data
                else:
                    if ticker not in data.columns.get_level_values(0):
                        print(f"No data found for {ticker}")
                        continue
                    ticker_data = data[ticker]
                
                # Skip if empty
                if ticker_data.empty:
                    print(f"Empty data for {ticker}")
                    continue
                
                # Get the most recent row with valid data
                ticker_data = ticker_data.dropna(subset=['Close'])
                if ticker_data.empty:
                    continue
                
                latest = ticker_data.iloc[-1]
                
                close_price = clean_value(latest.get('Close'))
                volume = clean_value(latest.get('Volume'))
                
                if close_price is None or close_price == 0:
                    continue
                
                # Calculate change
                if len(ticker_data) >= 2:
                    prev_close = clean_value(ticker_data.iloc[-2].get('Close'))
                    if prev_close and prev_close > 0:
                        change = close_price - prev_close
                        pct_change = ((close_price - prev_close) / prev_close) * 100
                    else:
                        change = 0
                        pct_change = 0
                else:
                    change = 0
                    pct_change = 0
                
                # Get company name
                name = COMPANY_NAMES.get(ticker, ticker)
                
                results.append({
                    'symbol': ticker,
                    'name': name,
                    'price': round(float(close_price), 2),
                    'change': round(float(change), 2),
                    'pct_change': round(float(pct_change), 2),
                    'volume': int(float(volume)) if volume else 0
                })
                
                print(f"Got data for {ticker}: ${close_price:.2f} ({pct_change:+.2f}%)")
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
    except Exception as e:
        print(f"Error downloading stock data: {e}")
    
    return results


def scrape_sentdex_sentiment():
    """Scrape sentiment data from Sentdex with caching."""
    global _sentdex_cache
    
    # Check cache
    if _sentdex_cache['data'] is not None and (time.time() - _sentdex_cache['timestamp']) < _sentdex_cache['ttl']:
        return _sentdex_cache['data']
    
    try:
        url = 'http://www.sentdex.com/financial-analysis/?tf=30d'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find_all('tr')
        
        result = {}
        for row in table:
            cells = row.find_all('td')
            try:
                if len(cells) >= 5:
                    symbol = cells[0].get_text().strip().upper()
                    sentiment = cells[3].get_text().strip()
                    mentions = cells[2].get_text().strip()
                    
                    trend_up = cells[4].find('span', {"class": "glyphicon glyphicon-chevron-up"})
                    trend = 'up' if trend_up else 'down'
                    
                    try:
                        sentiment_val = float(sentiment)
                    except:
                        sentiment_val = 0
                    
                    try:
                        mentions_val = int(mentions.replace(',', ''))
                    except:
                        mentions_val = 0
                    
                    if symbol:
                        result[symbol] = {
                            'sentiment': sentiment_val,
                            'mentions': mentions_val,
                            'trend': trend
                        }
            except:
                continue
        
        # Update cache
        _sentdex_cache['data'] = result
        _sentdex_cache['timestamp'] = time.time()
        
        print(f"Sentdex: Got sentiment for {len(result)} stocks")
        return result
        
    except Exception as e:
        print(f"Error scraping Sentdex: {e}")
        return _sentdex_cache.get('data') or {}


def get_news_sentiment_for_symbol(symbol):
    """Get news sentiment for a single symbol using VADER."""
    try:
        rss_url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return None
        
        sentiments = []
        for entry in feed.entries[:5]:  # Analyze 5 headlines
            title = entry.get('title', '')
            if title:
                scores = vader_analyzer.polarity_scores(title)
                sentiments.append(scores['compound'])
        
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            # Normalize from [-1, 1] to [0, 100]
            normalized = (avg_sentiment + 1) * 50
            return round(normalized, 2)
        
        return None
    except Exception as e:
        print(f"Error getting news sentiment for {symbol}: {e}")
        return None


def get_social_screening_data(tickers):
    """
    Get comprehensive social screening data for specified tickers.
    Uses yfinance for stock data, Sentdex for 30-day sentiment, and news for real-time sentiment.
    """
    global _screening_cache
    
    # Validate tickers
    if not tickers:
        tickers = DEFAULT_TICKERS.copy()
    
    clean_tickers = [t.strip().upper() for t in tickers if t and t.strip()][:10]  # Max 10
    
    if not clean_tickers:
        clean_tickers = DEFAULT_TICKERS.copy()
    
    # Check cache
    cache_key = get_cache_key(clean_tickers)
    if cache_key in _screening_cache:
        cached = _screening_cache[cache_key]
        if time.time() - cached['timestamp'] < _screening_cache_ttl:
            print(f"Returning cached data for {cache_key}")
            return cached['data']
    
    print(f"Fetching fresh social screening data for: {clean_tickers}")
    start_time = time.time()
    
    # Fetch stock data
    stocks = get_stock_data_yfinance(clean_tickers)
    
    if not stocks:
        return {
            'stocks': [],
            'sources': {
                'market_data': False,
                'sentdex': False,
                'news_sentiment': False
            },
            'total_stocks': 0,
            'requested_tickers': clean_tickers,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
    
    # Fetch Sentdex data (cached)
    sentdex_data = scrape_sentdex_sentiment()
    
    # Process each stock and add sentiment data
    merged_stocks = []
    news_success = 0
    sentdex_matches = 0
    
    for stock in stocks:
        symbol = stock['symbol']
        
        merged = {
            'symbol': symbol,
            'name': stock['name'],
            'price': stock['price'],
            'change': stock['change'],
            'pct_change': stock['pct_change'],
            'volume': stock['volume'],
            'sentdex_sentiment': None,
            'sentdex_mentions': None,
            'sentdex_trend': None,
            'news_sentiment': None,
            'composite_score': None,
            'sentiment_status': 'neutral'
        }
        
        # Add Sentdex data if available
        if symbol in sentdex_data:
            merged['sentdex_sentiment'] = sentdex_data[symbol]['sentiment']
            merged['sentdex_mentions'] = sentdex_data[symbol]['mentions']
            merged['sentdex_trend'] = sentdex_data[symbol]['trend']
            sentdex_matches += 1
        
        # Get news sentiment
        news_score = get_news_sentiment_for_symbol(symbol)
        if news_score is not None:
            merged['news_sentiment'] = news_score
            news_success += 1
        
        # Calculate composite score
        scores = []
        weights = []
        
        if merged['sentdex_sentiment'] is not None:
            # Normalize Sentdex from [-6, 6] to [0, 100]
            normalized_sentdex = ((merged['sentdex_sentiment'] + 6) / 12) * 100
            scores.append(normalized_sentdex)
            weights.append(1.5)  # Sentdex weighted higher (30-day data)
        
        if merged['news_sentiment'] is not None:
            scores.append(merged['news_sentiment'])
            weights.append(1.0)
        
        # Fallback: use price momentum if no sentiment data
        if not scores:
            pct = max(-10, min(10, merged['pct_change']))
            price_score = ((pct + 10) / 20) * 100
            scores.append(price_score)
            weights.append(0.5)
        
        # Weighted average
        if scores:
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            merged['composite_score'] = round(weighted_sum / total_weight, 1)
        else:
            merged['composite_score'] = 50.0
        
        # Determine sentiment status
        if merged['composite_score'] >= 60:
            merged['sentiment_status'] = 'positive'
        elif merged['composite_score'] <= 40:
            merged['sentiment_status'] = 'negative'
        else:
            merged['sentiment_status'] = 'neutral'
        
        merged_stocks.append(merged)
    
    # Sort by composite score (highest first)
    merged_stocks.sort(key=lambda x: x['composite_score'], reverse=True)
    
    elapsed = time.time() - start_time
    print(f"Social screening completed in {elapsed:.2f}s - {len(merged_stocks)} stocks, {sentdex_matches} sentdex, {news_success} news")
    
    # Build response
    response_data = {
        'stocks': merged_stocks,
        'sources': {
            'market_data': len(stocks) > 0,
            'sentdex': sentdex_matches > 0,
            'news_sentiment': news_success > 0
        },
        'total_stocks': len(merged_stocks),
        'requested_tickers': clean_tickers,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    
    # Cache the results
    _screening_cache[cache_key] = {
        'data': response_data,
        'timestamp': time.time()
    }
    
    return response_data


# ==================== API ROUTES ====================

@sentiment_bp.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze news sentiment for a ticker/asset."""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()

        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        num_articles = data.get('num_articles', 8)

        try:
            num_articles = int(num_articles)
            if num_articles < 1 or num_articles > 15:
                num_articles = 8
        except:
            num_articles = 8

        results = analyze_ticker_sentiment(ticker, num_articles)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@sentiment_bp.route('/api/social-screening', methods=['POST'])
def social_screening():
    """
    API endpoint for social screening with custom tickers.
    Accepts POST with JSON body: {"tickers": ["AAPL", "MSFT", ...]}
    """
    try:
        data = request.get_json() or {}
        tickers = data.get('tickers', DEFAULT_TICKERS)
        
        # Validate and clean tickers
        if not isinstance(tickers, list):
            tickers = DEFAULT_TICKERS
        
        result = get_social_screening_data(tickers)
        return jsonify(result)
        
    except Exception as e:
        print(f"Social screening error: {e}")
        return jsonify({
            'error': str(e),
            'stocks': [],
            'sources': {
                'market_data': False,
                'sentdex': False,
                'news_sentiment': False
            },
            'total_stocks': 0,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }), 500


@sentiment_bp.route('/api/social-screening/defaults', methods=['GET'])
def get_default_tickers():
    """Return the default tickers list."""
    return jsonify({
        'defaults': DEFAULT_TICKERS,
        'max_tickers': 10
    })


@sentiment_bp.route('/api/sentiment/health', methods=['GET'])
def sentiment_health():
    """Health check with model status."""
    finbert_ok = check_finbert_availability()
    
    return jsonify({
        'status': 'healthy',
        'finbert_available': finbert_ok,
        'fallback': 'VADER',
        'hf_token_set': bool(HF_API_TOKEN)
    })
