# Made by Noah C, enhanced by Claude.ai
# Social Screener with StockTwits, X/Twitter, and News Sentiment
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
import re
import math

sentiment_bp = Blueprint('sentiment', __name__)

# Hugging Face Inference API for FinBERT
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')

# Initialize VADER as fallback
vader_analyzer = SentimentIntensityAnalyzer()

# Track which model is being used
_finbert_available = None
_finbert_last_check = 0

# Cache for social screening data
_screening_cache = {}
_screening_cache_time = {}
SCREENING_CACHE_TTL = 300  # 5 minutes

# StockTwits cache
_stocktwits_cache = {}
_stocktwits_cache_time = {}
STOCKTWITS_CACHE_TTL = 300  # 5 minutes

# Default tickers for social screener
DEFAULT_TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA']

# Company name mappings
COMPANY_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'GOOG': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    'BRK.B': 'Berkshire Hathaway',
    'JPM': 'JPMorgan Chase',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble',
    'MA': 'Mastercard Inc.',
    'UNH': 'UnitedHealth Group',
    'HD': 'Home Depot Inc.',
    'DIS': 'Walt Disney Co.',
    'BAC': 'Bank of America',
    'ADBE': 'Adobe Inc.',
    'CRM': 'Salesforce Inc.',
    'NFLX': 'Netflix Inc.',
    'XOM': 'Exxon Mobil',
    'PFE': 'Pfizer Inc.',
    'TMO': 'Thermo Fisher',
    'CSCO': 'Cisco Systems',
    'ABT': 'Abbott Labs',
    'COST': 'Costco Wholesale',
    'CVX': 'Chevron Corp.',
    'PEP': 'PepsiCo Inc.',
    'AVGO': 'Broadcom Inc.',
    'MRK': 'Merck & Co.',
    'KO': 'Coca-Cola Co.',
    'ABBV': 'AbbVie Inc.',
    'AMD': 'AMD Inc.',
    'INTC': 'Intel Corp.',
    'ORCL': 'Oracle Corp.',
    'QCOM': 'Qualcomm Inc.',
    'NKE': 'Nike Inc.',
    'MCD': "McDonald's Corp.",
    'T': 'AT&T Inc.',
    'VZ': 'Verizon',
    'IBM': 'IBM Corp.',
    'GS': 'Goldman Sachs',
    'CAT': 'Caterpillar Inc.',
    'BA': 'Boeing Co.',
    'GE': 'General Electric',
    'F': 'Ford Motor Co.',
    'GM': 'General Motors',
    'AAL': 'American Airlines',
    'DAL': 'Delta Air Lines',
    'UAL': 'United Airlines',
    'COIN': 'Coinbase',
    'HOOD': 'Robinhood',
    'SQ': 'Block Inc.',
    'PYPL': 'PayPal Holdings',
    'SHOP': 'Shopify Inc.',
    'UBER': 'Uber Technologies',
    'LYFT': 'Lyft Inc.',
    'ABNB': 'Airbnb Inc.',
    'ZM': 'Zoom Video',
    'PLTR': 'Palantir',
    'SNOW': 'Snowflake Inc.',
    'RBLX': 'Roblox Corp.',
    'GME': 'GameStop Corp.',
    'AMC': 'AMC Entertainment',
    'BB': 'BlackBerry Ltd.',
    'NOK': 'Nokia Corp.',
    'SOFI': 'SoFi Technologies',
    'RIVN': 'Rivian Automotive',
    'LCID': 'Lucid Group',
    'NIO': 'NIO Inc.',
    'XPEV': 'XPeng Inc.',
    'LI': 'Li Auto Inc.',
    'BABA': 'Alibaba Group',
    'JD': 'JD.com Inc.',
    'PDD': 'PDD Holdings',
    'BIDU': 'Baidu Inc.',
}


def clean_value(value):
    """Clean NaN, Inf, and None values"""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


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
        _finbert_available = False
        _finbert_last_check = time.time()
        return False


def query_finbert(text):
    """Query FinBERT model via Hugging Face Inference API."""
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    truncated_text = text[:1000]
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": truncated_text},
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
        return None
    
    return None


def query_vader(text):
    """Analyze sentiment using VADER (fallback)."""
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


# =============================================================================
# StockTwits Sentiment - Free API for user-labeled bullish/bearish posts
# =============================================================================

def get_stocktwits_sentiment(ticker):
    """
    Fetch sentiment from StockTwits API.
    Returns bullish/bearish ratio based on user-labeled posts.
    """
    global _stocktwits_cache, _stocktwits_cache_time
    
    cache_key = ticker.upper()
    current_time = time.time()
    
    # Check cache
    if cache_key in _stocktwits_cache:
        if current_time - _stocktwits_cache_time.get(cache_key, 0) < STOCKTWITS_CACHE_TTL:
            return _stocktwits_cache[cache_key]
    
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        messages = data.get('messages', [])
        
        if not messages:
            return None
        
        bullish_count = 0
        bearish_count = 0
        total_with_sentiment = 0
        
        for msg in messages:
            entities = msg.get('entities', {})
            sentiment = entities.get('sentiment', {})
            
            if sentiment:
                basic = sentiment.get('basic')
                if basic == 'Bullish':
                    bullish_count += 1
                    total_with_sentiment += 1
                elif basic == 'Bearish':
                    bearish_count += 1
                    total_with_sentiment += 1
        
        # Calculate sentiment score (0-100 scale)
        if total_with_sentiment > 0:
            bullish_ratio = bullish_count / total_with_sentiment
            sentiment_score = bullish_ratio * 100
        else:
            sentiment_score = 50  # Neutral if no labeled posts
        
        # Get watchlist count if available
        symbol_data = data.get('symbol', {})
        watchlist_count = symbol_data.get('watchlist_count', 0)
        
        result = {
            'score': round(sentiment_score, 1),
            'bullish': bullish_count,
            'bearish': bearish_count,
            'total_posts': len(messages),
            'labeled_posts': total_with_sentiment,
            'watchlist_count': watchlist_count,
            'source': 'StockTwits'
        }
        
        # Cache the result
        _stocktwits_cache[cache_key] = result
        _stocktwits_cache_time[cache_key] = current_time
        
        return result
        
    except Exception as e:
        print(f"StockTwits error for {ticker}: {e}")
        return None


# =============================================================================
# X/Twitter Sentiment - Using cashtag analysis from social aggregators
# =============================================================================

def get_x_sentiment(ticker):
    """
    Get X/Twitter sentiment by analyzing cashtag discussions.
    Uses Google News to find social media coverage and sentiment.
    """
    try:
        # Search for recent X/Twitter discussions about the stock
        query = f"${ticker} twitter OR x.com stock"
        rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            # Fallback: search for general social sentiment
            query = f"{ticker} stock sentiment social media"
            rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return None
        
        # Analyze sentiment of headlines
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total = 0
        
        for entry in feed.entries[:10]:
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            text = f"{title} {summary}"
            
            # Use VADER for quick sentiment
            scores = vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            total += 1
            if compound > 0.05:
                positive_count += 1
            elif compound < -0.05:
                negative_count += 1
            else:
                neutral_count += 1
        
        if total == 0:
            return None
        
        # Calculate score (0-100)
        sentiment_score = ((positive_count - negative_count) / total + 1) * 50
        sentiment_score = max(0, min(100, sentiment_score))
        
        return {
            'score': round(sentiment_score, 1),
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'total_analyzed': total,
            'source': 'X/Social'
        }
        
    except Exception as e:
        print(f"X sentiment error for {ticker}: {e}")
        return None


# =============================================================================
# News Sentiment
# =============================================================================

def fetch_article_content(url):
    """Fetch and extract article text content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=3, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        content = response.content[:50000]
        soup = BeautifulSoup(content, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        paragraphs = soup.find_all('p')[:5]
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if len(text) > 50:
            return text[:1000]
        return ""
    except:
        return ""


def get_news_sentiment(ticker, num_articles=5):
    """Get news sentiment for a ticker using Google News RSS."""
    try:
        queries = [
            f"{ticker} stock news",
            f"{ticker} market"
        ]
        
        all_articles = []
        
        for query in queries:
            rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            for item in feed.entries[:num_articles]:
                all_articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', '')[:300]
                })
        
        if not all_articles:
            return None
        
        # Remove duplicates
        seen = set()
        unique_articles = []
        for article in all_articles:
            if article['title'] not in seen:
                seen.add(article['title'])
                unique_articles.append(article)
        
        # Analyze sentiment
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in unique_articles[:10]:
            text = f"{article['title']} {article['summary']}"
            scores = vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound > 0.05:
                positive_count += 1
            elif compound < -0.05:
                negative_count += 1
            else:
                neutral_count += 1
        
        total = positive_count + negative_count + neutral_count
        if total == 0:
            return None
        
        # Calculate score (0-100)
        sentiment_score = ((positive_count - negative_count) / total + 1) * 50
        sentiment_score = max(0, min(100, sentiment_score))
        
        return {
            'score': round(sentiment_score, 1),
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'total_articles': total,
            'source': 'News'
        }
        
    except Exception as e:
        print(f"News sentiment error for {ticker}: {e}")
        return None


# =============================================================================
# Stock Data via yfinance
# =============================================================================

def get_stock_data(tickers):
    """Fetch stock data for multiple tickers using yfinance."""
    try:
        ticker_str = ' '.join(tickers)
        data = yf.download(ticker_str, period='5d', interval='1d', progress=False, threads=True)
        
        results = {}
        
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    close_data = data['Close']
                    volume_data = data['Volume']
                else:
                    close_data = data['Close'][ticker] if ticker in data['Close'].columns else None
                    volume_data = data['Volume'][ticker] if ticker in data['Volume'].columns else None
                
                if close_data is None or close_data.empty:
                    results[ticker] = None
                    continue
                
                # Get latest values
                current_price = clean_value(float(close_data.iloc[-1]))
                
                # Calculate change
                if len(close_data) >= 2:
                    prev_price = clean_value(float(close_data.iloc[-2]))
                    if prev_price and current_price:
                        change = current_price - prev_price
                        pct_change = (change / prev_price) * 100
                    else:
                        change = 0
                        pct_change = 0
                else:
                    change = 0
                    pct_change = 0
                
                # Get volume
                volume = clean_value(float(volume_data.iloc[-1])) if volume_data is not None and not volume_data.empty else 0
                
                results[ticker] = {
                    'price': current_price,
                    'change': clean_value(change),
                    'pct_change': clean_value(pct_change),
                    'volume': volume
                }
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                results[ticker] = None
        
        return results
        
    except Exception as e:
        print(f"yfinance error: {e}")
        return {ticker: None for ticker in tickers}


def format_volume(volume):
    """Format volume for display."""
    if volume is None:
        return 'N/A'
    if volume >= 1e9:
        return f"{volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"{volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"{volume/1e3:.1f}K"
    return str(int(volume))


# =============================================================================
# Social Screening Endpoint
# =============================================================================

@sentiment_bp.route('/api/social-screening', methods=['POST'])
def social_screening():
    """
    Main social screening endpoint.
    Returns stock data with StockTwits, X, and News sentiment.
    """
    global _screening_cache, _screening_cache_time
    
    try:
        data = request.get_json()
        tickers = data.get('tickers', DEFAULT_TICKERS)
        
        # Validate tickers
        if not tickers:
            tickers = DEFAULT_TICKERS
        
        # Limit to 10 tickers
        tickers = [t.upper().strip() for t in tickers[:10] if t.strip()]
        
        if not tickers:
            tickers = DEFAULT_TICKERS
        
        # Create cache key
        cache_key = ':'.join(sorted(tickers))
        current_time = time.time()
        
        # Check cache
        if cache_key in _screening_cache:
            if current_time - _screening_cache_time.get(cache_key, 0) < SCREENING_CACHE_TTL:
                return jsonify(_screening_cache[cache_key])
        
        # Fetch stock data
        stock_data = get_stock_data(tickers)
        
        results = []
        sources_status = {
            'market_data': False,
            'stocktwits': False,
            'x_sentiment': False,
            'news': False
        }
        
        for ticker in tickers:
            stock = stock_data.get(ticker)
            
            # Get company name
            company_name = COMPANY_NAMES.get(ticker, ticker)
            
            # Initialize result
            result = {
                'ticker': ticker,
                'company': company_name,
                'price': None,
                'price_display': '--',
                'change': None,
                'pct_change': None,
                'change_display': '--',
                'volume': None,
                'volume_display': '--',
                'stocktwits': None,
                'stocktwits_display': '--',
                'stocktwits_detail': None,
                'x_sentiment': None,
                'x_display': '--',
                'x_detail': None,
                'news_sentiment': None,
                'news_display': '--',
                'news_detail': None,
                'composite': None,
                'composite_display': '--',
                'composite_status': 'neutral'
            }
            
            # Add stock data
            if stock:
                sources_status['market_data'] = True
                result['price'] = stock['price']
                result['price_display'] = f"${stock['price']:.2f}" if stock['price'] else '--'
                result['change'] = stock['change']
                result['pct_change'] = stock['pct_change']
                
                if stock['pct_change'] is not None:
                    sign = '+' if stock['pct_change'] >= 0 else ''
                    result['change_display'] = f"{sign}{stock['pct_change']:.2f}%"
                
                result['volume'] = stock['volume']
                result['volume_display'] = format_volume(stock['volume'])
            
            # Get StockTwits sentiment
            stocktwits = get_stocktwits_sentiment(ticker)
            if stocktwits:
                sources_status['stocktwits'] = True
                result['stocktwits'] = stocktwits['score']
                result['stocktwits_display'] = f"{stocktwits['score']:.0f}"
                result['stocktwits_detail'] = f"🐂 {stocktwits['bullish']} / 🐻 {stocktwits['bearish']}"
            
            # Get X/Twitter sentiment
            x_sent = get_x_sentiment(ticker)
            if x_sent:
                sources_status['x_sentiment'] = True
                result['x_sentiment'] = x_sent['score']
                result['x_display'] = f"{x_sent['score']:.0f}"
                if 'positive' in x_sent:
                    result['x_detail'] = f"+{x_sent['positive']} / -{x_sent['negative']}"
            
            # Get News sentiment
            news = get_news_sentiment(ticker, num_articles=5)
            if news:
                sources_status['news'] = True
                result['news_sentiment'] = news['score']
                result['news_display'] = f"{news['score']:.0f}"
                result['news_detail'] = f"+{news['positive']} / -{news['negative']}"
            
            # Calculate composite score
            scores = []
            weights = []
            
            if result['stocktwits'] is not None:
                scores.append(result['stocktwits'])
                weights.append(1.5)  # StockTwits weighted higher (user-labeled)
            
            if result['x_sentiment'] is not None:
                scores.append(result['x_sentiment'])
                weights.append(1.0)
            
            if result['news_sentiment'] is not None:
                scores.append(result['news_sentiment'])
                weights.append(1.0)
            
            # Fallback: use price momentum if no sentiment data
            if not scores and result['pct_change'] is not None:
                # Convert pct_change to 0-100 scale
                momentum_score = ((result['pct_change'] + 10) / 20) * 100
                momentum_score = max(0, min(100, momentum_score))
                scores.append(momentum_score)
                weights.append(0.5)
            
            if scores:
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                total_weight = sum(weights)
                composite = weighted_sum / total_weight
                
                result['composite'] = round(composite, 1)
                result['composite_display'] = f"{composite:.0f}"
                
                if composite >= 60:
                    result['composite_status'] = 'positive'
                elif composite <= 40:
                    result['composite_status'] = 'negative'
                else:
                    result['composite_status'] = 'neutral'
            
            results.append(result)
        
        response = {
            'tickers': tickers,
            'results': results,
            'sources': sources_status,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cache_ttl': SCREENING_CACHE_TTL
        }
        
        # Cache the response
        _screening_cache[cache_key] = response
        _screening_cache_time[cache_key] = current_time
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Social screening error: {e}")
        return jsonify({
            'error': str(e),
            'tickers': [],
            'results': []
        }), 500


@sentiment_bp.route('/api/social-screening/defaults', methods=['GET'])
def get_defaults():
    """Return default tickers for social screener."""
    return jsonify({
        'tickers': DEFAULT_TICKERS,
        'max_tickers': 10
    })


# =============================================================================
# Original News Analysis Endpoint (for deep analysis)
# =============================================================================

def fetch_news_batch(queries, num_articles_per_query):
    """Fetch news articles from Google News RSS."""
    all_articles = []
    
    for query in queries:
        try:
            rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
            feed = feedparser.parse(rss_url)
            news_items = feed.entries[:num_articles_per_query]
            
            for item in news_items:
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


@sentiment_bp.route('/api/analyze', methods=['POST'])
def analyze():
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


@sentiment_bp.route('/api/sentiment/health', methods=['GET'])
def sentiment_health():
    """Health check with model status."""
    finbert_ok = check_finbert_availability()
    
    return jsonify({
        'status': 'healthy',
        'finbert_available': finbert_ok,
        'fallback': 'VADER',
        'hf_token_set': bool(HF_API_TOKEN),
        'features': ['stocktwits', 'x_sentiment', 'news_sentiment', 'composite_score']
    })
