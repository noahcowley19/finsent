# Made by Noah C, debugged by Claude.ai
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
_social_cache = {
    'data': None,
    'timestamp': 0,
    'ttl': 300  # 5 minutes cache
}


def check_finbert_availability():
    """Check if FinBERT API is responding."""
    global _finbert_available, _finbert_last_check
    
    # Cache the check for 5 minutes
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
                print("FinBERT: Available and responding")
                return True
        elif response.status_code == 503:
            # Model loading - wait and retry once
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
                print("FinBERT: Available after warm-up")
                return True
                
        print(f"FinBERT: Unavailable (status {response.status_code})")
        _finbert_available = False
        _finbert_last_check = time.time()
        return False
        
    except Exception as e:
        print(f"FinBERT: Error checking availability - {e}")
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
            return None  # Signal to use fallback
            
        result = response.json()
        
        if result and isinstance(result, list) and len(result) > 0:
            predictions = result[0] if isinstance(result[0], list) else result
            
            # Find the highest scoring label
            best = max(predictions, key=lambda x: x['score'])
            label = best['label'].lower()
            score = best['score']
            
            # Convert to polarity score (-1 to 1)
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
    """
    Analyze sentiment - tries FinBERT first, falls back to VADER.
    Returns (polarity, sentiment, model_used)
    """
    if use_finbert:
        result = query_finbert(text)
        if result is not None:
            return result[0], result[1], 'FinBERT'
    
    # Fallback to VADER
    polarity, sentiment = query_vader(text)
    return polarity, sentiment, 'VADER'


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
    
    # Check if FinBERT is available
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
    
    # Determine primary model used
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

def get_most_active_stocks_yfinance():
    """Get most active stocks using yfinance."""
    try:
        # List of commonly active tickers to check
        active_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'NFLX',
            'BA', 'DIS', 'V', 'JPM', 'WMT', 'PFE', 'KO', 'PEP', 'MCD', 'NKE',
            'COIN', 'PLTR', 'SOFI', 'HOOD', 'RIVN', 'LCID', 'F', 'GM', 'AAL', 'UAL',
            'BAC', 'C', 'WFC', 'GS', 'MS', 'PYPL', 'SQ', 'SHOP', 'ROKU', 'SNAP',
            'UBER', 'LYFT', 'ABNB', 'ZM', 'DOCU', 'CRWD', 'NET', 'DDOG', 'SNOW', 'SPY'
        ]
        
        results = []
        
        # Fetch data for all tickers using yfinance
        tickers_str = ' '.join(active_tickers)
        data = yf.download(tickers_str, period='2d', group_by='ticker', progress=False, threads=True)
        
        for ticker in active_tickers:
            try:
                # Handle both single and multi-ticker data structures
                if len(active_tickers) == 1:
                    ticker_data = data
                else:
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    ticker_data = data[ticker]
                
                if ticker_data.empty or len(ticker_data) < 1:
                    continue
                
                # Get the latest row
                latest = ticker_data.iloc[-1]
                
                close_price = float(latest.get('Close', 0))
                open_price = float(latest.get('Open', close_price))
                volume = int(latest.get('Volume', 0))
                
                # Get previous close for accurate change calculation
                if len(ticker_data) >= 2:
                    prev_close = float(ticker_data.iloc[-2].get('Close', open_price))
                else:
                    prev_close = open_price
                
                if close_price == 0 or volume == 0:
                    continue
                
                # Calculate change
                change = close_price - prev_close
                pct_change = ((close_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                
                # Get company name using info (cached by yfinance)
                try:
                    stock = yf.Ticker(ticker)
                    name = stock.info.get('shortName', ticker)
                    if name and len(name) > 30:
                        name = name[:27] + '...'
                except:
                    name = ticker
                
                results.append({
                    'symbol': ticker,
                    'name': name,
                    'price': round(close_price, 2),
                    'change': round(change, 2),
                    'pct_change': round(pct_change, 2),
                    'volume': volume
                })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        # Sort by volume (most active first)
        results.sort(key=lambda x: x['volume'], reverse=True)
        
        return results[:25]
        
    except Exception as e:
        print(f"Error fetching yfinance data: {e}")
        return []


def scrape_sentdex_sentiment():
    """Scrape sentiment data from Sentdex."""
    try:
        url = 'http://www.sentdex.com/financial-analysis/?tf=30d'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find_all('tr')
        
        result = {}
        for ticker in table:
            ticker_info = ticker.find_all('td')
            try:
                if len(ticker_info) >= 5:
                    symbol = ticker_info[0].get_text().strip().upper()
                    sentiment = ticker_info[3].get_text().strip()
                    mentions = ticker_info[2].get_text().strip()
                    
                    # Determine trend direction
                    trend_up = ticker_info[4].find('span', {"class": "glyphicon glyphicon-chevron-up"})
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
            except Exception as e:
                continue
        
        return result
    except Exception as e:
        print(f"Error scraping Sentdex: {e}")
    
    return {}


def scrape_stocktwits_trending():
    """Scrape trending stocks from StockTwits as alternative to TradeFollowers."""
    try:
        url = 'https://api.stocktwits.com/api/2/trending/symbols.json'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            result = {}
            
            symbols = data.get('symbols', [])
            for i, symbol_data in enumerate(symbols[:30]):
                symbol = symbol_data.get('symbol', '').upper()
                if symbol:
                    # Use ranking as activity score
                    result[symbol] = {
                        'sector': symbol_data.get('sector', 'N/A'),
                        'twitter_score': max(0, 100 - (i * 3)),  # Higher rank = higher score
                        'watchlist_count': symbol_data.get('watchlist_count', 0)
                    }
            
            return result
    except Exception as e:
        print(f"Error scraping StockTwits: {e}")
    
    return {}


def generate_sentiment_from_news(symbol):
    """Generate sentiment score from recent news for a symbol."""
    try:
        # Quick news sentiment check
        rss_url = f"https://news.google.com/rss/search?q={symbol}+stock"
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return None
        
        # Analyze up to 5 recent headlines
        sentiments = []
        for entry in feed.entries[:5]:
            title = entry.get('title', '')
            if title:
                scores = vader_analyzer.polarity_scores(title)
                sentiments.append(scores['compound'])
        
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            # Convert from -1 to 1 scale to 0-100 scale
            normalized = (avg_sentiment + 1) * 50
            return round(normalized, 2)
        
    except:
        pass
    
    return None


def get_social_screening_data():
    """
    Get combined social screening data from multiple sources.
    Returns merged data from yfinance, Sentdex, and StockTwits.
    """
    global _social_cache
    
    # Check cache
    if _social_cache['data'] is not None and (time.time() - _social_cache['timestamp']) < _social_cache['ttl']:
        return _social_cache['data']
    
    results = {
        'most_active': [],
        'sentdex': {},
        'stocktwits': {},
    }
    
    # Scrape all sources in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_yfinance = executor.submit(get_most_active_stocks_yfinance)
        future_sentdex = executor.submit(scrape_sentdex_sentiment)
        future_stocktwits = executor.submit(scrape_stocktwits_trending)
        
        try:
            results['most_active'] = future_yfinance.result(timeout=45)
        except Exception as e:
            print(f"yfinance timeout/error: {e}")
            results['most_active'] = []
        
        try:
            results['sentdex'] = future_sentdex.result(timeout=20)
        except Exception as e:
            print(f"Sentdex timeout/error: {e}")
            results['sentdex'] = {}
        
        try:
            results['stocktwits'] = future_stocktwits.result(timeout=15)
        except Exception as e:
            print(f"StockTwits timeout/error: {e}")
            results['stocktwits'] = {}
    
    # Merge data
    merged_stocks = []
    for stock in results['most_active']:
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
            'twitter_sector': None,
            'twitter_strength_score': None,
            'twitter_activity_score': None,
            'news_sentiment': None,
            'composite_score': None
        }
        
        # Add Sentdex data
        if symbol in results['sentdex']:
            sentdex = results['sentdex'][symbol]
            merged['sentdex_sentiment'] = sentdex['sentiment']
            merged['sentdex_mentions'] = sentdex['mentions']
            merged['sentdex_trend'] = sentdex['trend']
        
        # Add StockTwits data (as Twitter alternative)
        if symbol in results['stocktwits']:
            stocktwits = results['stocktwits'][symbol]
            merged['twitter_sector'] = stocktwits.get('sector')
            merged['twitter_strength_score'] = stocktwits.get('twitter_score')
            merged['twitter_activity_score'] = stocktwits.get('twitter_score')
        
        # Generate news-based sentiment if no other sentiment data
        if merged['sentdex_sentiment'] is None and merged['twitter_strength_score'] is None:
            news_sent = generate_sentiment_from_news(symbol)
            if news_sent is not None:
                merged['news_sentiment'] = news_sent
        
        # Calculate composite score
        scores = []
        
        if merged['sentdex_sentiment'] is not None:
            # Normalize sentdex sentiment (typically -6 to 6) to 0-100
            normalized_sentdex = ((merged['sentdex_sentiment'] + 6) / 12) * 100
            scores.append(normalized_sentdex)
        
        if merged['twitter_strength_score'] is not None:
            scores.append(merged['twitter_strength_score'])
        
        if merged['news_sentiment'] is not None:
            scores.append(merged['news_sentiment'])
        
        # If no sentiment data, use price change as a rough indicator
        if not scores:
            # Convert price change to 0-100 scale (capped at ±10%)
            pct = max(-10, min(10, merged['pct_change']))
            price_score = ((pct + 10) / 20) * 100
            scores.append(price_score)
        
        if scores:
            merged['composite_score'] = round(sum(scores) / len(scores), 2)
        
        merged_stocks.append(merged)
    
    # Sort by composite score (highest first), then by volume
    merged_stocks.sort(key=lambda x: (x['composite_score'] or 0, x['volume']), reverse=True)
    
    # Determine overall sentiment for each stock
    for stock in merged_stocks:
        sentiment_status = 'neutral'
        if stock['composite_score'] is not None:
            if stock['composite_score'] >= 60:
                sentiment_status = 'positive'
            elif stock['composite_score'] <= 40:
                sentiment_status = 'negative'
        
        stock['sentiment_status'] = sentiment_status
    
    # Build response
    response_data = {
        'stocks': merged_stocks[:25],
        'sources': {
            'yahoo_finance': len(results['most_active']) > 0,
            'sentdex': len(results['sentdex']) > 0,
            'twitter_strength': len(results['stocktwits']) > 0,
            'twitter_activity': len(results['stocktwits']) > 0
        },
        'total_stocks': len(merged_stocks),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    
    # Cache the results
    _social_cache['data'] = response_data
    _social_cache['timestamp'] = time.time()
    
    return response_data


# ==================== API ROUTES ====================

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


@sentiment_bp.route('/api/social-screening', methods=['GET'])
def social_screening():
    """API endpoint for social screening data."""
    try:
        data = get_social_screening_data()
        return jsonify(data)
    except Exception as e:
        print(f"Social screening error: {e}")
        return jsonify({
            'error': str(e),
            'stocks': [],
            'sources': {
                'yahoo_finance': False,
                'sentdex': False,
                'twitter_strength': False,
                'twitter_activity': False
            },
            'total_stocks': 0,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }), 500


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
