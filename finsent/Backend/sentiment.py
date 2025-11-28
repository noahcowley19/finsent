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

sentiment_bp = Blueprint('sentiment', __name__)

# Hugging Face Inference API for FinBERT
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')

# Initialize VADER as fallback
vader_analyzer = SentimentIntensityAnalyzer()

# Track which model is being used
_finbert_available = None
_finbert_last_check = 0

# Cache for social screening data - longer TTL to reduce load
_social_cache = {
    'data': None,
    'timestamp': 0,
    'ttl': 1800  # 30 minutes cache to reduce server load
}

# Predefined stock info to avoid slow API calls
STOCK_INFO = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
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
    'UBER': 'Uber Tech.',
    'SPY': 'SPDR S&P 500 ETF'
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

def get_stock_data_fast():
    """Get stock data by scraping Yahoo Finance most active page - no yfinance needed."""
    try:
        print("Fetching most active stocks from Yahoo Finance...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Scrape Yahoo Finance most active page
        url = 'https://finance.yahoo.com/markets/stocks/most-active/'
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        
        # Find the table rows
        rows = soup.find_all('tr', class_='row')
        
        if not rows:
            # Fallback: try finding table body rows
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows[:20]:  # Limit to 20 stocks
            try:
                cells = row.find_all('td')
                if len(cells) < 5:
                    continue
                
                # Extract symbol
                symbol_elem = row.find('span', class_='symbol') or cells[0].find('a')
                if not symbol_elem:
                    continue
                symbol = symbol_elem.get_text(strip=True).upper()
                
                # Extract name
                name_elem = row.find('span', class_='longName') or cells[1]
                name = name_elem.get_text(strip=True)[:30] if name_elem else symbol
                
                # Extract price - look for the price cell
                price = None
                for cell in cells:
                    text = cell.get_text(strip=True)
                    if text and text[0].isdigit() and '.' in text:
                        try:
                            price = float(text.replace(',', ''))
                            break
                        except:
                            continue
                
                if not price:
                    continue
                
                # Extract change percentage
                pct_change = 0
                change_elem = row.find('fin-streamer', {'data-field': 'regularMarketChangePercent'})
                if change_elem:
                    try:
                        pct_text = change_elem.get_text(strip=True).replace('%', '').replace('+', '')
                        pct_change = float(pct_text)
                    except:
                        pass
                
                # Extract volume
                volume = 0
                vol_elem = row.find('fin-streamer', {'data-field': 'regularMarketVolume'})
                if vol_elem:
                    try:
                        vol_text = vol_elem.get_text(strip=True).replace(',', '')
                        if 'M' in vol_text:
                            volume = int(float(vol_text.replace('M', '')) * 1_000_000)
                        elif 'B' in vol_text:
                            volume = int(float(vol_text.replace('B', '')) * 1_000_000_000)
                        elif 'K' in vol_text:
                            volume = int(float(vol_text.replace('K', '')) * 1_000)
                        else:
                            volume = int(float(vol_text))
                    except:
                        volume = 0
                
                change = round(price * pct_change / 100, 2)
                
                results.append({
                    'symbol': symbol,
                    'name': STOCK_INFO.get(symbol, name),
                    'price': round(price, 2),
                    'change': change,
                    'pct_change': round(pct_change, 2),
                    'volume': volume
                })
                
            except Exception as e:
                continue
        
        # If scraping failed, use static fallback with predefined data
        if not results:
            print("Scraping failed, using fallback data...")
            results = get_fallback_stock_data()
        
        print(f"Got {len(results)} stocks")
        return results
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return get_fallback_stock_data()


def get_fallback_stock_data():
    """Return static fallback data when scraping fails."""
    # Static fallback - just show the tickers with placeholder data
    fallback = []
    for symbol, name in list(STOCK_INFO.items())[:15]:
        fallback.append({
            'symbol': symbol,
            'name': name,
            'price': 0,
            'change': 0,
            'pct_change': 0,
            'volume': 0
        })
    return fallback


def scrape_sentdex_sentiment():
    """Scrape sentiment data from Sentdex."""
    try:
        url = 'http://www.sentdex.com/financial-analysis/?tf=30d'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
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
            except:
                continue
        
        return result
    except Exception as e:
        print(f"Error scraping Sentdex: {e}")
        return {}


def get_news_sentiment_for_symbol(symbol):
    """Get news sentiment for a single symbol - fast inline version."""
    try:
        rss_url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return None
        
        sentiments = []
        for entry in feed.entries[:3]:  # Only 3 headlines
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
    except:
        return None


def get_social_screening_data():
    """
    Get combined social screening data - ultra-lightweight version.
    """
    global _social_cache
    
    # Check cache first
    if _social_cache['data'] is not None and (time.time() - _social_cache['timestamp']) < _social_cache['ttl']:
        print("Returning cached social screening data")
        return _social_cache['data']
    
    print("Fetching fresh social screening data...")
    start_time = time.time()
    
    # Fetch stock data (lightweight scraping)
    stocks = get_stock_data_fast()
    
    if not stocks:
        print("No stock data available")
        return {
            'stocks': [],
            'sources': {
                'yahoo_finance': False,
                'sentdex': False,
                'twitter_strength': False,
                'twitter_activity': False
            },
            'total_stocks': 0,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
    
    # Fetch Sentdex data (single request)
    sentdex_data = {}
    try:
        sentdex_data = scrape_sentdex_sentiment()
        print(f"Sentdex returned {len(sentdex_data)} stocks")
    except Exception as e:
        print(f"Sentdex failed: {e}")
    
    # Merge all data and calculate news sentiment inline
    merged_stocks = []
    news_success_count = 0
    
    for stock in stocks[:15]:  # Limit to 15 stocks
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
            'twitter_strength_score': None,
            'twitter_activity_score': None,
            'news_sentiment': None,
            'composite_score': None
        }
        
        # Add Sentdex data
        if symbol in sentdex_data:
            merged['sentdex_sentiment'] = sentdex_data[symbol]['sentiment']
            merged['sentdex_mentions'] = sentdex_data[symbol]['mentions']
            merged['sentdex_trend'] = sentdex_data[symbol]['trend']
        
        # Calculate news sentiment for this stock
        news_score = get_news_sentiment_for_symbol(symbol)
        if news_score is not None:
            merged['news_sentiment'] = news_score
            news_success_count += 1
        
        # Calculate composite score
        scores = []
        
        if merged['sentdex_sentiment'] is not None:
            normalized_sentdex = ((merged['sentdex_sentiment'] + 6) / 12) * 100
            scores.append(normalized_sentdex)
        
        if merged['news_sentiment'] is not None:
            scores.append(merged['news_sentiment'])
        
        # If no sentiment data, use price change as indicator
        if not scores:
            pct = max(-10, min(10, merged['pct_change']))
            price_score = ((pct + 10) / 20) * 100
            scores.append(price_score)
        
        merged['composite_score'] = round(sum(scores) / len(scores), 2)
        
        # Determine sentiment status
        if merged['composite_score'] >= 60:
            merged['sentiment_status'] = 'positive'
        elif merged['composite_score'] <= 40:
            merged['sentiment_status'] = 'negative'
        else:
            merged['sentiment_status'] = 'neutral'
        
        merged_stocks.append(merged)
    
    # Sort by composite score
    merged_stocks.sort(key=lambda x: (x['composite_score'] or 0, x['volume']), reverse=True)
    
    elapsed = time.time() - start_time
    print(f"Social screening completed in {elapsed:.2f}s with {news_success_count} news scores")
    
    # Build response
    response_data = {
        'stocks': merged_stocks,
        'sources': {
            'yahoo_finance': len(stocks) > 0,
            'sentdex': len(sentdex_data) > 0,
            'twitter_strength': False,
            'twitter_activity': news_success_count > 0
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
