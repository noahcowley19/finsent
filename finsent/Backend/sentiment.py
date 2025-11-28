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
import pandas as pd

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

def scrape_most_active_stocks():
    """Scrape most active stocks from Yahoo Finance."""
    try:
        url = 'https://finance.yahoo.com/most-active/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table
        tables = pd.read_html(str(soup))
        if tables:
            df = tables[0]
            # Clean up column names and select relevant columns
            result = []
            for _, row in df.head(25).iterrows():
                try:
                    symbol = str(row.get('Symbol', ''))
                    name = str(row.get('Name', ''))
                    price = row.get('Price (Intraday)', row.get('Price', 0))
                    change = row.get('Change', 0)
                    pct_change = row.get('% Change', '0%')
                    volume = row.get('Volume', 0)
                    
                    # Parse percentage change
                    if isinstance(pct_change, str):
                        pct_change = pct_change.replace('%', '').replace('+', '')
                        try:
                            pct_change = float(pct_change)
                        except:
                            pct_change = 0
                    
                    # Format volume
                    if isinstance(volume, str):
                        volume = volume.replace(',', '')
                        if 'M' in volume:
                            volume = float(volume.replace('M', '')) * 1000000
                        elif 'B' in volume:
                            volume = float(volume.replace('B', '')) * 1000000000
                        elif 'K' in volume:
                            volume = float(volume.replace('K', '')) * 1000
                        else:
                            try:
                                volume = float(volume)
                            except:
                                volume = 0
                    
                    if symbol and symbol != 'nan':
                        result.append({
                            'symbol': symbol,
                            'name': name[:30] if len(name) > 30 else name,
                            'price': float(price) if price else 0,
                            'change': float(change) if change else 0,
                            'pct_change': pct_change,
                            'volume': int(volume) if volume else 0
                        })
                except Exception as e:
                    continue
            
            return result
    except Exception as e:
        print(f"Error scraping Yahoo Finance: {e}")
    
    return []


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
                    symbol = ticker_info[0].get_text().strip()
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


def scrape_twitter_sentiment(url):
    """Scrape Twitter sentiment data from Trade Followers."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        stock_twitter = soup.find_all('tr')
        
        result = {}
        for stock in stock_twitter:
            try:
                score_cells = stock.find_all("td", {"class": "datalistcolumn"})
                if len(score_cells) >= 5:
                    symbol = score_cells[0].get_text().replace('$', '').strip()
                    sector = score_cells[2].get_text().strip()
                    score = score_cells[4].get_text().strip()
                    
                    try:
                        score_val = float(score)
                    except:
                        score_val = 0
                    
                    if symbol and symbol not in result:
                        result[symbol] = {
                            'sector': sector,
                            'twitter_score': score_val
                        }
            except Exception as e:
                continue
        
        return result
    except Exception as e:
        print(f"Error scraping Twitter data: {e}")
    
    return {}


def get_social_screening_data():
    """
    Get combined social screening data from multiple sources.
    Returns merged data from Yahoo Finance, Sentdex, and Twitter sources.
    """
    global _social_cache
    
    # Check cache
    if _social_cache['data'] is not None and (time.time() - _social_cache['timestamp']) < _social_cache['ttl']:
        return _social_cache['data']
    
    # Scrape all sources in parallel
    results = {
        'most_active': [],
        'sentdex': {},
        'twitter_strength': {},
        'twitter_active': {}
    }
    
    def scrape_yahoo():
        return scrape_most_active_stocks()
    
    def scrape_sentdex():
        return scrape_sentdex_sentiment()
    
    def scrape_twitter_strength():
        return scrape_twitter_sentiment("https://www.tradefollowers.com/strength/twitter_strongest.jsp?tf=1m")
    
    def scrape_twitter_active():
        return scrape_twitter_sentiment("https://www.tradefollowers.com/active/twitter_active.jsp?tf=1m")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_yahoo = executor.submit(scrape_yahoo)
        future_sentdex = executor.submit(scrape_sentdex)
        future_twitter_str = executor.submit(scrape_twitter_strength)
        future_twitter_act = executor.submit(scrape_twitter_active)
        
        try:
            results['most_active'] = future_yahoo.result(timeout=15)
        except:
            results['most_active'] = []
        
        try:
            results['sentdex'] = future_sentdex.result(timeout=15)
        except:
            results['sentdex'] = {}
        
        try:
            results['twitter_strength'] = future_twitter_str.result(timeout=15)
        except:
            results['twitter_strength'] = {}
        
        try:
            results['twitter_active'] = future_twitter_act.result(timeout=15)
        except:
            results['twitter_active'] = {}
    
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
            'composite_score': None
        }
        
        # Add Sentdex data
        if symbol in results['sentdex']:
            sentdex = results['sentdex'][symbol]
            merged['sentdex_sentiment'] = sentdex['sentiment']
            merged['sentdex_mentions'] = sentdex['mentions']
            merged['sentdex_trend'] = sentdex['trend']
        
        # Add Twitter strength data
        if symbol in results['twitter_strength']:
            twitter_str = results['twitter_strength'][symbol]
            merged['twitter_sector'] = twitter_str['sector']
            merged['twitter_strength_score'] = twitter_str['twitter_score']
        
        # Add Twitter activity data
        if symbol in results['twitter_active']:
            twitter_act = results['twitter_active'][symbol]
            if not merged['twitter_sector']:
                merged['twitter_sector'] = twitter_act.get('sector')
            merged['twitter_activity_score'] = twitter_act['twitter_score']
        
        # Calculate composite score
        scores = []
        if merged['sentdex_sentiment'] is not None:
            # Normalize sentdex sentiment (typically -6 to 6) to 0-100
            normalized_sentdex = ((merged['sentdex_sentiment'] + 6) / 12) * 100
            scores.append(normalized_sentdex)
        if merged['twitter_strength_score'] is not None:
            scores.append(merged['twitter_strength_score'])
        if merged['twitter_activity_score'] is not None:
            scores.append(merged['twitter_activity_score'])
        
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
        elif stock['pct_change'] > 2:
            sentiment_status = 'positive'
        elif stock['pct_change'] < -2:
            sentiment_status = 'negative'
        
        stock['sentiment_status'] = sentiment_status
    
    # Cache the results
    response_data = {
        'stocks': merged_stocks[:25],  # Top 25 stocks
        'sources': {
            'yahoo_finance': len(results['most_active']) > 0,
            'sentdex': len(results['sentdex']) > 0,
            'twitter_strength': len(results['twitter_strength']) > 0,
            'twitter_activity': len(results['twitter_active']) > 0
        },
        'total_stocks': len(merged_stocks),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    
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
        return jsonify({
            'error': str(e),
            'stocks': [],
            'sources': {
                'yahoo_finance': False,
                'sentdex': False,
                'twitter_strength': False,
                'twitter_activity': False
            }
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
