from flask import Blueprint, request, jsonify
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import math
import time

dashboard_bp = Blueprint('dashboard', __name__)

# Cache settings
_dashboard_cache = {}
_dashboard_cache_time = {}
DASHBOARD_CACHE_TTL = 120  # 2 minutes for dashboard data


def clean_value(value):
    """Clean NaN, Inf, and None values"""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def safe_round(value, decimals=2):
    """Safely round a value"""
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return round(cleaned, decimals)
    except:
        return None


def format_large_number(value):
    """Format large numbers with suffixes"""
    if value is None:
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"${value / 1e3:.1f}K"
    else:
        return f"${value:,.0f}"


def format_number(value, decimals=2):
    """Format numbers without currency symbol"""
    if value is None:
        return 'N/A'
    return f"{value:,.{decimals}f}"


def get_quote_data(symbol):
    """Get quote data for a symbol using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        price = clean_value(info.get('regularMarketPrice')) or clean_value(info.get('currentPrice'))
        prev_close = clean_value(info.get('regularMarketPreviousClose') or info.get('previousClose'))
        
        change = None
        change_pct = None
        if price and prev_close:
            change = price - prev_close
            change_pct = (change / prev_close) * 100
        
        return {
            'symbol': symbol,
            'name': info.get('shortName', info.get('longName', symbol)),
            'price': safe_round(price, 2),
            'change': safe_round(change, 2),
            'change_pct': safe_round(change_pct, 2),
            'volume': clean_value(info.get('regularMarketVolume')),
            'market_cap': clean_value(info.get('marketCap')),
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def get_market_indices():
    """Get major market indices data"""
    indices = {
        '^GSPC': {'name': 'S&P 500', 'short': 'S&P'},
        '^DJI': {'name': 'Dow Jones', 'short': 'DOW'},
        '^IXIC': {'name': 'Nasdaq', 'short': 'NASDAQ'},
        '^RUT': {'name': 'Russell 2000', 'short': 'RUT'},
        '^VIX': {'name': 'VIX', 'short': 'VIX'}
    }
    
    results = []
    symbols = list(indices.keys())
    
    try:
        tickers = yf.Tickers(' '.join(symbols))
        
        for symbol in symbols:
            try:
                info = tickers.tickers[symbol].info
                
                price = clean_value(info.get('regularMarketPrice'))
                prev_close = clean_value(info.get('regularMarketPreviousClose'))
                
                change = None
                change_pct = None
                if price and prev_close:
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100
                
                results.append({
                    'symbol': symbol,
                    'name': indices[symbol]['name'],
                    'short_name': indices[symbol]['short'],
                    'price': safe_round(price, 2),
                    'price_display': f"{price:,.2f}" if price else 'N/A',
                    'change': safe_round(change, 2),
                    'change_pct': safe_round(change_pct, 2),
                    'change_display': f"{change:+,.2f}" if change else 'N/A',
                    'change_pct_display': f"{change_pct:+.2f}%" if change_pct else 'N/A',
                    'status': 'positive' if change and change >= 0 else 'negative' if change else 'neutral'
                })
            except Exception as e:
                print(f"Error getting {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'name': indices[symbol]['name'],
                    'short_name': indices[symbol]['short'],
                    'price': None,
                    'price_display': 'N/A',
                    'change': None,
                    'change_pct': None,
                    'change_display': 'N/A',
                    'change_pct_display': 'N/A',
                    'status': 'neutral'
                })
    except Exception as e:
        print(f"Error fetching indices: {e}")
    
    return results


def get_futures_data():
    """Get futures data for major contracts"""
    futures = {
        'ES=F': {'name': 'S&P 500 Futures', 'short': 'ES'},
        'NQ=F': {'name': 'Nasdaq Futures', 'short': 'NQ'},
        'YM=F': {'name': 'Dow Futures', 'short': 'YM'},
        'RTY=F': {'name': 'Russell 2000 Futures', 'short': 'RTY'},
        'GC=F': {'name': 'Gold', 'short': 'Gold'},
        'SI=F': {'name': 'Silver', 'short': 'Silver'},
        'CL=F': {'name': 'Crude Oil', 'short': 'WTI'},
        'NG=F': {'name': 'Natural Gas', 'short': 'NatGas'},
    }
    
    results = []
    symbols = list(futures.keys())
    
    try:
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                
                if not hist.empty and len(hist) >= 1:
                    price = clean_value(float(hist['Close'].iloc[-1]))
                    prev_close = clean_value(float(hist['Close'].iloc[-2])) if len(hist) >= 2 else price
                    
                    change = None
                    change_pct = None
                    if price and prev_close:
                        change = price - prev_close
                        change_pct = (change / prev_close) * 100
                    
                    # Format based on price magnitude
                    if price and price < 10:
                        price_display = f"{price:.4f}"
                    elif price and price < 100:
                        price_display = f"{price:.2f}"
                    else:
                        price_display = f"{price:,.2f}" if price else 'N/A'
                    
                    results.append({
                        'symbol': symbol,
                        'name': futures[symbol]['name'],
                        'short_name': futures[symbol]['short'],
                        'price': safe_round(price, 2),
                        'price_display': price_display,
                        'change': safe_round(change, 2),
                        'change_pct': safe_round(change_pct, 2),
                        'change_display': f"{change:+,.2f}" if change else 'N/A',
                        'change_pct_display': f"{change_pct:+.2f}%" if change_pct else 'N/A',
                        'status': 'positive' if change and change >= 0 else 'negative' if change else 'neutral'
                    })
            except Exception as e:
                print(f"Error getting futures {symbol}: {e}")
    except Exception as e:
        print(f"Error fetching futures: {e}")
    
    return results


def get_forex_data():
    """Get forex data for major currency pairs"""
    forex = {
        'EURUSD=X': {'name': 'EUR/USD', 'short': 'EUR/USD'},
        'GBPUSD=X': {'name': 'GBP/USD', 'short': 'GBP/USD'},
        'USDJPY=X': {'name': 'USD/JPY', 'short': 'USD/JPY'},
        'USDCAD=X': {'name': 'USD/CAD', 'short': 'USD/CAD'},
        'AUDUSD=X': {'name': 'AUD/USD', 'short': 'AUD/USD'},
        'USDCHF=X': {'name': 'USD/CHF', 'short': 'USD/CHF'},
    }
    
    results = []
    
    try:
        for symbol, info in forex.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                
                if not hist.empty and len(hist) >= 1:
                    price = clean_value(float(hist['Close'].iloc[-1]))
                    prev_close = clean_value(float(hist['Close'].iloc[-2])) if len(hist) >= 2 else price
                    
                    change = None
                    change_pct = None
                    if price and prev_close:
                        change = price - prev_close
                        change_pct = (change / prev_close) * 100
                    
                    results.append({
                        'symbol': symbol,
                        'name': info['name'],
                        'short_name': info['short'],
                        'price': safe_round(price, 4),
                        'price_display': f"{price:.4f}" if price else 'N/A',
                        'change': safe_round(change, 4),
                        'change_pct': safe_round(change_pct, 2),
                        'change_display': f"{change:+.4f}" if change else 'N/A',
                        'change_pct_display': f"{change_pct:+.2f}%" if change_pct else 'N/A',
                        'status': 'positive' if change and change >= 0 else 'negative' if change else 'neutral'
                    })
            except Exception as e:
                print(f"Error getting forex {symbol}: {e}")
    except Exception as e:
        print(f"Error fetching forex: {e}")
    
    return results


def get_crypto_data():
    """Get crypto data for major cryptocurrencies"""
    crypto = {
        'BTC-USD': {'name': 'Bitcoin', 'short': 'BTC'},
        'ETH-USD': {'name': 'Ethereum', 'short': 'ETH'},
        'SOL-USD': {'name': 'Solana', 'short': 'SOL'},
        'XRP-USD': {'name': 'XRP', 'short': 'XRP'},
    }
    
    results = []
    
    try:
        for symbol, info in crypto.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                
                if not hist.empty and len(hist) >= 1:
                    price = clean_value(float(hist['Close'].iloc[-1]))
                    prev_close = clean_value(float(hist['Close'].iloc[-2])) if len(hist) >= 2 else price
                    
                    change = None
                    change_pct = None
                    if price and prev_close:
                        change = price - prev_close
                        change_pct = (change / prev_close) * 100
                    
                    # Format price based on magnitude
                    if price and price > 1000:
                        price_display = f"${price:,.2f}"
                    elif price and price > 1:
                        price_display = f"${price:.2f}"
                    else:
                        price_display = f"${price:.4f}" if price else 'N/A'
                    
                    results.append({
                        'symbol': symbol,
                        'name': info['name'],
                        'short_name': info['short'],
                        'price': safe_round(price, 2),
                        'price_display': price_display,
                        'change': safe_round(change, 2),
                        'change_pct': safe_round(change_pct, 2),
                        'change_display': f"${change:+,.2f}" if change else 'N/A',
                        'change_pct_display': f"{change_pct:+.2f}%" if change_pct else 'N/A',
                        'status': 'positive' if change and change >= 0 else 'negative' if change else 'neutral'
                    })
            except Exception as e:
                print(f"Error getting crypto {symbol}: {e}")
    except Exception as e:
        print(f"Error fetching crypto: {e}")
    
    return results


def get_market_movers():
    """Get top gainers, losers, and most active stocks"""
    movers = {
        'gainers': [],
        'losers': [],
        'most_active': []
    }
    
    # Define some well-known large cap stocks to check
    watchlist = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC',
        'ADBE', 'CRM', 'NFLX', 'XOM', 'PFE', 'CSCO', 'ABT', 'COST', 'CVX',
        'PEP', 'AVGO', 'MRK', 'KO', 'ABBV', 'AMD', 'INTC', 'ORCL', 'QCOM',
        'NKE', 'MCD', 'T', 'VZ', 'IBM', 'GS', 'CAT', 'BA', 'GE', 'F', 'GM'
    ]
    
    stock_data = []
    
    try:
        # Get data for watchlist stocks
        for symbol in watchlist[:30]:  # Limit to avoid timeout
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                price = clean_value(info.get('regularMarketPrice')) or clean_value(info.get('currentPrice'))
                prev_close = clean_value(info.get('regularMarketPreviousClose') or info.get('previousClose'))
                volume = clean_value(info.get('regularMarketVolume') or info.get('volume'))
                
                if price and prev_close:
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    stock_data.append({
                        'symbol': symbol,
                        'name': info.get('shortName', symbol)[:20],
                        'price': safe_round(price, 2),
                        'price_display': f"${price:.2f}",
                        'change': safe_round(change, 2),
                        'change_pct': safe_round(change_pct, 2),
                        'change_display': f"{change:+.2f}",
                        'change_pct_display': f"{change_pct:+.2f}%",
                        'volume': volume,
                        'volume_display': format_volume(volume),
                        'status': 'positive' if change >= 0 else 'negative'
                    })
            except Exception as e:
                continue
        
        # Sort for gainers (top performers)
        sorted_by_change = sorted(stock_data, key=lambda x: x['change_pct'] if x['change_pct'] else 0, reverse=True)
        movers['gainers'] = sorted_by_change[:5]
        
        # Sort for losers (worst performers)
        movers['losers'] = sorted(stock_data, key=lambda x: x['change_pct'] if x['change_pct'] else 0)[:5]
        
        # Sort for most active
        sorted_by_volume = sorted(stock_data, key=lambda x: x['volume'] if x['volume'] else 0, reverse=True)
        movers['most_active'] = sorted_by_volume[:5]
        
    except Exception as e:
        print(f"Error getting market movers: {e}")
    
    return movers


def format_volume(volume):
    """Format volume for display"""
    if volume is None:
        return 'N/A'
    if volume >= 1e9:
        return f"{volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"{volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"{volume/1e3:.1f}K"
    return str(int(volume))


def get_market_news(num_articles=8):
    """Get market news from Google News RSS"""
    try:
        queries = [
            "stock market today",
            "wall street news",
            "financial markets"
        ]
        
        all_articles = []
        seen_titles = set()
        
        for query in queries:
            try:
                rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(rss_url)
                
                for item in feed.entries[:5]:
                    title = item.get('title', '')
                    if title and title.lower() not in seen_titles:
                        seen_titles.add(title.lower())
                        
                        # Parse published date
                        published = item.get('published', '')
                        try:
                            pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                            pub_relative = get_relative_time(pub_date)
                        except:
                            pub_relative = 'Recently'
                        
                        # Extract source from title
                        source = 'News'
                        if ' - ' in title:
                            parts = title.rsplit(' - ', 1)
                            if len(parts) == 2:
                                title = parts[0]
                                source = parts[1]
                        
                        all_articles.append({
                            'title': title[:100],
                            'source': source[:25],
                            'link': item.get('link', '#'),
                            'published': pub_relative
                        })
            except:
                continue
        
        return all_articles[:num_articles]
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def get_relative_time(dt):
    """Convert datetime to relative time string"""
    now = datetime.now()
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes}m ago"
    else:
        return "Just now"


def get_upcoming_earnings():
    """Get upcoming earnings for major companies"""
    # List of major companies to check
    major_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
                     'JPM', 'BAC', 'WMT', 'HD', 'DIS', 'NFLX', 'CRM', 'ADBE']
    
    earnings = []
    
    try:
        for ticker_symbol in major_tickers[:10]:  # Limit to avoid timeout
            try:
                ticker = yf.Ticker(ticker_symbol)
                calendar = ticker.calendar
                
                if calendar is not None and not calendar.empty:
                    # Try to get earnings date
                    if 'Earnings Date' in calendar.index:
                        earnings_date = calendar.loc['Earnings Date']
                        if hasattr(earnings_date, 'iloc'):
                            earnings_date = earnings_date.iloc[0]
                        
                        if earnings_date:
                            if hasattr(earnings_date, 'strftime'):
                                date_str = earnings_date.strftime('%b %d')
                            else:
                                date_str = str(earnings_date)[:10]
                            
                            earnings.append({
                                'ticker': ticker_symbol,
                                'name': ticker.info.get('shortName', ticker_symbol)[:15],
                                'date': date_str,
                                'time': 'TBD'
                            })
            except Exception as e:
                continue
        
        # Sort by date if possible
        return earnings[:6]
        
    except Exception as e:
        print(f"Error fetching earnings: {e}")
        return []


def get_economic_events():
    """Get upcoming economic events (Fed calendar)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(
            'https://www.federalreserve.gov/json/calendar.json',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            
            result = []
            now = datetime.now()
            
            for event in events[:20]:
                try:
                    event_date_str = event.get('start_date', '')
                    event_title = event.get('title', '')
                    
                    if event_date_str:
                        # Parse date
                        try:
                            event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                            # Only include upcoming events
                            if event_date >= now - timedelta(days=1):
                                result.append({
                                    'date': event_date.strftime('%b %d'),
                                    'event': event_title[:50] if event_title else 'Fed Event',
                                    'type': 'Fed'
                                })
                        except:
                            pass
                except:
                    continue
            
            return result[:5]
        
        return []
        
    except Exception as e:
        print(f"Error fetching economic events: {e}")
        return []


def get_market_status():
    """Get current market status"""
    try:
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute
        
        # Simple market hours check (ET timezone assumed)
        # Markets open 9:30 AM - 4:00 PM ET, Monday-Friday
        if weekday >= 5:  # Weekend
            return {'status': 'Closed', 'message': 'Weekend', 'color': 'neutral'}
        
        market_open = 9 * 60 + 30  # 9:30 AM in minutes
        market_close = 16 * 60  # 4:00 PM in minutes
        current_time = hour * 60 + minute
        
        if current_time < market_open:
            return {'status': 'Pre-Market', 'message': 'Opens 9:30 AM ET', 'color': 'neutral'}
        elif current_time >= market_close:
            return {'status': 'After Hours', 'message': 'Closed 4:00 PM ET', 'color': 'neutral'}
        else:
            return {'status': 'Open', 'message': 'Trading', 'color': 'positive'}
            
    except:
        return {'status': 'Unknown', 'message': '', 'color': 'neutral'}


@dashboard_bp.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Main dashboard endpoint"""
    global _dashboard_cache, _dashboard_cache_time
    
    cache_key = 'dashboard_main'
    current_time = time.time()
    
    # Check cache
    if cache_key in _dashboard_cache:
        cache_age = current_time - _dashboard_cache_time.get(cache_key, 0)
        if cache_age < DASHBOARD_CACHE_TTL:
            return jsonify(_dashboard_cache[cache_key])
    
    try:
        # Fetch all data
        response = {
            'indices': get_market_indices(),
            'futures': get_futures_data(),
            'forex': get_forex_data(),
            'crypto': get_crypto_data(),
            'movers': get_market_movers(),
            'news': get_market_news(),
            'earnings': get_upcoming_earnings(),
            'economic_events': get_economic_events(),
            'market_status': get_market_status(),
            'timestamp': datetime.now().isoformat(),
            'cache_ttl': DASHBOARD_CACHE_TTL
        }
        
        # Cache the response
        _dashboard_cache[cache_key] = response
        _dashboard_cache_time[cache_key] = current_time
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'indices': [],
            'futures': [],
            'forex': [],
            'crypto': [],
            'movers': {'gainers': [], 'losers': [], 'most_active': []},
            'news': [],
            'earnings': [],
            'economic_events': [],
            'market_status': {'status': 'Unknown', 'message': '', 'color': 'neutral'},
            'timestamp': datetime.now().isoformat()
        }), 500


@dashboard_bp.route('/api/dashboard/indices', methods=['GET'])
def get_indices():
    """Get market indices only"""
    try:
        return jsonify({
            'indices': get_market_indices(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'indices': []}), 500


@dashboard_bp.route('/api/dashboard/movers', methods=['GET'])
def get_movers():
    """Get market movers only"""
    try:
        return jsonify({
            'movers': get_market_movers(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'movers': {}}), 500


@dashboard_bp.route('/api/dashboard/news', methods=['GET'])
def get_news():
    """Get market news only"""
    try:
        return jsonify({
            'news': get_market_news(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'news': []}), 500


@dashboard_bp.route('/api/dashboard/health', methods=['GET'])
def dashboard_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'dashboard'
    })
