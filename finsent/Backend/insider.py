from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime, timedelta
from cache import financial_cache, Cache
import math
import pandas as pd

insider_bp = Blueprint('insider', __name__)


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def safe_round(value, decimals=2):
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return round(cleaned, decimals)
    except:
        return None


def parse_percentage(value):
    """
    Parse a percentage value from various formats yfinance might return.
    Returns a float between 0 and 100, or None if invalid.
    """
    if value is None:
        return None
    
    try:
        # Handle string formats
        if isinstance(value, str):
            # Remove % sign and whitespace
            cleaned = value.replace('%', '').strip()
            if not cleaned:
                return None
            value = float(cleaned)
        
        # Convert to float if not already
        value = float(value)
        
        # Check for NaN or Inf
        if math.isnan(value) or math.isinf(value):
            return None
        
        # Determine if value is decimal (0-1) or percentage (0-100)
        # Values > 1 and <= 100 are assumed to be percentages already
        # Values <= 1 are assumed to be decimals that need * 100
        # Values > 100 are likely errors - try to correct if possible
        if value <= 1 and value >= 0:
            # Decimal format (e.g., 0.7066 for 70.66%)
            value = value * 100
        elif value > 100:
            # Likely an error - value might be in basis points or double-converted
            # Try to detect if it's a reasonable percentage * 100
            if value <= 10000:
                # Could be basis points (7066 = 70.66%)
                value = value / 100
            else:
                # Too large, likely corrupted data
                return None
        
        # Final validation - percentage should be 0-100
        if value < 0 or value > 100:
            return None
        
        return round(value, 2)
        
    except (ValueError, TypeError):
        return None


def format_currency(value):
    if value is None:
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"${value / 1e3:.1f}K"
    else:
        return f"${value:,.0f}"


def format_shares(value):
    if value is None:
        return 'N/A'
    abs_value = abs(value)
    if abs_value >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"{value / 1e3:.1f}K"
    else:
        return f"{value:,.0f}"


def get_insider_data(ticker):
    cache_key = 'insider_data'
    cached = financial_cache.get(ticker, cache_key)
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        quote_type = info.get('quoteType', '').upper()
        if quote_type not in ['EQUITY', 'STOCK', '']:
            return {
                'error': f'Insider data not available for {quote_type}. Please enter a stock ticker.',
                'error_type': 'invalid_asset'
            }
        
        if not info.get('shortName') and not info.get('longName'):
            return {
                'error': 'Ticker not recognized. Please enter a valid stock symbol.',
                'error_type': 'invalid_ticker'
            }
        
        insider_transactions = None
        insider_purchases = None
        institutional_holders = None
        major_holders = None
        
        try:
            insider_transactions = stock.insider_transactions
        except:
            pass
        
        try:
            insider_purchases = stock.insider_purchases
        except:
            pass
        
        try:
            institutional_holders = stock.institutional_holders
        except:
            pass
        
        try:
            major_holders = stock.major_holders
        except:
            pass
        
        data = {
            'info': info,
            'insider_transactions': insider_transactions,
            'insider_purchases': insider_purchases,
            'institutional_holders': institutional_holders,
            'major_holders': major_holders,
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set(ticker, cache_key, data, Cache.TTL_FUNDAMENTAL)
        
        return data
        
    except Exception as e:
        return {
            'error': f'Unable to fetch data for {ticker}. Please try again.',
            'error_type': 'fetch_error',
            'details': str(e)
        }


def parse_transaction_type(text):
    if text is None:
        return 'unknown', 'neutral'
    
    text_lower = str(text).lower()
    
    if 'purchase' in text_lower or 'buy' in text_lower:
        return 'Purchase', 'positive'
    elif 'sale' in text_lower or 'sell' in text_lower:
        return 'Sale', 'negative'
    elif 'option' in text_lower or 'exercise' in text_lower:
        return 'Option Exercise', 'neutral'
    elif 'gift' in text_lower:
        return 'Gift', 'neutral'
    elif 'automatic' in text_lower:
        return 'Automatic', 'neutral'
    else:
        return str(text)[:20], 'neutral'


def get_insider_title(text):
    if text is None:
        return 'Insider'
    
    text_str = str(text)
    
    title_mappings = {
        'CEO': ['chief executive', 'ceo'],
        'CFO': ['chief financial', 'cfo'],
        'COO': ['chief operating', 'coo'],
        'CTO': ['chief technology', 'cto'],
        'President': ['president'],
        'Director': ['director'],
        'VP': ['vice president', 'vp'],
        'SVP': ['senior vice president', 'svp'],
        'EVP': ['executive vice president', 'evp'],
        'Chairman': ['chairman', 'chair'],
        '10% Owner': ['10%', 'beneficial owner'],
        'Officer': ['officer'],
        'General Counsel': ['general counsel', 'legal']
    }
    
    text_lower = text_str.lower()
    
    for title, keywords in title_mappings.items():
        for keyword in keywords:
            if keyword in text_lower:
                return title
    
    if len(text_str) > 25:
        return text_str[:22] + '...'
    return text_str


def filter_transactions_by_period(transactions, months):
    if transactions is None or transactions.empty:
        return pd.DataFrame()
    
    cutoff_date = datetime.now() - timedelta(days=months * 30)
    
    filtered = []
    for idx, row in transactions.iterrows():
        try:
            if 'Start Date' in transactions.columns:
                date_val = row.get('Start Date')
            elif 'Date' in transactions.columns:
                date_val = row.get('Date')
            else:
                date_val = idx
            
            if pd.isna(date_val):
                continue
            
            if isinstance(date_val, str):
                try:
                    date_val = pd.to_datetime(date_val)
                except:
                    continue
            
            if hasattr(date_val, 'to_pydatetime'):
                date_val = date_val.to_pydatetime()
            
            if hasattr(date_val, 'tzinfo') and date_val.tzinfo is not None:
                date_val = date_val.replace(tzinfo=None)
            
            if date_val >= cutoff_date:
                filtered.append(row)
        except:
            continue
    
    if filtered:
        return pd.DataFrame(filtered)
    return pd.DataFrame()


def process_transactions(data, months=12):
    transactions_df = data.get('insider_transactions')
    
    if transactions_df is None or transactions_df.empty:
        return {
            'transactions': [],
            'summary': {
                'total_buys': 0,
                'total_sells': 0,
                'buy_value': 0,
                'sell_value': 0,
                'net_value': 0,
                'buy_shares': 0,
                'sell_shares': 0
            },
            'monthly_data': [],
            'cluster_alerts': [],
            'has_data': False
        }
    
    filtered_df = filter_transactions_by_period(transactions_df, months)
    
    if filtered_df.empty:
        filtered_df = transactions_df.head(50)
    
    transactions = []
    total_buys = 0
    total_sells = 0
    buy_value = 0
    sell_value = 0
    buy_shares = 0
    sell_shares = 0
    
    monthly_buys = {}
    monthly_sells = {}
    
    cluster_window = {}
    
    for idx, row in filtered_df.iterrows():
        try:
            insider_name = str(row.get('Insider Trading', row.get('Insider', 'Unknown')))
            if pd.isna(insider_name) or insider_name == 'nan':
                insider_name = 'Unknown Insider'
            
            position = row.get('Position', row.get('Relationship', ''))
            title = get_insider_title(position)
            
            trans_text = row.get('Text', row.get('Transaction', ''))
            trans_type, trans_status = parse_transaction_type(trans_text)
            
            shares = clean_value(row.get('Shares', 0))
            if shares is None:
                shares = 0
            shares = abs(float(shares))
            
            value = clean_value(row.get('Value', 0))
            if value is None:
                value = 0
            value = abs(float(value))
            
            if 'Start Date' in filtered_df.columns:
                date_val = row.get('Start Date')
            elif 'Date' in filtered_df.columns:
                date_val = row.get('Date')
            else:
                date_val = idx
            
            if pd.isna(date_val):
                date_str = 'Unknown'
                month_key = None
            else:
                if isinstance(date_val, str):
                    try:
                        date_val = pd.to_datetime(date_val)
                    except:
                        date_str = date_val
                        month_key = None
                
                if hasattr(date_val, 'strftime'):
                    date_str = date_val.strftime('%b %d, %Y')
                    month_key = date_val.strftime('%Y-%m')
                else:
                    date_str = str(date_val)
                    month_key = None
            
            transaction = {
                'insider': insider_name[:30] if len(insider_name) > 30 else insider_name,
                'title': title,
                'type': trans_type,
                'type_status': trans_status,
                'shares': shares,
                'shares_display': format_shares(shares),
                'value': value,
                'value_display': format_currency(value),
                'date': date_str,
                'date_raw': str(date_val) if date_val is not None else None
            }
            
            transactions.append(transaction)
            
            if trans_status == 'positive':
                total_buys += 1
                buy_value += value
                buy_shares += shares
                if month_key:
                    monthly_buys[month_key] = monthly_buys.get(month_key, 0) + 1
                    
                    week_key = date_val.strftime('%Y-%W') if hasattr(date_val, 'strftime') else None
                    if week_key:
                        if week_key not in cluster_window:
                            cluster_window[week_key] = {'buys': [], 'sells': [], 'date': date_val}
                        cluster_window[week_key]['buys'].append({
                            'name': insider_name,
                            'title': title,
                            'value': value,
                            'shares': shares
                        })
                        
            elif trans_status == 'negative':
                total_sells += 1
                sell_value += value
                sell_shares += shares
                if month_key:
                    monthly_sells[month_key] = monthly_sells.get(month_key, 0) + 1
                    
                    week_key = date_val.strftime('%Y-%W') if hasattr(date_val, 'strftime') else None
                    if week_key:
                        if week_key not in cluster_window:
                            cluster_window[week_key] = {'buys': [], 'sells': [], 'date': date_val}
                        cluster_window[week_key]['sells'].append({
                            'name': insider_name,
                            'title': title,
                            'value': value,
                            'shares': shares
                        })
                        
        except Exception as e:
            continue
    
    all_months = set(list(monthly_buys.keys()) + list(monthly_sells.keys()))
    monthly_data = []
    
    for month in sorted(all_months):
        try:
            month_date = datetime.strptime(month, '%Y-%m')
            month_label = month_date.strftime('%b %Y')
        except:
            month_label = month
        
        monthly_data.append({
            'month': month,
            'label': month_label,
            'buys': monthly_buys.get(month, 0),
            'sells': monthly_sells.get(month, 0)
        })
    
    monthly_data = monthly_data[-12:]
    
    # Process cluster alerts with more detail
    cluster_alerts = []
    for week, week_data in cluster_window.items():
        unique_buyers = {}
        unique_sellers = {}
        
        # Dedupe by name and aggregate
        for buy in week_data['buys']:
            name = buy['name']
            if name not in unique_buyers:
                unique_buyers[name] = {'title': buy['title'], 'total_value': 0, 'total_shares': 0}
            unique_buyers[name]['total_value'] += buy['value']
            unique_buyers[name]['total_shares'] += buy['shares']
        
        for sell in week_data['sells']:
            name = sell['name']
            if name not in unique_sellers:
                unique_sellers[name] = {'title': sell['title'], 'total_value': 0, 'total_shares': 0}
            unique_sellers[name]['total_value'] += sell['value']
            unique_sellers[name]['total_shares'] += sell['shares']
        
        # Format week date for display
        try:
            week_date = week_data.get('date')
            if week_date and hasattr(week_date, 'strftime'):
                week_display = week_date.strftime('%b %d, %Y')
            else:
                week_display = week
        except:
            week_display = week
        
        if len(unique_buyers) >= 2:
            total_cluster_value = sum(b['total_value'] for b in unique_buyers.values())
            insiders_list = [{'name': name, 'title': info['title'], 'value': format_currency(info['total_value'])} 
                          for name, info in list(unique_buyers.items())[:5]]
            cluster_alerts.append({
                'type': 'cluster_buy',
                'status': 'positive',
                'message': f'{len(unique_buyers)} insiders bought within the same week',
                'description': f'Total cluster buying value: {format_currency(total_cluster_value)}',
                'insiders': insiders_list,
                'insider_count': len(unique_buyers),
                'total_value': total_cluster_value,
                'total_value_display': format_currency(total_cluster_value),
                'week': week,
                'week_display': week_display
            })
        
        if len(unique_sellers) >= 2:
            total_cluster_value = sum(s['total_value'] for s in unique_sellers.values())
            insiders_list = [{'name': name, 'title': info['title'], 'value': format_currency(info['total_value'])} 
                          for name, info in list(unique_sellers.items())[:5]]
            cluster_alerts.append({
                'type': 'cluster_sell',
                'status': 'negative',
                'message': f'{len(unique_sellers)} insiders sold within the same week',
                'description': f'Total cluster selling value: {format_currency(total_cluster_value)}',
                'insiders': insiders_list,
                'insider_count': len(unique_sellers),
                'total_value': total_cluster_value,
                'total_value_display': format_currency(total_cluster_value),
                'week': week,
                'week_display': week_display
            })
    
    cluster_alerts = sorted(cluster_alerts, key=lambda x: x['week'], reverse=True)[:10]
    
    net_value = buy_value - sell_value
    
    return {
        'transactions': transactions[:50],
        'summary': {
            'total_buys': total_buys,
            'total_sells': total_sells,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'net_value': net_value,
            'buy_shares': buy_shares,
            'sell_shares': sell_shares,
            'buy_value_display': format_currency(buy_value),
            'sell_value_display': format_currency(sell_value),
            'net_value_display': format_currency(abs(net_value)),
            'net_positive': net_value >= 0
        },
        'monthly_data': monthly_data,
        'cluster_alerts': cluster_alerts,
        'has_data': len(transactions) > 0
    }


def calculate_insider_sentiment(summary):
    total_buys = summary.get('total_buys', 0)
    total_sells = summary.get('total_sells', 0)
    net_value = summary.get('net_value', 0)
    
    total_transactions = total_buys + total_sells
    
    if total_transactions == 0:
        return {
            'sentiment': 'No Data',
            'status': 'neutral',
            'score': 0,
            'description': 'No insider transactions found for this period'
        }
    
    buy_ratio = total_buys / total_transactions if total_transactions > 0 else 0
    
    if buy_ratio >= 0.7 and net_value > 0:
        return {
            'sentiment': 'Very Bullish',
            'status': 'positive',
            'score': 5,
            'description': 'Strong insider buying activity with significant net purchases'
        }
    elif buy_ratio >= 0.5 and net_value > 0:
        return {
            'sentiment': 'Bullish',
            'status': 'positive',
            'score': 4,
            'description': 'Insiders are net buyers during this period'
        }
    elif buy_ratio >= 0.4:
        return {
            'sentiment': 'Neutral',
            'status': 'neutral',
            'score': 3,
            'description': 'Mixed insider activity with balanced buying and selling'
        }
    elif buy_ratio >= 0.2:
        return {
            'sentiment': 'Bearish',
            'status': 'negative',
            'score': 2,
            'description': 'Insiders are net sellers during this period'
        }
    else:
        return {
            'sentiment': 'Very Bearish',
            'status': 'negative',
            'score': 1,
            'description': 'Strong insider selling activity with minimal purchases'
        }


def process_institutional_holdings(data):
    institutional_df = data.get('institutional_holders')
    major_df = data.get('major_holders')
    
    result = {
        'holders': [],
        'summary': {
            'total_institutional': None,
            'total_insider': None,
            'top_holders_percent': 0
        },
        'has_data': False
    }
    
    # Parse major holders for summary stats
    if major_df is not None and not major_df.empty:
        try:
            # yfinance major_holders DataFrame structure:
            # Index contains descriptions, single column contains values
            for idx, row in major_df.iterrows():
                # Get value from first column
                value = row.iloc[0] if len(row) > 0 else None
                # Get label from index or second column if exists
                if len(row) > 1:
                    label = str(row.iloc[1])
                else:
                    label = str(idx)
                
                if value is not None:
                    # Parse the percentage value properly
                    parsed_pct = parse_percentage(value)
                    
                    if parsed_pct is not None:
                        label_lower = label.lower()
                        if 'institution' in label_lower and 'float' not in label_lower:
                            result['summary']['total_institutional'] = parsed_pct
                        elif 'insider' in label_lower:
                            result['summary']['total_insider'] = parsed_pct
        except Exception as e:
            pass
    
    # Parse individual institutional holders
    if institutional_df is not None and not institutional_df.empty:
        result['has_data'] = True
        
        for idx, row in institutional_df.head(10).iterrows():
            try:
                holder_name = str(row.get('Holder', 'Unknown'))
                if len(holder_name) > 35:
                    holder_name = holder_name[:32] + '...'
                
                shares = clean_value(row.get('Shares', 0))
                if shares is None:
                    shares = 0
                
                value = clean_value(row.get('Value', 0))
                if value is None:
                    value = 0
                
                # Parse percentage with proper validation
                pct_raw = row.get('% Out', row.get('pctHeld', 0))
                pct_out = parse_percentage(pct_raw)
                if pct_out is None:
                    pct_out = 0
                
                date_reported = row.get('Date Reported', None)
                if date_reported is not None and not pd.isna(date_reported):
                    if hasattr(date_reported, 'strftime'):
                        date_str = date_reported.strftime('%b %d, %Y')
                    else:
                        date_str = str(date_reported)[:10]
                else:
                    date_str = 'N/A'
                
                result['holders'].append({
                    'name': holder_name,
                    'shares': shares,
                    'shares_display': format_shares(shares),
                    'value': value,
                    'value_display': format_currency(value),
                    'percent': pct_out,
                    'percent_display': f"{pct_out}%" if pct_out else 'N/A',
                    'date_reported': date_str
                })
                
                result['summary']['top_holders_percent'] += pct_out
                
            except:
                continue
        
        result['summary']['top_holders_percent'] = safe_round(result['summary']['top_holders_percent'], 2)
    
    return result


def get_company_info(data):
    try:
        info = data.get('info', {})
        
        market_cap = clean_value(info.get('marketCap'))
        if market_cap:
            if market_cap >= 1e12:
                market_cap_display = f"${market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_display = f"${market_cap / 1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_display = f"${market_cap / 1e6:.2f}M"
            else:
                market_cap_display = f"${market_cap:,.0f}"
        else:
            market_cap_display = 'N/A'
        
        price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
        
        return {
            'name': info.get('longName') or info.get('shortName') or 'Unknown',
            'ticker': info.get('symbol', '').upper(),
            'sector': info.get('sector') or 'N/A',
            'industry': info.get('industry') or 'N/A',
            'price': price,
            'price_display': f"${price:.2f}" if price else 'N/A',
            'market_cap': market_cap,
            'market_cap_display': market_cap_display,
            'currency': info.get('currency', 'USD')
        }
        
    except Exception as e:
        return {
            'name': 'Unknown',
            'ticker': '',
            'sector': 'N/A',
            'industry': 'N/A',
            'price': None,
            'price_display': 'N/A',
            'market_cap': None,
            'market_cap_display': 'N/A',
            'currency': 'USD'
        }


def generate_key_signals(transaction_data, institutional_data, sentiment):
    signals = []
    
    summary = transaction_data.get('summary', {})
    cluster_alerts = transaction_data.get('cluster_alerts', [])
    
    for alert in cluster_alerts[:3]:
        signals.append({
            'type': alert['type'],
            'status': alert['status'],
            'title': 'Cluster Buying Detected' if alert['type'] == 'cluster_buy' else 'Cluster Selling Detected',
            'description': alert['message']
        })
    
    total_buys = summary.get('total_buys', 0)
    total_sells = summary.get('total_sells', 0)
    
    if total_buys > 0 and total_sells == 0:
        signals.append({
            'type': 'all_buys',
            'status': 'positive',
            'title': 'Buy-Only Period',
            'description': f'All {total_buys} transactions were purchases with no insider selling'
        })
    elif total_sells > 0 and total_buys == 0:
        signals.append({
            'type': 'all_sells',
            'status': 'negative',
            'title': 'Sell-Only Period',
            'description': f'All {total_sells} transactions were sales with no insider buying'
        })
    
    buy_value = summary.get('buy_value', 0)
    if buy_value >= 10000000:
        signals.append({
            'type': 'large_buys',
            'status': 'positive',
            'title': 'Significant Buy Volume',
            'description': f'Insider purchases totaling {summary.get("buy_value_display", "N/A")}'
        })
    
    sell_value = summary.get('sell_value', 0)
    if sell_value >= 50000000:
        signals.append({
            'type': 'large_sells',
            'status': 'negative',
            'title': 'Heavy Selling',
            'description': f'Insider sales totaling {summary.get("sell_value_display", "N/A")}'
        })
    
    inst_summary = institutional_data.get('summary', {})
    inst_pct = inst_summary.get('total_institutional')
    if inst_pct is not None:
        if inst_pct >= 80:
            signals.append({
                'type': 'high_institutional',
                'status': 'neutral',
                'title': 'High Institutional Ownership',
                'description': f'{inst_pct}% owned by institutions - highly followed stock'
            })
        elif inst_pct <= 20:
            signals.append({
                'type': 'low_institutional',
                'status': 'neutral',
                'title': 'Low Institutional Ownership',
                'description': f'Only {inst_pct}% owned by institutions'
            })
    
    return signals[:5]


@insider_bp.route('/api/insider', methods=['POST'])
def analyze_insider():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        months = data.get('months', 12)
        
        try:
            months = int(months)
            if months < 1:
                months = 3
            elif months > 24:
                months = 24
        except:
            months = 12
        
        if not ticker:
            return jsonify({
                'error': 'Ticker is required',
                'error_type': 'validation'
            }), 400
        
        stock_data = get_insider_data(ticker)
        
        if 'error' in stock_data:
            return jsonify(stock_data), 400
        
        company = get_company_info(stock_data)
        transaction_data = process_transactions(stock_data, months)
        institutional_data = process_institutional_holdings(stock_data)
        sentiment = calculate_insider_sentiment(transaction_data['summary'])
        signals = generate_key_signals(transaction_data, institutional_data, sentiment)
        
        response = {
            'ticker': ticker,
            'company': company,
            'period_months': months,
            'transactions': transaction_data['transactions'],
            'summary': transaction_data['summary'],
            'monthly_data': transaction_data['monthly_data'],
            'cluster_alerts': transaction_data['cluster_alerts'],
            'sentiment': sentiment,
            'institutional': institutional_data,
            'signals': signals,
            'has_transaction_data': transaction_data['has_data'],
            'has_institutional_data': institutional_data['has_data'],
            'timestamp': datetime.now().isoformat(),
            'disclaimer': (
                "Insider trading data is sourced from SEC Form 4 filings via Yahoo Finance and may be "
                "delayed or incomplete. Insider transactions have many legitimate motivations including "
                "diversification, tax planning, and personal needs. This data should not be used as the "
                "sole basis for investment decisions. Always conduct thorough research and consult a "
                "qualified financial advisor."
            )
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@insider_bp.route('/api/insider/health', methods=['GET'])
def insider_health():
    return jsonify({
        'status': 'healthy',
        'service': 'insider_trading'
    })
