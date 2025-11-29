from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime
import math

intrinsic_bp = Blueprint('intrinsic', __name__)

# Cache for intrinsic value calculations
_intrinsic_cache = {}
_intrinsic_cache_time = {}
INTRINSIC_CACHE_TTL = 300  # 5 minutes


def clean_value(value):
    """Clean and validate numeric values"""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
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
    """Format large numbers for display"""
    if value is None:
        return 'N/A'
    try:
        value = float(value)
        if abs(value) >= 1e12:
            return f"${value / 1e12:.2f}T"
        elif abs(value) >= 1e9:
            return f"${value / 1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value / 1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value / 1e3:.2f}K"
        else:
            return f"${value:.2f}"
    except:
        return 'N/A'


def safe_get_statement(dataframe, keys):
    """Safely get value from financial statement"""
    try:
        if dataframe is None or dataframe.empty:
            return None
        
        for key in keys:
            if key in dataframe.index:
                value = dataframe.loc[key].iloc[0] if len(dataframe.loc[key]) > 0 else None
                cleaned = clean_value(value)
                if cleaned is not None:
                    return cleaned
        return None
    except:
        return None


def get_stock_data(ticker):
    """Fetch comprehensive stock data for valuation"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Validate ticker
        quote_type = info.get('quoteType', '').upper()
        if quote_type not in ['EQUITY', 'STOCK', '']:
            return {
                'error': f'Intrinsic value not available for {quote_type}. Please enter a stock ticker.',
                'error_type': 'invalid_asset'
            }
        
        if not info.get('shortName') and not info.get('longName'):
            return {
                'error': 'Ticker not recognized. Please enter a valid stock symbol.',
                'error_type': 'invalid_ticker'
            }
        
        # Get financial statements
        cash_flow = stock.cashflow
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        
        return {
            'info': info,
            'stock': stock,
            'cash_flow': cash_flow,
            'balance_sheet': balance_sheet,
            'income_stmt': income_stmt
        }
        
    except Exception as e:
        return {
            'error': f'Unable to fetch data for {ticker}. Please try again.',
            'error_type': 'fetch_error'
        }


def get_historical_fcf(cash_flow):
    """Get historical free cash flow data"""
    try:
        if cash_flow is None or cash_flow.empty:
            return []
        
        fcf_list = []
        
        for i, col in enumerate(cash_flow.columns):
            year = col.year
            fcf = safe_get_statement(cash_flow[[col]], [
                'Free Cash Flow',
                'Operating Cash Flow'
            ])
            
            if fcf is None:
                continue
            
            fcf_list.append({
                'year': str(year),
                'value': fcf
            })
            
            if len(fcf_list) >= 5:
                break
        
        return fcf_list[::-1]  # Reverse to chronological order
        
    except:
        return []


def calculate_fcf_growth_rate(fcf_history):
    """Calculate historical FCF CAGR"""
    try:
        if len(fcf_history) < 2:
            return None
        
        # Filter out negative values for CAGR calculation
        positive_fcf = [f for f in fcf_history if f['value'] > 0]
        
        if len(positive_fcf) < 2:
            return None
        
        start_value = positive_fcf[0]['value']
        end_value = positive_fcf[-1]['value']
        years = len(positive_fcf) - 1
        
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return None
        
        cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
        
        # Cap at reasonable bounds
        cagr = max(-20, min(50, cagr))
        
        return cagr
        
    except:
        return None


def get_discount_rate(beta):
    """Calculate discount rate based on CAPM"""
    try:
        risk_free_rate = 4.5  # Current approximate 10-year Treasury
        market_premium = 5.5  # Historical equity risk premium
        
        if beta is None:
            beta = 1.0
        
        # Cap beta at reasonable bounds
        beta = max(0.5, min(2.5, beta))
        
        discount_rate = risk_free_rate + (beta * market_premium)
        
        return discount_rate
        
    except:
        return 10.0  # Default discount rate


def calculate_dcf_value(data):
    """Calculate DCF intrinsic value"""
    try:
        info = data['info']
        cash_flow = data['cash_flow']
        balance_sheet = data['balance_sheet']
        
        # Get current free cash flow
        fcf_history = get_historical_fcf(cash_flow)
        
        if not fcf_history:
            return None, "Unable to retrieve free cash flow data"
        
        current_fcf = fcf_history[-1]['value'] if fcf_history else None
        
        if current_fcf is None or current_fcf <= 0:
            # Try to use operating cash flow
            current_fcf = safe_get_statement(cash_flow, [
                'Operating Cash Flow',
                'Cash Flow From Continuing Operating Activities'
            ])
            if current_fcf is None or current_fcf <= 0:
                return None, "Company has negative or no cash flow"
        
        # Get growth rate
        historical_growth = calculate_fcf_growth_rate(fcf_history)
        analyst_growth = clean_value(info.get('earningsGrowth'))
        if analyst_growth:
            analyst_growth = analyst_growth * 100
        
        # Use blended growth rate
        if historical_growth and analyst_growth:
            base_growth = (historical_growth + analyst_growth) / 2
        elif analyst_growth:
            base_growth = analyst_growth
        elif historical_growth:
            base_growth = historical_growth
        else:
            base_growth = 5.0  # Conservative default
        
        # Cap growth rates
        growth_rate = min(25, max(-5, base_growth))
        terminal_growth = 2.5  # Long-term GDP growth
        
        # Get discount rate
        beta = clean_value(info.get('beta'))
        discount_rate = get_discount_rate(beta)
        
        # Get balance sheet items
        total_debt = safe_get_statement(balance_sheet, [
            'Total Debt',
            'Long Term Debt',
            'Long Term Debt And Capital Lease Obligation'
        ]) or 0
        
        cash = safe_get_statement(balance_sheet, [
            'Cash And Cash Equivalents',
            'Cash Cash Equivalents And Short Term Investments',
            'Cash And Short Term Investments'
        ]) or 0
        
        shares_outstanding = clean_value(info.get('sharesOutstanding'))
        
        if not shares_outstanding:
            return None, "Unable to determine shares outstanding"
        
        # Project future cash flows (5 years)
        projected_cf = []
        fcf = current_fcf
        
        for year in range(1, 6):
            fcf = fcf * (1 + growth_rate / 100)
            discounted_cf = fcf / ((1 + discount_rate / 100) ** year)
            projected_cf.append({
                'year': year,
                'fcf': fcf,
                'discounted': discounted_cf
            })
        
        # Calculate terminal value
        terminal_fcf = projected_cf[-1]['fcf'] * (1 + terminal_growth / 100)
        terminal_value = terminal_fcf / ((discount_rate / 100) - (terminal_growth / 100))
        discounted_terminal = terminal_value / ((1 + discount_rate / 100) ** 5)
        
        # Sum of discounted cash flows
        sum_dcf = sum(cf['discounted'] for cf in projected_cf)
        
        # Enterprise value
        enterprise_value = sum_dcf + discounted_terminal
        
        # Equity value
        equity_value = enterprise_value - total_debt + cash
        
        # Intrinsic value per share
        intrinsic_value = equity_value / shares_outstanding
        
        return {
            'value': intrinsic_value,
            'fcf_ttm': current_fcf,
            'growth_rate': growth_rate,
            'discount_rate': discount_rate,
            'terminal_value': terminal_value,
            'fcf_history': fcf_history
        }, None
        
    except Exception as e:
        return None, f"DCF calculation error: {str(e)}"


def get_industry_pe(sector):
    """Get approximate industry P/E ratio by sector"""
    sector_pe = {
        'Technology': 28,
        'Healthcare': 22,
        'Financial Services': 14,
        'Consumer Cyclical': 20,
        'Consumer Defensive': 22,
        'Industrials': 20,
        'Energy': 12,
        'Utilities': 18,
        'Real Estate': 35,
        'Basic Materials': 15,
        'Communication Services': 18,
    }
    return sector_pe.get(sector, 18)  # Default market average


def get_industry_ev_ebitda(sector):
    """Get approximate industry EV/EBITDA by sector"""
    sector_ev = {
        'Technology': 18,
        'Healthcare': 14,
        'Financial Services': 10,
        'Consumer Cyclical': 12,
        'Consumer Defensive': 14,
        'Industrials': 12,
        'Energy': 6,
        'Utilities': 10,
        'Real Estate': 16,
        'Basic Materials': 8,
        'Communication Services': 10,
    }
    return sector_ev.get(sector, 12)


def calculate_relative_value(data):
    """Calculate relative valuation using peer multiples"""
    try:
        info = data['info']
        
        current_price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
        
        if not current_price:
            return None, "Unable to get current price"
        
        valuations = []
        
        # P/E Based Valuation
        trailing_eps = clean_value(info.get('trailingEps'))
        sector = info.get('sector', '')
        industry_pe = get_industry_pe(sector)
        
        if trailing_eps and industry_pe:
            pe_value = trailing_eps * industry_pe
            valuations.append({
                'method': 'P/E (Industry Avg)',
                'value': pe_value,
                'multiple': industry_pe,
                'base_metric': trailing_eps,
                'base_name': 'EPS (TTM)'
            })
        
        # EV/EBITDA Based Valuation
        ebitda = clean_value(info.get('ebitda'))
        shares = clean_value(info.get('sharesOutstanding'))
        total_debt = clean_value(info.get('totalDebt')) or 0
        cash = clean_value(info.get('totalCash')) or 0
        industry_ev = get_industry_ev_ebitda(sector)
        
        if ebitda and shares and industry_ev:
            ev = ebitda * industry_ev
            equity_value = ev - total_debt + cash
            ev_value = equity_value / shares
            valuations.append({
                'method': 'EV/EBITDA',
                'value': ev_value,
                'multiple': industry_ev,
                'base_metric': ebitda,
                'base_name': 'EBITDA'
            })
        
        if not valuations:
            return None, "Could not calculate relative value"
        
        # Calculate weighted average
        if len(valuations) > 0:
            weights = [1.0] * len(valuations)
            weighted_values = [v['value'] * w for v, w in zip(valuations, weights)]
            avg_value = sum(weighted_values) / sum(weights)
        else:
            return None, "Could not calculate relative value"
        
        return {
            'value': avg_value,
            'methods': valuations,
            'current_price': current_price,
            'sector': sector,
            'company_pe': clean_value(info.get('trailingPE')),
            'company_ev_ebitda': clean_value(info.get('enterpriseToEbitda')),
            'company_ps': clean_value(info.get('priceToSalesTrailing12Months')),
            'company_pb': clean_value(info.get('priceToBook'))
        }, None
        
    except Exception as e:
        return None, f"Relative valuation error: {str(e)}"


def combine_valuations(dcf_result, relative_result, info):
    """Combine DCF and relative valuation with appropriate weighting"""
    try:
        dcf_value = dcf_result['value'] if dcf_result else None
        relative_value = relative_result['value'] if relative_result else None
        
        current_price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
        
        if dcf_value and relative_value:
            # Weight DCF at 60%, relative at 40%
            dcf_weight = 0.6
            relative_weight = 0.4
            
            combined_value = (dcf_value * dcf_weight) + (relative_value * relative_weight)
            
            return {
                'intrinsic_value': combined_value,
                'current_price': current_price,
                'upside_percent': ((combined_value / current_price) - 1) * 100 if current_price else None,
                'dcf_weight': dcf_weight * 100,
                'relative_weight': relative_weight * 100
            }
        elif dcf_value:
            return {
                'intrinsic_value': dcf_value,
                'current_price': current_price,
                'upside_percent': ((dcf_value / current_price) - 1) * 100 if current_price else None,
                'dcf_weight': 100,
                'relative_weight': 0
            }
        elif relative_value:
            return {
                'intrinsic_value': relative_value,
                'current_price': current_price,
                'upside_percent': ((relative_value / current_price) - 1) * 100 if current_price else None,
                'dcf_weight': 0,
                'relative_weight': 100
            }
        else:
            return None
            
    except Exception as e:
        return None


def get_company_info(info):
    """Extract company information"""
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
    
    current_price = clean_value(info.get('currentPrice')) or clean_value(info.get('regularMarketPrice'))
    
    return {
        'name': info.get('longName') or info.get('shortName') or 'Unknown',
        'ticker': info.get('symbol', '').upper(),
        'sector': info.get('sector') or 'N/A',
        'industry': info.get('industry') or 'N/A',
        'price': current_price,
        'price_display': f"${current_price:.2f}" if current_price else 'N/A',
        'market_cap': market_cap,
        'market_cap_display': market_cap_display
    }


@intrinsic_bp.route('/api/intrinsic', methods=['POST'])
def analyze_intrinsic():
    """Main intrinsic value endpoint"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        
        if not ticker:
            return jsonify({
                'error': 'Ticker is required',
                'error_type': 'validation'
            }), 400
        
        # Check cache
        cache_key = ticker
        current_time = datetime.now().timestamp()
        
        if cache_key in _intrinsic_cache:
            cache_age = current_time - _intrinsic_cache_time.get(cache_key, 0)
            if cache_age < INTRINSIC_CACHE_TTL:
                return jsonify(_intrinsic_cache[cache_key])
        
        # Get stock data
        stock_data = get_stock_data(ticker)
        
        if 'error' in stock_data:
            return jsonify(stock_data), 400
        
        info = stock_data['info']
        
        # Get company info
        company = get_company_info(info)
        
        # Calculate DCF value
        dcf_result, dcf_error = calculate_dcf_value(stock_data)
        
        # Calculate relative value
        relative_result, relative_error = calculate_relative_value(stock_data)
        
        # Combine valuations
        combined = combine_valuations(dcf_result, relative_result, info)
        
        if not combined:
            return jsonify({
                'error': 'Unable to calculate intrinsic value. Insufficient financial data.',
                'error_type': 'calculation_error',
                'dcf_error': dcf_error,
                'relative_error': relative_error
            }), 400
        
        # Build response
        response = {
            'company': company,
            'intrinsic_value': safe_round(combined['intrinsic_value'], 2),
            'intrinsic_value_display': f"${combined['intrinsic_value']:.2f}" if combined['intrinsic_value'] else 'N/A',
            'upside_percent': safe_round(combined['upside_percent'], 1),
            'dcf': {
                'value': safe_round(dcf_result['value'], 2) if dcf_result else None,
                'value_display': f"${dcf_result['value']:.2f}" if dcf_result else 'N/A',
                'fcf_ttm': dcf_result['fcf_ttm'] if dcf_result else None,
                'fcf_ttm_display': format_large_number(dcf_result['fcf_ttm']) if dcf_result else 'N/A',
                'growth_rate': safe_round(dcf_result['growth_rate'], 1) if dcf_result else None,
                'discount_rate': safe_round(dcf_result['discount_rate'], 1) if dcf_result else None,
                'terminal_value': dcf_result['terminal_value'] if dcf_result else None,
                'terminal_value_display': format_large_number(dcf_result['terminal_value']) if dcf_result else 'N/A',
                'fcf_history': dcf_result['fcf_history'] if dcf_result else [],
                'error': dcf_error
            },
            'relative': {
                'value': safe_round(relative_result['value'], 2) if relative_result else None,
                'value_display': f"${relative_result['value']:.2f}" if relative_result else 'N/A',
                'sector': relative_result['sector'] if relative_result else None,
                'methods': relative_result['methods'] if relative_result else [],
                'company_pe': safe_round(relative_result['company_pe'], 1) if relative_result else None,
                'company_ev_ebitda': safe_round(relative_result['company_ev_ebitda'], 1) if relative_result else None,
                'company_ps': safe_round(relative_result['company_ps'], 2) if relative_result else None,
                'company_pb': safe_round(relative_result['company_pb'], 2) if relative_result else None,
                'error': relative_error
            }
        }
        
        # Format relative methods for display
        if response['relative']['methods']:
            for method in response['relative']['methods']:
                method['value_display'] = f"${method['value']:.2f}" if method['value'] else 'N/A'
                method['multiple_display'] = f"{method['multiple']:.1f}x"
                method['base_display'] = format_large_number(method['base_metric'])
        
        # Cache the response
        _intrinsic_cache[cache_key] = response
        _intrinsic_cache_time[cache_key] = current_time
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500


@intrinsic_bp.route('/api/intrinsic/health', methods=['GET'])
def intrinsic_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'intrinsic_value'
    })
