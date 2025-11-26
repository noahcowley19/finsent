from flask import Blueprint, request, jsonify
import yfinance as yf
from datetime import datetime
from cache import financial_cache, Cache


financials_bp = Blueprint('financials', __name__)


def get_stock_data(ticker):
    cached = financial_cache.get(ticker, 'full_data')
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        quote_type = info.get('quoteType', '').upper()
        if quote_type not in ['EQUITY', 'STOCK', '']:
            return {
                'error': f'Financial statements not available for {quote_type}. Please enter a stock ticker.',
                'error_type': 'invalid_asset'
            }
        
        if not info.get('shortName') and not info.get('longName'):
            return {
                'error': 'Ticker not recognized. Please enter a valid stock symbol.',
                'error_type': 'invalid_ticker'
            }
        
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        income_stmt_annual = stock.income_stmt
        balance_sheet_annual = stock.balance_sheet
        cash_flow_annual = stock.cashflow
        
        data = {
            'info': info,
            'income_stmt': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'income_stmt_annual': income_stmt_annual,
            'balance_sheet_annual': balance_sheet_annual,
            'cash_flow_annual': cash_flow_annual,
            'timestamp': datetime.now().isoformat()
        }
        
        financial_cache.set(ticker, 'full_data', data, Cache.TTL_PRICE)
        
        return data
        
    except Exception as e:
        return {
            'error': f'Unable to fetch data for {ticker}. Please try again.',
            'error_type': 'fetch_error',
            'details': str(e)
        }


def safe_get(data, key, default=None):
    try:
        if hasattr(data, 'get'):
            return data.get(key, default)
        return default
    except:
        return default


def safe_get_statement(df, row_name, column_index=0, default=None):
    try:
        if df is None or df.empty:
            return default
        if row_name in df.index:
            value = df.loc[row_name].iloc[column_index]
            if value is not None and not (hasattr(value, '__iter__') and len(value) == 0):
                return float(value)
        return default
    except:
        return default


def safe_get_statement_yoy(df, row_name, default=None):
    try:
        if df is None or df.empty or len(df.columns) < 2:
            return default, default
        if row_name in df.index:
            current = float(df.loc[row_name].iloc[0])
            prior = float(df.loc[row_name].iloc[1])
            return current, prior
        return default, default
    except:
        return default, default


def calculate_piotroski_score(data):
    try:
        info = data.get('info', {})
        income_stmt = data.get('income_stmt')
        balance_sheet = data.get('balance_sheet')
        cash_flow = data.get('cash_flow')
        
        score = 0
        components = {}
        
        net_income = safe_get_statement(income_stmt, 'Net Income')
        if net_income is None:
            net_income = safe_get_statement(income_stmt, 'Net Income Common Stockholders')
        components['net_income_positive'] = None
        if net_income is not None:
            passed = net_income > 0
            components['net_income_positive'] = passed
            if passed:
                score += 1
        
        ocf = safe_get_statement(cash_flow, 'Operating Cash Flow')
        if ocf is None:
            ocf = safe_get_statement(cash_flow, 'Cash Flow From Continuing Operating Activities')
        components['ocf_positive'] = None
        if ocf is not None:
            passed = ocf > 0
            components['ocf_positive'] = passed
            if passed:
                score += 1
        
        total_assets_curr, total_assets_prior = safe_get_statement_yoy(balance_sheet, 'Total Assets')
        net_income_curr, net_income_prior = safe_get_statement_yoy(income_stmt, 'Net Income')
        if net_income_curr is None:
            net_income_curr, net_income_prior = safe_get_statement_yoy(income_stmt, 'Net Income Common Stockholders')
        
        components['roa_increasing'] = None
        if all(v is not None and v != 0 for v in [total_assets_curr, total_assets_prior, net_income_curr, net_income_prior]):
            roa_curr = net_income_curr / total_assets_curr
            roa_prior = net_income_prior / total_assets_prior
            passed = roa_curr > roa_prior
            components['roa_increasing'] = passed
            if passed:
                score += 1
        
        components['ocf_exceeds_net_income'] = None
        if ocf is not None and net_income is not None:
            passed = ocf > net_income
            components['ocf_exceeds_net_income'] = passed
            if passed:
                score += 1
        
        ltd_curr, ltd_prior = safe_get_statement_yoy(balance_sheet, 'Long Term Debt')
        components['leverage_decreasing'] = None
        if total_assets_curr and total_assets_prior and ltd_curr is not None and ltd_prior is not None:
            if total_assets_curr != 0 and total_assets_prior != 0:
                ltd_ratio_curr = ltd_curr / total_assets_curr
                ltd_ratio_prior = ltd_prior / total_assets_prior
                passed = ltd_ratio_curr <= ltd_ratio_prior
                components['leverage_decreasing'] = passed
                if passed:
                    score += 1
        
        ca_curr, ca_prior = safe_get_statement_yoy(balance_sheet, 'Current Assets')
        cl_curr, cl_prior = safe_get_statement_yoy(balance_sheet, 'Current Liabilities')
        components['current_ratio_increasing'] = None
        if all(v is not None and v != 0 for v in [ca_curr, ca_prior, cl_curr, cl_prior]):
            cr_curr = ca_curr / cl_curr
            cr_prior = ca_prior / cl_prior
            passed = cr_curr > cr_prior
            components['current_ratio_increasing'] = passed
            if passed:
                score += 1
        
        shares_curr, shares_prior = safe_get_statement_yoy(balance_sheet, 'Ordinary Shares Number')
        if shares_curr is None:
            shares_curr, shares_prior = safe_get_statement_yoy(balance_sheet, 'Share Issued')
        components['no_dilution'] = None
        if shares_curr is not None and shares_prior is not None:
            passed = shares_curr <= shares_prior
            components['no_dilution'] = passed
            if passed:
                score += 1
        
        gp_curr, gp_prior = safe_get_statement_yoy(income_stmt, 'Gross Profit')
        rev_curr, rev_prior = safe_get_statement_yoy(income_stmt, 'Total Revenue')
        components['gross_margin_increasing'] = None
        if all(v is not None and v != 0 for v in [gp_curr, gp_prior, rev_curr, rev_prior]):
            gm_curr = gp_curr / rev_curr
            gm_prior = gp_prior / rev_prior
            passed = gm_curr > gm_prior
            components['gross_margin_increasing'] = passed
            if passed:
                score += 1
        
        components['asset_turnover_increasing'] = None
        if all(v is not None and v != 0 for v in [rev_curr, rev_prior, total_assets_curr, total_assets_prior]):
            at_curr = rev_curr / total_assets_curr
            at_prior = rev_prior / total_assets_prior
            passed = at_curr > at_prior
            components['asset_turnover_increasing'] = passed
            if passed:
                score += 1
        
        if score >= 7:
            interpretation = "Strong"
            status = "positive"
        elif score >= 4:
            interpretation = "Neutral"
            status = "neutral"
        else:
            interpretation = "Weak"
            status = "negative"
        
        return {
            'score': score,
            'max_score': 9,
            'display': f"{score}/9",
            'interpretation': interpretation,
            'status': status,
            'components': components
        }
        
    except Exception as e:
        return {
            'score': None,
            'display': 'N/A',
            'interpretation': 'Insufficient Data',
            'status': 'neutral',
            'error': str(e)
        }


def calculate_altman_zscore(data):
    try:
        info = data.get('info', {})
        income_stmt = data.get('income_stmt')
        balance_sheet = data.get('balance_sheet')
        
        current_assets = safe_get_statement(balance_sheet, 'Current Assets')
        current_liabilities = safe_get_statement(balance_sheet, 'Current Liabilities')
        total_assets = safe_get_statement(balance_sheet, 'Total Assets')
        retained_earnings = safe_get_statement(balance_sheet, 'Retained Earnings')
        ebit = safe_get_statement(income_stmt, 'EBIT')
        if ebit is None:
            ebit = safe_get_statement(income_stmt, 'Operating Income')
        total_liabilities = safe_get_statement(balance_sheet, 'Total Liabilities Net Minority Interest')
        if total_liabilities is None:
            total_liabilities = safe_get_statement(balance_sheet, 'Total Debt')
        revenue = safe_get_statement(income_stmt, 'Total Revenue')
        market_cap = info.get('marketCap')
        
        if total_assets is None or total_assets == 0:
            raise ValueError("Total assets not available")
        
        working_capital = (current_assets or 0) - (current_liabilities or 0)
        
        A = working_capital / total_assets if total_assets else 0
        B = (retained_earnings / total_assets) if retained_earnings and total_assets else 0
        C = (ebit / total_assets) if ebit and total_assets else 0
        D = (market_cap / total_liabilities) if market_cap and total_liabilities and total_liabilities != 0 else 0
        E = (revenue / total_assets) if revenue and total_assets else 0
        
        z_score = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
        
        if z_score > 2.99:
            interpretation = "Safe Zone"
            status = "positive"
        elif z_score >= 1.81:
            interpretation = "Grey Zone"
            status = "neutral"
        else:
            interpretation = "Distress Zone"
            status = "negative"
        
        return {
            'score': round(z_score, 2),
            'display': f"{round(z_score, 2)}",
            'interpretation': interpretation,
            'status': status,
            'components': {
                'working_capital_to_assets': round(A, 4),
                'retained_earnings_to_assets': round(B, 4),
                'ebit_to_assets': round(C, 4),
                'market_cap_to_liabilities': round(D, 4),
                'revenue_to_assets': round(E, 4)
            }
        }
        
    except Exception as e:
        return {
            'score': None,
            'display': 'N/A',
            'interpretation': 'Insufficient Data',
            'status': 'neutral',
            'error': str(e)
        }


def calculate_beneish_mscore(data):
    try:
        info = data.get('info', {})
        income_stmt = data.get('income_stmt')
        balance_sheet = data.get('balance_sheet')
        cash_flow = data.get('cash_flow')
        
        receivables_curr, receivables_prior = safe_get_statement_yoy(balance_sheet, 'Accounts Receivable')
        if receivables_curr is None:
            receivables_curr, receivables_prior = safe_get_statement_yoy(balance_sheet, 'Receivables')
        
        revenue_curr, revenue_prior = safe_get_statement_yoy(income_stmt, 'Total Revenue')
        gross_profit_curr, gross_profit_prior = safe_get_statement_yoy(income_stmt, 'Gross Profit')
        
        current_assets_curr, current_assets_prior = safe_get_statement_yoy(balance_sheet, 'Current Assets')
        ppe_curr, ppe_prior = safe_get_statement_yoy(balance_sheet, 'Net PPE')
        if ppe_curr is None:
            ppe_curr, ppe_prior = safe_get_statement_yoy(balance_sheet, 'Property Plant Equipment Net')
        
        total_assets_curr, total_assets_prior = safe_get_statement_yoy(balance_sheet, 'Total Assets')
        
        depreciation_curr, depreciation_prior = safe_get_statement_yoy(cash_flow, 'Depreciation And Amortization')
        if depreciation_curr is None:
            depreciation_curr, depreciation_prior = safe_get_statement_yoy(income_stmt, 'Depreciation And Amortization')
        
        sga_curr, sga_prior = safe_get_statement_yoy(income_stmt, 'Selling General And Administration')
        
        net_income = safe_get_statement(income_stmt, 'Net Income')
        if net_income is None:
            net_income = safe_get_statement(income_stmt, 'Net Income Common Stockholders')
        
        ocf = safe_get_statement(cash_flow, 'Operating Cash Flow')
        if ocf is None:
            ocf = safe_get_statement(cash_flow, 'Cash Flow From Continuing Operating Activities')
        
        total_liabilities_curr, total_liabilities_prior = safe_get_statement_yoy(balance_sheet, 'Total Liabilities Net Minority Interest')
        
        components = {}
        
        dsri = 1.0
        if all(v is not None and v != 0 for v in [receivables_curr, receivables_prior, revenue_curr, revenue_prior]):
            dsr_curr = receivables_curr / revenue_curr
            dsr_prior = receivables_prior / revenue_prior
            if dsr_prior != 0:
                dsri = dsr_curr / dsr_prior
        components['dsri'] = round(dsri, 4)
        
        gmi = 1.0
        if all(v is not None and v != 0 for v in [gross_profit_curr, gross_profit_prior, revenue_curr, revenue_prior]):
            gm_curr = gross_profit_curr / revenue_curr
            gm_prior = gross_profit_prior / revenue_prior
            if gm_curr != 0:
                gmi = gm_prior / gm_curr
        components['gmi'] = round(gmi, 4)
        
        aqi = 1.0
        if all(v is not None for v in [current_assets_curr, current_assets_prior, ppe_curr, ppe_prior, total_assets_curr, total_assets_prior]):
            if total_assets_curr != 0 and total_assets_prior != 0:
                aq_curr = 1 - ((current_assets_curr + ppe_curr) / total_assets_curr)
                aq_prior = 1 - ((current_assets_prior + ppe_prior) / total_assets_prior)
                if aq_prior != 0:
                    aqi = aq_curr / aq_prior
        components['aqi'] = round(aqi, 4)
        
        sgi = 1.0
        if revenue_curr is not None and revenue_prior is not None and revenue_prior != 0:
            sgi = revenue_curr / revenue_prior
        components['sgi'] = round(sgi, 4)
        
        depi = 1.0
        if all(v is not None and v != 0 for v in [depreciation_curr, depreciation_prior, ppe_curr, ppe_prior]):
            dep_rate_curr = depreciation_curr / (ppe_curr + depreciation_curr)
            dep_rate_prior = depreciation_prior / (ppe_prior + depreciation_prior)
            if dep_rate_curr != 0:
                depi = dep_rate_prior / dep_rate_curr
        components['depi'] = round(depi, 4)
        
        sgai = 1.0
        if all(v is not None and v != 0 for v in [sga_curr, sga_prior, revenue_curr, revenue_prior]):
            sga_ratio_curr = sga_curr / revenue_curr
            sga_ratio_prior = sga_prior / revenue_prior
            if sga_ratio_prior != 0:
                sgai = sga_ratio_curr / sga_ratio_prior
        components['sgai'] = round(sgai, 4)
        
        tata = 0.0
        if all(v is not None for v in [net_income, ocf, total_assets_curr]) and total_assets_curr != 0:
            tata = (net_income - ocf) / total_assets_curr
        components['tata'] = round(tata, 4)
        
        lvgi = 1.0
        if all(v is not None and v != 0 for v in [total_liabilities_curr, total_liabilities_prior, total_assets_curr, total_assets_prior]):
            lev_curr = total_liabilities_curr / total_assets_curr
            lev_prior = total_liabilities_prior / total_assets_prior
            if lev_prior != 0:
                lvgi = lev_curr / lev_prior
        components['lvgi'] = round(lvgi, 4)
        
        m_score = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 
                   0.892 * sgi + 0.115 * depi - 0.172 * sgai + 
                   4.679 * tata - 0.327 * lvgi)
        
        if m_score < -2.22:
            interpretation = "Unlikely Manipulator"
            status = "positive"
        else:
            interpretation = "Possible Manipulator"
            status = "negative"
        
        return {
            'score': round(m_score, 2),
            'display': f"{round(m_score, 2)}",
            'interpretation': interpretation,
            'status': status,
            'components': components
        }
        
    except Exception as e:
        return {
            'score': None,
            'display': 'N/A',
            'interpretation': 'Insufficient Data',
            'status': 'neutral',
            'error': str(e)
        }


def calculate_metrics(data):
    try:
        info = data.get('info', {})
        income_stmt = data.get('income_stmt')
        balance_sheet = data.get('balance_sheet')
        
        metrics = {
            'valuation': [],
            'profitability': [],
            'leverage': []
        }
        
        pe = info.get('trailingPE') or info.get('forwardPE')
        pe_status = 'neutral'
        if pe is not None:
            if pe < 0:
                pe = None
            elif pe < 15:
                pe_status = 'positive'
            elif pe > 25:
                pe_status = 'negative'
        metrics['valuation'].append({
            'name': 'P/E',
            'value': round(pe, 2) if pe else None,
            'display': f"{round(pe, 1)}x" if pe else 'N/A',
            'status': pe_status if pe else 'neutral'
        })
        
        ps = info.get('priceToSalesTrailing12Months')
        ps_status = 'neutral'
        if ps is not None:
            if ps < 2:
                ps_status = 'positive'
            elif ps > 5:
                ps_status = 'negative'
        metrics['valuation'].append({
            'name': 'P/S',
            'value': round(ps, 2) if ps else None,
            'display': f"{round(ps, 1)}x" if ps else 'N/A',
            'status': ps_status if ps else 'neutral'
        })
        
        pb = info.get('priceToBook')
        pb_status = 'neutral'
        if pb is not None:
            if pb < 0:
                pb = None
            elif pb < 1.5:
                pb_status = 'positive'
            elif pb > 3:
                pb_status = 'negative'
        metrics['valuation'].append({
            'name': 'P/B',
            'value': round(pb, 2) if pb else None,
            'display': f"{round(pb, 1)}x" if pb else 'N/A',
            'status': pb_status if pb else 'neutral'
        })
        
        ev_ebitda = info.get('enterpriseToEbitda')
        ev_status = 'neutral'
        if ev_ebitda is not None:
            if ev_ebitda < 0:
                ev_ebitda = None
            elif ev_ebitda < 10:
                ev_status = 'positive'
            elif ev_ebitda > 15:
                ev_status = 'negative'
        metrics['valuation'].append({
            'name': 'EV/EBITDA',
            'value': round(ev_ebitda, 2) if ev_ebitda else None,
            'display': f"{round(ev_ebitda, 1)}x" if ev_ebitda else 'N/A',
            'status': ev_status if ev_ebitda else 'neutral'
        })
        
        gross_margin = info.get('grossMargins')
        if gross_margin:
            gross_margin = gross_margin * 100
        gm_status = 'neutral'
        if gross_margin is not None:
            if gross_margin > 40:
                gm_status = 'positive'
            elif gross_margin < 20:
                gm_status = 'negative'
        metrics['profitability'].append({
            'name': 'Gross Margin',
            'value': round(gross_margin, 2) if gross_margin else None,
            'display': f"{round(gross_margin, 1)}%" if gross_margin else 'N/A',
            'status': gm_status if gross_margin else 'neutral'
        })
        
        op_margin = info.get('operatingMargins')
        if op_margin:
            op_margin = op_margin * 100
        om_status = 'neutral'
        if op_margin is not None:
            if op_margin > 15:
                om_status = 'positive'
            elif op_margin < 5:
                om_status = 'negative'
        metrics['profitability'].append({
            'name': 'Op. Margin',
            'value': round(op_margin, 2) if op_margin else None,
            'display': f"{round(op_margin, 1)}%" if op_margin else 'N/A',
            'status': om_status if op_margin else 'neutral'
        })
        
        net_margin = info.get('profitMargins')
        if net_margin:
            net_margin = net_margin * 100
        nm_status = 'neutral'
        if net_margin is not None:
            if net_margin > 10:
                nm_status = 'positive'
            elif net_margin < 5:
                nm_status = 'negative'
        metrics['profitability'].append({
            'name': 'Net Margin',
            'value': round(net_margin, 2) if net_margin else None,
            'display': f"{round(net_margin, 1)}%" if net_margin else 'N/A',
            'status': nm_status if net_margin else 'neutral'
        })
        
        de = info.get('debtToEquity')
        if de:
            de = de / 100
        de_status = 'neutral'
        if de is not None:
            if de < 0:
                de = None
            elif de < 0.5:
                de_status = 'positive'
            elif de > 1.5:
                de_status = 'negative'
        metrics['leverage'].append({
            'name': 'Debt/Equity',
            'value': round(de, 2) if de else None,
            'display': f"{round(de, 2)}" if de else 'N/A',
            'status': de_status if de else 'neutral'
        })
        
        cr = info.get('currentRatio')
        cr_status = 'neutral'
        if cr is not None:
            if cr > 1.5:
                cr_status = 'positive'
            elif cr < 1:
                cr_status = 'negative'
        metrics['leverage'].append({
            'name': 'Current Ratio',
            'value': round(cr, 2) if cr else None,
            'display': f"{round(cr, 2)}" if cr else 'N/A',
            'status': cr_status if cr else 'neutral'
        })
        
        ebit = safe_get_statement(income_stmt, 'EBIT')
        if ebit is None:
            ebit = safe_get_statement(income_stmt, 'Operating Income')
        interest_expense = safe_get_statement(income_stmt, 'Interest Expense')
        
        ic = None
        ic_status = 'neutral'
        if ebit is not None and interest_expense is not None and interest_expense != 0:
            ic = abs(ebit / interest_expense)
            if ic > 5:
                ic_status = 'positive'
            elif ic < 2:
                ic_status = 'negative'
        elif interest_expense == 0 or interest_expense is None:
            ic_status = 'positive'
        
        metrics['leverage'].append({
            'name': 'Int. Coverage',
            'value': round(ic, 2) if ic else None,
            'display': f"{round(ic, 1)}x" if ic else 'No Debt',
            'status': ic_status
        })
        
        return metrics
        
    except Exception as e:
        return {
            'valuation': [],
            'profitability': [],
            'leverage': [],
            'error': str(e)
        }


def get_company_info(data):
    try:
        info = data.get('info', {})
        
        market_cap = info.get('marketCap')
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
        
        return {
            'name': info.get('longName') or info.get('shortName') or 'Unknown',
            'ticker': info.get('symbol', '').upper(),
            'sector': info.get('sector') or 'N/A',
            'industry': info.get('industry') or 'N/A',
            'price': info.get('currentPrice') or info.get('regularMarketPrice'),
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
            'market_cap': None,
            'market_cap_display': 'N/A',
            'currency': 'USD',
            'error': str(e)
        }


@financials_bp.route('/api/financials', methods=['POST'])
def analyze_financials():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        
        if not ticker:
            return jsonify({
                'error': 'Ticker is required',
                'error_type': 'validation'
            }), 400
        
        stock_data = get_stock_data(ticker)
        
        if 'error' in stock_data:
            return jsonify(stock_data), 400
        
        piotroski = calculate_piotroski_score(stock_data)
        altman = calculate_altman_zscore(stock_data)
        beneish = calculate_beneish_mscore(stock_data)
        
        metrics = calculate_metrics(stock_data)
        
        company = get_company_info(stock_data)
        
        response = {
            'ticker': ticker,
            'company': company,
            'scores': {
                'piotroski': piotroski,
                'altman': altman,
                'beneish': beneish
            },
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'disclaimer': (
                "Financial data is provided for informational purposes only and should not be "
                "considered financial advice. Data is sourced from Yahoo Finance and may be "
                "delayed, incomplete, or inaccurate. Metric color coding represents general "
                "guidelines that vary significantly by industry. Always verify with official "
                "SEC filings and consult a qualified financial advisor before making investment decisions."
            )
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': 'server_error'
        }), 500
