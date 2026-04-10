import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time
import warnings
from scipy.optimize import brentq

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro v4.0 | DCF + Momentum",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
BRAPI_TOKEN = "5gVedSQ928pxhFuTvBFPfr"

# ==============================================================================
# MÓDULO 0: UNIVERSO DCF & PARÂMETROS MACROECONÔMICOS
# ==============================================================================

DCF_UNIVERSE = {
    'EGIE3':  {'name': 'Engie Brasil Energia',    'sector': 'Utilities',   'subsector': 'Geração Energia',   'beta': 0.60},
    'ITUB3':  {'name': 'Itaú Unibanco',           'sector': 'Financial',   'subsector': 'Banco Varejo',      'beta': 1.10},
    'PSSA3':  {'name': 'Porto Seguro',             'sector': 'Insurance',   'subsector': 'Seguros Multi',     'beta': 0.80},
    'WEGE3':  {'name': 'WEG S.A.',                'sector': 'Industrial',  'subsector': 'Máquinas & Equip.', 'beta': 0.95},
    'CXSE3':  {'name': 'Caixa Seguridade',         'sector': 'Insurance',   'subsector': 'Seguros Vida/Prev', 'beta': 0.75},
    'SBSP3':  {'name': 'Sabesp',                  'sector': 'Utilities',   'subsector': 'Saneamento',        'beta': 0.65},
    'TAEE3':  {'name': 'Taesa',                   'sector': 'Utilities',   'subsector': 'T&D Energia',       'beta': 0.55},
    'VIVT3':  {'name': 'Vivo / Telefônica',        'sector': 'Telecom',     'subsector': 'Telecom Varejo',    'beta': 0.72},
    'CPFE3':  {'name': 'CPFL Energia',             'sector': 'Utilities',   'subsector': 'Distribuição',      'beta': 0.60},
    'SAPR3':  {'name': 'Sanepar',                 'sector': 'Utilities',   'subsector': 'Saneamento',        'beta': 0.60},
    'BBAS3':  {'name': 'Banco do Brasil',          'sector': 'Financial',   'subsector': 'Banco Público',     'beta': 1.05},
    'PRIO3':  {'name': 'PRIO S.A.',               'sector': 'Energy',      'subsector': 'E&P Petróleo',      'beta': 1.00},
    'TOTS3':  {'name': 'Totvs',                   'sector': 'Technology',  'subsector': 'ERP / SaaS',        'beta': 1.15},
    'BPAC3':  {'name': 'BTG Pactual',             'sector': 'Financial',   'subsector': 'Banco Investimento', 'beta': 1.25},
    'ALUP3':  {'name': 'Alupar Investimento',      'sector': 'Utilities',   'subsector': 'T&D Energia',       'beta': 0.58},
    'BMOB3':  {'name': 'Bemobi Mobile Tech',       'sector': 'Technology',  'subsector': 'Software/Mobile',   'beta': 1.20},
}

MACRO_PARAMS = {
    'selic':              0.1315,   # SELIC / CDI atual
    'ipca_proj':          0.0450,   # IPCA projetado (FOCUS)
    'igpm_proj':          0.0500,   # IGP-M projetado
    'erp_brazil':         0.0650,   # ERP Brasil (prêmio de risco)
    'country_risk_prem':  0.0200,   # CDS Brasil ≈ 200bps
    'tax_rate':           0.3400,   # IR + CSLL Brasil
    'terminal_g_conserv': 0.0250,   # Perpetuidade conservadora (IPCA - 2pp)
    'terminal_g_moderate':0.0350,   # Perpetuidade moderada (IPCA - 1pp)
}

SECTOR_ASSUMPTIONS = {
    'Utilities': {
        'conserv_rev_growth':  0.0450, 'moderate_rev_growth':  0.0750,
        'conserv_ebitda_adj':  -0.010, 'moderate_ebitda_adj':   0.000,
        'capex_rev_ratio':      0.150,  'da_rev_ratio':          0.080,
        'nwc_rev_ratio':        0.020,  'debt_weight':           0.550,
        'kd_spread':            0.020,  'wacc_premium_conserv':  0.0150,
        'wacc_premium_moderate':0.000,  'ebitda_margin_floor':   0.35,
    },
    'Financial': {
        'conserv_rev_growth':  0.0450, 'moderate_rev_growth':  0.1000,
        'conserv_ebitda_adj':  -0.015, 'moderate_ebitda_adj':   0.000,
        'capex_rev_ratio':      0.020,  'da_rev_ratio':          0.015,
        'nwc_rev_ratio':        0.005,  'debt_weight':           0.650,
        'kd_spread':           -0.010,  'wacc_premium_conserv':  0.0200,
        'wacc_premium_moderate':0.000,  'ebitda_margin_floor':   0.15,
    },
    'Insurance': {
        'conserv_rev_growth':  0.0450, 'moderate_rev_growth':  0.0850,
        'conserv_ebitda_adj':  -0.010, 'moderate_ebitda_adj':   0.000,
        'capex_rev_ratio':      0.020,  'da_rev_ratio':          0.010,
        'nwc_rev_ratio':        0.010,  'debt_weight':           0.400,
        'kd_spread':            0.015,  'wacc_premium_conserv':  0.0150,
        'wacc_premium_moderate':0.000,  'ebitda_margin_floor':   0.12,
    },
    'Industrial': {
        'conserv_rev_growth':  0.0450, 'moderate_rev_growth':  0.1200,
        'conserv_ebitda_adj':  -0.015, 'moderate_ebitda_adj':   0.000,
        'capex_rev_ratio':      0.050,  'da_rev_ratio':          0.040,
        'nwc_rev_ratio':        0.035,  'debt_weight':           0.250,
        'kd_spread':            0.025,  'wacc_premium_conserv':  0.0200,
        'wacc_premium_moderate':0.000,  'ebitda_margin_floor':   0.15,
    },
    'Technology': {
        'conserv_rev_growth':  0.0600, 'moderate_rev_growth':  0.1500,
        'conserv_ebitda_adj':  -0.020, 'moderate_ebitda_adj':   0.005,
        'capex_rev_ratio':      0.030,  'da_rev_ratio':          0.050,
        'nwc_rev_ratio':        0.025,  'debt_weight':           0.150,
        'kd_spread':            0.030,  'wacc_premium_conserv':  0.0250,
        'wacc_premium_moderate':0.000,  'ebitda_margin_floor':   0.10,
    },
    'Telecom': {
        'conserv_rev_growth':  0.0450, 'moderate_rev_growth':  0.0650,
        'conserv_ebitda_adj':  -0.010, 'moderate_ebitda_adj':   0.000,
        'capex_rev_ratio':      0.180,  'da_rev_ratio':          0.120,
        'nwc_rev_ratio':        0.020,  'debt_weight':           0.450,
        'kd_spread':            0.020,  'wacc_premium_conserv':  0.0150,
        'wacc_premium_moderate':0.000,  'ebitda_margin_floor':   0.30,
    },
    'Energy': {
        'conserv_rev_growth':  0.0500, 'moderate_rev_growth':  0.1200,
        'conserv_ebitda_adj':  -0.020, 'moderate_ebitda_adj':   0.000,
        'capex_rev_ratio':      0.250,  'da_rev_ratio':          0.100,
        'nwc_rev_ratio':        0.030,  'debt_weight':           0.400,
        'kd_spread':            0.025,  'wacc_premium_conserv':  0.0200,
        'wacc_premium_moderate':0.000,  'ebitda_margin_floor':   0.35,
    },
}

# ==============================================================================
# MÓDULO 1: DATA FETCHING (HÍBRIDO: BRAPI + YFINANCE FALLBACK)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados via YFinance."""
    t_list = list(tickers)
    for bench in ['BOVA11.SA', 'DIVO11.SA']:
        if bench not in t_list:
            t_list.append(bench)

    try:
        data = yf.download(
            t_list,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=True
        )['Adj Close']

        if isinstance(data, pd.Series):
            data = data.to_frame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna(axis=1, how='all')
        return data
    except Exception as e:
        st.error(f"Erro crítico ao baixar preços YF: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600*4)
def fetch_fundamentals_hybrid(tickers: list, token: str) -> pd.DataFrame:
    """
    Busca fundamentos. Tenta Brapi primeiro.
    Se faltar dados (P/VP, ROE, etc.), preenche com YFinance (.info).
    """
    clean_tickers = [t.replace('.SA', '') for t in tickers if 'BOVA11' not in t and 'DIVO11' not in t]

    if not clean_tickers:
        return pd.DataFrame()

    fundamental_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_tickers = len(clean_tickers)

    def safe_float(val):
        if val is None or val == '' or str(val).lower() == 'nan': return np.nan
        try:
            return float(val)
        except:
            return np.nan

    def get_nested_val(item, keys_list):
        for key in keys_list:
            if key in item and item[key] is not None:
                return item[key]
        for module in ['defaultKeyStatistics', 'financialData', 'summaryProfile', 'price']:
            if module in item and isinstance(item[module], dict):
                for key in keys_list:
                    if key in item[module]:
                        return item[module][key]
        return None

    for i, ticker in enumerate(clean_tickers):
        status_text.text(f"Analisando: {ticker} ({i+1}/{total_tickers}) - Fonte: Brapi...")

        price = market_cap = pe_ratio = p_vp = ev_ebitda = roe = net_margin = np.nan
        sector = 'Outros'

        url = f"https://brapi.dev/api/quote/{ticker}"
        params = {'token': token, 'fundamental': 'true'}

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data_json = response.json()
                results = data_json.get('results', [])
                if results:
                    item = results[0]
                    price = safe_float(item.get('regularMarketPrice'))
                    market_cap = safe_float(item.get('marketCap'))
                    sector = item.get('sector') or item.get('summaryProfile', {}).get('sector', 'Outros')
                    pe_ratio = safe_float(get_nested_val(item, ['priceEarnings', 'trailingPE', 'peRatio']))
                    p_vp = safe_float(get_nested_val(item, ['priceToBook', 'priceToBookRatio', 'p_vp']))
                    ev_ebitda = safe_float(get_nested_val(item, ['enterpriseToEbitda', 'enterpriseValueToEBITDA', 'ev_ebitda']))
                    roe = safe_float(get_nested_val(item, ['returnOnEquity', 'roe']))
                    net_margin = safe_float(get_nested_val(item, ['profitMargin', 'netMargin', 'netProfitMargin']))
        except Exception:
            pass

        if np.isnan(p_vp) or np.isnan(roe) or np.isnan(ev_ebitda):
            status_text.text(f"Complementando dados: {ticker} via Yahoo Finance...")
            try:
                yf_t = yf.Ticker(f"{ticker}.SA")
                info = yf_t.info
                if np.isnan(price): price = info.get('currentPrice') or info.get('previousClose')
                if np.isnan(market_cap): market_cap = info.get('marketCap')
                if sector == 'Outros': sector = info.get('sector', 'Outros')
                if np.isnan(pe_ratio): pe_ratio = info.get('trailingPE')
                if np.isnan(p_vp): p_vp = info.get('priceToBook')
                if np.isnan(ev_ebitda): ev_ebitda = info.get('enterpriseToEbitda')
                if np.isnan(roe): roe = info.get('returnOnEquity')
                if np.isnan(net_margin): net_margin = info.get('profitMargins')
            except Exception:
                pass

        fundamental_data.append({
            'ticker': f"{ticker}.SA",
            'sector': sector,
            'currentPrice': price,
            'marketCap': market_cap,
            'PE': pe_ratio,
            'P_VP': p_vp,
            'EV_EBITDA': ev_ebitda,
            'ROE': roe,
            'Net_Margin': net_margin,
        })

        progress_bar.progress((i + 1) / total_tickers)
        time.sleep(0.5)

    progress_bar.empty()
    status_text.empty()

    df = pd.DataFrame(fundamental_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['ticker'])
        df = df.set_index('ticker')
        cols_check = ['PE', 'P_VP', 'EV_EBITDA', 'ROE', 'Net_Margin']
        for col in cols_check:
            if col in df.columns:
                df[col] = df[col].replace([0, 0.0], np.nan)

    return df


# ==============================================================================
# MÓDULO 6: DCF DATA FETCHING
# ==============================================================================

@st.cache_data(ttl=3600*6)
def fetch_dcf_financials(ticker_clean: str) -> dict:
    """
    Busca dados financeiros para o DCF via YFinance (TTM).
    Retorna métricas-chave normalizadas.
    """
    result = {
        'ticker': ticker_clean,
        'revenue': np.nan, 'ebit': np.nan, 'ebitda': np.nan,
        'da': np.nan, 'capex': np.nan, 'net_debt': np.nan,
        'shares': np.nan, 'current_price': np.nan,
        'market_cap': np.nan, 'net_income': np.nan,
        'op_cash_flow': np.nan, 'bvps': np.nan,
        'dividends_paid': np.nan, 'total_debt': np.nan,
        'cash': np.nan, 'ebitda_margin': np.nan,
    }

    def safe(v):
        try:
            f = float(v)
            return f if not np.isnan(f) and not np.isinf(f) else np.nan
        except:
            return np.nan

    def get_row(df, keys):
        for k in keys:
            if k in df.index:
                val = df.loc[k]
                if isinstance(val, pd.Series):
                    val = val.dropna()
                    return val.iloc[0] if not val.empty else np.nan
                return val
        return np.nan

    try:
        ticker_sa = f"{ticker_clean}.SA"
        t = yf.Ticker(ticker_sa)
        info = t.info

        # Básico via info
        result['current_price'] = safe(info.get('currentPrice') or info.get('regularMarketPrice'))
        result['shares']        = safe(info.get('sharesOutstanding'))
        result['market_cap']    = safe(info.get('marketCap'))
        result['bvps']          = safe(info.get('bookValue'))
        result['revenue']       = safe(info.get('totalRevenue'))
        result['ebitda']        = safe(info.get('ebitda'))
        result['net_debt']      = safe(info.get('totalDebt', np.nan))

        # Demonstrativos trimestrais para cálculo TTM
        try:
            qfin = t.quarterly_financials
            qcf  = t.quarterly_cashflow
            qbs  = t.quarterly_balance_sheet

            # ---- Income Statement TTM ----
            if not qfin.empty and qfin.shape[1] >= 4:
                f4 = qfin.iloc[:, :4]
                rev = get_row(f4, ['Total Revenue', 'Revenue'])
                if not np.isnan(safe(rev)):
                    result['revenue'] = safe(f4.loc[f4.index[f4.index.isin(['Total Revenue', 'Revenue'])][0]].sum())

                ebit_ttm = get_row(f4, ['EBIT', 'Operating Income', 'Operating Revenue'])
                if not np.isnan(safe(ebit_ttm)):
                    key = [k for k in ['EBIT', 'Operating Income'] if k in f4.index]
                    if key:
                        result['ebit'] = safe(f4.loc[key[0]].sum())

                ni_ttm = get_row(f4, ['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operations'])
                if not np.isnan(safe(ni_ttm)):
                    key = [k for k in ['Net Income', 'Net Income Common Stockholders'] if k in f4.index]
                    if key:
                        result['net_income'] = safe(f4.loc[key[0]].sum())

            # ---- Cash Flow TTM ----
            if not qcf.empty and qcf.shape[1] >= 4:
                cf4 = qcf.iloc[:, :4]
                da_key = [k for k in ['Depreciation And Amortization', 'Depreciation', 'Depreciation Amortization Depletion'] if k in cf4.index]
                if da_key:
                    result['da'] = abs(safe(cf4.loc[da_key[0]].sum()))

                capex_key = [k for k in ['Capital Expenditure', 'Purchase Of PPE', 'Capital Expenditures', 'Purchases Of Property Plant And Equipment'] if k in cf4.index]
                if capex_key:
                    result['capex'] = abs(safe(cf4.loc[capex_key[0]].sum()))

                ocf_key = [k for k in ['Operating Cash Flow', 'Cash Flows From Operations Used', 'Net Cash Provided By Operating Activities'] if k in cf4.index]
                if ocf_key:
                    result['op_cash_flow'] = safe(cf4.loc[ocf_key[0]].sum())

                div_key = [k for k in ['Dividends Paid', 'Common Stock Dividend Paid', 'Payment Of Dividends'] if k in cf4.index]
                if div_key:
                    result['dividends_paid'] = abs(safe(cf4.loc[div_key[0]].sum()))

            # ---- Balance Sheet (último quarter) ----
            if not qbs.empty:
                bs_l = qbs.iloc[:, 0]

                td_key = [k for k in ['Total Debt', 'Long Term Debt', 'Total Long Term Debt'] if k in bs_l.index]
                if td_key:
                    result['total_debt'] = safe(bs_l[td_key[0]])

                cash_key = [k for k in ['Cash And Cash Equivalents', 'Cash', 'Cash Cash Equivalents And Short Term Investments'] if k in bs_l.index]
                if cash_key:
                    result['cash'] = safe(bs_l[cash_key[0]])

                if not np.isnan(result['total_debt']) and not np.isnan(result['cash']):
                    result['net_debt'] = result['total_debt'] - result['cash']
                elif not np.isnan(result['total_debt']):
                    result['net_debt'] = result['total_debt']

        except Exception:
            pass

        # Inferências derivadas
        if np.isnan(result['ebitda']) and not np.isnan(result['ebit']) and not np.isnan(result['da']):
            result['ebitda'] = result['ebit'] + result['da']
        if np.isnan(result['ebit']) and not np.isnan(result['ebitda']) and not np.isnan(result['da']):
            result['ebit'] = result['ebitda'] - result['da']
        if np.isnan(result['net_debt']):
            result['net_debt'] = 0.0  # Conservador: assume dívida zero se desconhecida

        if not np.isnan(result['ebitda']) and not np.isnan(result['revenue']) and result['revenue'] > 0:
            result['ebitda_margin'] = result['ebitda'] / result['revenue']

    except Exception:
        pass

    return result


# ==============================================================================
# MÓDULO 7: DCF ENGINE — SCENARIOS & VALUATION
# ==============================================================================

def _compute_wacc(sector: str, beta: float, scenario: str,
                  custom_wacc: float = None) -> tuple:
    """Calcula Ke, Kd e WACC para o cenário."""
    sp = SECTOR_ASSUMPTIONS.get(sector, SECTOR_ASSUMPTIONS['Utilities'])
    macro = MACRO_PARAMS

    rf  = macro['selic']
    erp = macro['erp_brazil']
    tax = macro['tax_rate']

    premium = sp['wacc_premium_conserv'] if scenario == 'conservative' else sp['wacc_premium_moderate']
    ke = rf + beta * erp + premium

    kd = rf + sp['kd_spread']
    kd = max(kd, 0.06)

    dw = sp['debt_weight']
    ew = 1 - dw

    if custom_wacc is not None:
        wacc = custom_wacc
    else:
        wacc = ew * ke + dw * kd * (1 - tax)

    return ke, kd, wacc


def _project_fcff(revenue: float, ebitda_margin: float, scenario: str,
                  sector: str, projection_years: int = 5,
                  custom_g_terminal: float = None) -> tuple:
    """
    Projeta FCFFs e retorna (fcff_list, ebitda_proj_list, revenue_proj_list, terminal_g).
    """
    sp   = SECTOR_ASSUMPTIONS.get(sector, SECTOR_ASSUMPTIONS['Utilities'])
    macro = MACRO_PARAMS
    tax  = macro['tax_rate']

    growth    = sp['conserv_rev_growth'] if scenario == 'conservative' else sp['moderate_rev_growth']
    margin_adj = sp['conserv_ebitda_adj'] if scenario == 'conservative' else sp['moderate_ebitda_adj']
    g         = custom_g_terminal if custom_g_terminal is not None else (
        macro['terminal_g_conserv'] if scenario == 'conservative' else macro['terminal_g_moderate']
    )
    margin_floor = sp['ebitda_margin_floor']

    fcff_list    = []
    rev_list     = []
    ebitda_list  = []
    rev_curr     = revenue
    m_curr       = ebitda_margin

    for yr in range(1, projection_years + 1):
        rev_curr *= (1 + growth)
        m_curr = max(m_curr + margin_adj / projection_years, margin_floor)

        ebitda_yr  = rev_curr * m_curr
        da_yr      = rev_curr * sp['da_rev_ratio']
        ebit_yr    = ebitda_yr - da_yr
        nopat_yr   = ebit_yr * (1 - tax)
        capex_yr   = rev_curr * sp['capex_rev_ratio']
        delta_nwc  = rev_curr * growth * sp['nwc_rev_ratio']

        fcff = nopat_yr + da_yr - capex_yr - delta_nwc
        fcff_list.append(fcff)
        rev_list.append(rev_curr)
        ebitda_list.append(ebitda_yr)

    return fcff_list, ebitda_list, rev_list, g


def _pv_fcff_tv(fcff_list: list, wacc: float, g: float,
                net_debt: float, shares: float) -> dict:
    """
    Desconta FCFFs, calcula TV (Gordon) e retorna preço justo.
    """
    n = len(fcff_list)
    pv_fcffs = sum(fcff_list[t] / (1 + wacc) ** (t + 1) for t in range(n))

    if wacc <= g:
        return {'error': f'WACC ({wacc:.2%}) ≤ g ({g:.2%})'}

    fcff_terminal = fcff_list[-1] * (1 + g)
    tv = fcff_terminal / (wacc - g)
    pv_tv = tv / (1 + wacc) ** n

    ev = pv_fcffs + pv_tv
    equity_val = ev - net_debt
    fair_price = equity_val / shares if shares > 0 else np.nan
    tv_pct = pv_tv / ev if ev > 0 else np.nan

    return {
        'pv_fcffs': pv_fcffs, 'pv_terminal': pv_tv,
        'terminal_value': tv, 'enterprise_value': ev,
        'equity_value': equity_val, 'fair_price': fair_price,
        'tv_pct': tv_pct, 'error': None,
    }


def _compute_irr(current_price: float, shares: float, net_debt: float,
                 fcff_list: list, terminal_value: float) -> float:
    """Calcula a TIR implícita dado o preço atual de mercado."""
    try:
        market_ev = current_price * shares + net_debt
        cash_flows = [-market_ev] + list(fcff_list) + [terminal_value]

        def npv_fn(r):
            return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))

        irr = brentq(npv_fn, -0.50, 10.0, maxiter=500, xtol=1e-8)
        return irr
    except Exception:
        return np.nan


def run_dcf_model(ticker: str, fin_data: dict, scenario: str = 'moderate',
                  projection_years: int = 5, custom_wacc: float = None,
                  custom_g: float = None) -> dict:
    """
    Motor principal de DCF.
    Retorna dicionário completo com premissas, projeções e outputs.
    """
    co = DCF_UNIVERSE.get(ticker, {})
    sector = co.get('sector', 'Utilities')
    beta   = co.get('beta', 1.0)

    # Validação de dados mínimos
    revenue       = fin_data.get('revenue', np.nan)
    current_price = fin_data.get('current_price', np.nan)
    shares        = fin_data.get('shares', np.nan)

    if any(np.isnan(v) for v in [revenue, current_price, shares]) or revenue <= 0:
        return {'error': 'Dados financeiros insuficientes (receita/preço/ações)', 'ticker': ticker, 'scenario': scenario}

    macro = MACRO_PARAMS
    sp    = SECTOR_ASSUMPTIONS.get(sector, SECTOR_ASSUMPTIONS['Utilities'])

    # Margens base (com fallback setorial)
    raw_ebitda_margin = fin_data.get('ebitda_margin', np.nan)
    if np.isnan(raw_ebitda_margin):
        fallback_margins = {
            'Utilities': 0.40, 'Financial': 0.30, 'Insurance': 0.20,
            'Industrial': 0.22, 'Technology': 0.25, 'Telecom': 0.38, 'Energy': 0.50,
        }
        raw_ebitda_margin = fallback_margins.get(sector, 0.25)
    ebitda_margin = max(raw_ebitda_margin, sp['ebitda_margin_floor'])

    net_debt = fin_data.get('net_debt', 0.0)
    if np.isnan(net_debt):
        net_debt = 0.0

    # Custo de capital
    ke, kd, wacc = _compute_wacc(sector, beta, scenario, custom_wacc)

    # Projeção de FCFFs
    fcff_list, ebitda_proj, rev_proj, g = _project_fcff(
        revenue, ebitda_margin, scenario, sector, projection_years, custom_g
    )

    # Desconto e preço justo
    pv_result = _pv_fcff_tv(fcff_list, wacc, g, net_debt, shares)
    if pv_result.get('error'):
        return {**pv_result, 'ticker': ticker, 'scenario': scenario}

    fair_price = pv_result['fair_price']

    # Margem de segurança
    mos = (fair_price - current_price) / fair_price if fair_price and fair_price > 0 else np.nan

    # TIR implícita
    irr = _compute_irr(current_price, shares, net_debt, fcff_list, pv_result['terminal_value'])

    growth_used = sp['conserv_rev_growth'] if scenario == 'conservative' else sp['moderate_rev_growth']

    return {
        'error': None,
        'ticker': ticker,
        'name': co.get('name', ticker),
        'sector': sector,
        'subsector': co.get('subsector', ''),
        'scenario': scenario,
        # Preços
        'current_price': current_price,
        'fair_price': fair_price,
        'margin_of_safety': mos,
        # Capital
        'wacc': wacc, 'ke': ke, 'kd': kd,
        'beta': beta,
        # Perpetuidade
        'terminal_g': g,
        'tv_pct_ev': pv_result['tv_pct'],
        # Valores
        'enterprise_value': pv_result['enterprise_value'],
        'equity_value': pv_result['equity_value'],
        'net_debt': net_debt,
        'pv_fcffs': pv_result['pv_fcffs'],
        'pv_terminal': pv_result['pv_terminal'],
        # Projeções
        'fcff_projection': fcff_list,
        'revenue_projection': rev_proj,
        'ebitda_projection': ebitda_proj,
        # Base financeira
        'revenue_ttm': revenue,
        'ebitda_margin_base': ebitda_margin,
        # Métricas
        'irr': irr,
        'rev_growth_used': growth_used,
    }


# ==============================================================================
# MÓDULO 8: ANÁLISE DE SENSIBILIDADE
# ==============================================================================

def build_sensitivity_table(ticker: str, fin_data: dict, base_result: dict,
                             scenario: str) -> pd.DataFrame:
    """
    Tabela de sensibilidade: Preço Justo × WACC × Taxa de Crescimento na Perpetuidade.
    """
    if not base_result or base_result.get('error'):
        return pd.DataFrame()

    base_wacc = base_result['wacc']
    base_g    = base_result['terminal_g']

    wacc_range = np.round(np.arange(base_wacc - 0.02, base_wacc + 0.025, 0.005), 4)
    g_range    = np.round(np.arange(base_g - 0.010,   base_g + 0.015,  0.005), 4)

    co     = DCF_UNIVERSE.get(ticker, {})
    sector = co.get('sector', 'Utilities')
    macro  = MACRO_PARAMS
    sp     = SECTOR_ASSUMPTIONS.get(sector, SECTOR_ASSUMPTIONS['Utilities'])
    tax    = macro['tax_rate']

    revenue       = fin_data.get('revenue', np.nan)
    net_debt      = fin_data.get('net_debt', 0.0) or 0.0
    shares        = fin_data.get('shares', np.nan)
    ebitda_margin = base_result.get('ebitda_margin_base', 0.25)

    if np.isnan(revenue) or np.isnan(shares):
        return pd.DataFrame()

    rows, row_labels = [], []
    for w in wacc_range:
        row_labels.append(f"{w:.2%}")
        row = []
        for g in g_range:
            if w <= g:
                row.append(np.nan)
                continue
            fcff_list, _, _, _ = _project_fcff(revenue, ebitda_margin, scenario, sector,
                                                projection_years=5, custom_g_terminal=g)
            pv = _pv_fcff_tv(fcff_list, w, g, net_debt, shares)
            row.append(round(pv['fair_price'], 2) if not pv.get('error') else np.nan)
        rows.append(row)

    col_labels = [f"{g:.2%}" for g in g_range]
    df = pd.DataFrame(rows, index=row_labels, columns=col_labels)
    df.index.name   = 'WACC ↓ / g →'
    return df


# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (ORIGINAL)
# ==============================================================================

def compute_residual_momentum_enhanced(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Residual Momentum (Blitz) com Volatility Scaling."""
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()

    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
    market = rets['BOVA11.SA']
    scores = {}
    regression_window = 36

    for ticker in rets.columns:
        if ticker in ['BOVA11.SA', 'DIVO11.SA']: continue
        y_full = rets[ticker].tail(regression_window + skip)
        x_full = market.tail(regression_window + skip)
        if len(y_full) < 12: continue
        try:
            common_idx = y_full.index.intersection(x_full.index)
            y_full = y_full.loc[common_idx]
            x_full = x_full.loc[common_idx]
            X = sm.add_constant(x_full.values)
            model = sm.OLS(y_full.values, X).fit()
            residuals = pd.Series(model.resid, index=y_full.index)
            resid_12m = residuals.iloc[-(12 + skip): -skip]
            if len(resid_12m) == 0:
                scores[ticker] = 0
                continue
            raw_momentum = resid_12m.sum()
            resid_vol = residuals.std()
            scores[ticker] = 0 if resid_vol == 0 else raw_momentum / resid_vol
        except:
            scores[ticker] = 0

    return pd.Series(scores, name='Residual_Momentum')


def compute_value_robust(fund_df: pd.DataFrame) -> pd.Series:
    """Composite Value Score."""
    scores = pd.DataFrame(index=fund_df.index)

    def invert_metric(series):
        return 1.0 / series.replace(0, np.nan)

    if 'PE' in fund_df: scores['Earnings_Yield'] = invert_metric(fund_df['PE'])
    if 'P_VP' in fund_df: scores['Book_Yield']    = invert_metric(fund_df['P_VP'])
    if 'EV_EBITDA' in fund_df: scores['EBITDA_Yield'] = invert_metric(fund_df['EV_EBITDA'])

    if scores.empty or scores.dropna(how='all').empty:
        return pd.Series(0, index=fund_df.index, name="Value_Score")

    for col in scores.columns:
        filled = scores[col].fillna(scores[col].median())
        scores[col] = (filled - filled.mean()) / filled.std() if filled.std() > 0 else 0

    return scores.mean(axis=1).rename("Value_Score")


def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Composite Quality Score."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'ROE' in fund_df: scores['ROE']    = fund_df['ROE']
    if 'Net_Margin' in fund_df: scores['Margin'] = fund_df['Net_Margin']

    if scores.empty or scores.dropna(how='all').empty:
        return pd.Series(0, index=fund_df.index, name="Quality_Score")

    for col in scores.columns:
        filled = scores[col].fillna(scores[col].median())
        scores[col] = (filled - filled.mean()) / filled.std() if filled.std() > 0 else 0

    return scores.mean(axis=1).rename("Quality_Score")


# ==============================================================================
# MÓDULO 3: MATEMÁTICA E MÉTRICAS (ORIGINAL)
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6: return series - median
    z = (series - median) / (mad * 1.4826)
    return z.clip(-3, 3)


def calculate_advanced_metrics(prices_series: pd.Series, risk_free_rate_annual: float = 0.10):
    if prices_series.empty or len(prices_series) < 2:
        return {}
    daily_rets = prices_series.pct_change().dropna()
    if daily_rets.empty: return {}

    total_ret = (prices_series.iloc[-1] / prices_series.iloc[0]) - 1
    days = (prices_series.index[-1] - prices_series.index[0]).days
    cagr = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
    vol_ann = daily_rets.std() * np.sqrt(252)
    rf_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    excess_rets = daily_rets - rf_daily
    sharpe = (excess_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    downside_rets = excess_rets[excess_rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    sortino = (excess_rets.mean() * 252) / downside_std if (downside_std > 0 and not np.isnan(downside_std)) else 0
    cum_rets = (1 + daily_rets).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    ulcer_index = np.sqrt((drawdown ** 2).mean())
    return {
        'Retorno Total': total_ret, 'CAGR': cagr, 'Volatilidade': vol_ann,
        'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar,
        'Max Drawdown': max_dd, 'Ulcer Index': ulcer_index
    }


# ==============================================================================
# MÓDULO 4: SIMULAÇÃO MONTE CARLO (ORIGINAL)
# ==============================================================================

def run_monte_carlo(initial_balance, monthly_contrib, mu_annual, sigma_annual, years, simulations=1000):
    if np.isnan(mu_annual) or np.isnan(sigma_annual):
        return pd.DataFrame()
    months = int(years * 12)
    dt = 1 / 12
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    if sigma_annual == 0: sigma_annual = 0.01
    shock = sigma_annual * np.sqrt(dt) * np.random.normal(0, 1, (months, simulations))
    monthly_returns = np.exp(drift + shock) - 1
    portfolio_paths = np.zeros((months + 1, simulations))
    portfolio_paths[0] = initial_balance
    for t in range(1, months + 1):
        portfolio_paths[t] = portfolio_paths[t - 1] * (1 + monthly_returns[t - 1]) + monthly_contrib
    percentiles = np.percentile(portfolio_paths, [5, 50, 95], axis=1)
    dates = [datetime.now() + timedelta(days=30 * i) for i in range(months + 1)]
    return pd.DataFrame({
        'Pessimista (5%)': percentiles[0],
        'Base (50%)': percentiles[1],
        'Otimista (95%)': percentiles[2]
    }, index=dates)


# ==============================================================================
# MÓDULO 5: BACKTEST & ENGINE (ORIGINAL)
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: bool = False):
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()
    if vol_target:
        valid_sel = [s for s in selected if s in prices.columns]
        if not valid_sel: return pd.Series()
        recent_rets = prices[valid_sel].pct_change().tail(63)
        vols = recent_rets.std() * (252 ** 0.5)
        vols = vols.replace(0, 1e-6)
        raw_weights_inv = 1 / vols
        weights = raw_weights_inv / raw_weights_inv.sum() if raw_weights_inv.sum() != 0 else pd.Series(1.0 / len(valid_sel), index=valid_sel)
    else:
        weights = pd.Series(1.0 / len(selected), index=selected)
    return weights.sort_values(ascending=False)


def run_dca_backtest_robust(all_prices: pd.DataFrame, top_n: int, dca_amount: float,
                            use_vol_target: bool, start_date: datetime, end_date: datetime):
    """Backtest Robusto."""
    all_prices = all_prices.ffill()
    dca_start = start_date + timedelta(days=30)
    market_calendar = pd.Series(all_prices.index, index=all_prices.index)
    dates_series = market_calendar.loc[dca_start:end_date].resample('MS').first()
    dates = dates_series.dropna().tolist()

    if not dates or len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    portfolio_value   = pd.Series(0.0, index=all_prices.index)
    portfolio_holdings = {}
    monthly_transactions = []
    cash = 0.0

    for i, month_start in enumerate(dates):
        eval_date = month_start - timedelta(days=1)
        mom_start = month_start - timedelta(days=365 * 3)
        prices_historical = all_prices.loc[:eval_date]
        prices_window = prices_historical.loc[mom_start:]
        if prices_window.empty: continue

        res_mom = compute_residual_momentum_enhanced(prices_window, lookback=12, skip=1)
        if res_mom.empty: continue

        df_rank = pd.DataFrame(index=res_mom.index)
        df_rank['Score'] = robust_zscore(res_mom)
        df_rank = df_rank.sort_values('Score', ascending=False)

        risk_window = prices_historical.tail(90)
        target_weights = construct_portfolio(df_rank, risk_window, top_n, use_vol_target)

        try:
            if month_start not in all_prices.index:
                next_days = all_prices.index[all_prices.index > month_start]
                if next_days.empty: break
                exec_date = next_days[0]
            else:
                exec_date = month_start

            current_date_prices = all_prices.loc[exec_date]
        except KeyError:
            continue

        current_portfolio_val_mtm = cash
        for t, qtd in portfolio_holdings.items():
            if t in current_date_prices and not np.isnan(current_date_prices[t]):
                current_portfolio_val_mtm += qtd * current_date_prices[t]

        total_portfolio_val = current_portfolio_val_mtm + dca_amount
        new_holdings = {}

        for ticker, weight in target_weights.items():
            if ticker in current_date_prices and not np.isnan(current_date_prices[ticker]):
                price = current_date_prices[ticker]
                if price > 0:
                    alloc_val = total_portfolio_val * weight
                    qty = alloc_val / price
                    new_holdings[ticker] = qty
                    monthly_transactions.append({
                        'Date': exec_date, 'Ticker': ticker,
                        'Action': 'Rebalance/Buy', 'Price': price, 'Weight': weight
                    })

        portfolio_holdings = new_holdings
        next_rebalance = dates[i + 1] if i < len(dates) - 1 else end_date
        valid_end = min(next_rebalance, all_prices.index[-1])
        if exec_date > valid_end: continue

        valuation_dates = all_prices.loc[exec_date:valid_end].index
        for d in valuation_dates:
            val = 0
            for t, q in portfolio_holdings.items():
                p = all_prices.at[d, t]
                if not np.isnan(p):
                    val += q * p
            portfolio_value[d] = val

    portfolio_value = portfolio_value[portfolio_value > 0].sort_index()
    equity_curve = pd.DataFrame({'Strategy_DCA': portfolio_value})
    transactions_df = pd.DataFrame(monthly_transactions)
    final_holdings = portfolio_holdings

    return equity_curve, transactions_df, final_holdings


def run_benchmark_dca(price_series: pd.Series, dates: list, dca_amount: float):
    """Simula DCA Benchmark."""
    if price_series.empty: return pd.Series()
    price_series = price_series.dropna()
    df_flow = pd.DataFrame(index=price_series.index)
    df_flow['Price'] = price_series
    df_flow['Add_Units'] = 0.0

    for d in sorted(dates):
        idx_loc = price_series.index.asof(d)
        if idx_loc is not None:
            price = price_series.loc[idx_loc]
            if price > 0:
                buy_units = dca_amount / price
                if idx_loc in df_flow.index:
                    df_flow.at[idx_loc, 'Add_Units'] = buy_units

    df_flow['Cumulative_Units'] = df_flow['Add_Units'].cumsum()
    equity_curve = df_flow['Cumulative_Units'] * df_flow['Price']
    return equity_curve[equity_curve > 0]


# ==============================================================================
# MÓDULO 9: UI — RENDER VALUATION TAB
# ==============================================================================

def render_valuation_summary(results_conserv: dict, results_moderate: dict,
                             fin_data: dict, ticker: str):
    """Renderiza painel completo de valuation para um ticker."""
    co = DCF_UNIVERSE.get(ticker, {})
    st.markdown(f"#### {co.get('name', ticker)} — `{ticker}` | {co.get('sector','')} › {co.get('subsector','')}")

    err_c = results_conserv.get('error')
    err_m = results_moderate.get('error')

    if err_c and err_m:
        st.error(f"Sem dados suficientes: {err_m}")
        return

    # ── KPI Row ──────────────────────────────────────────────────────────────
    cp  = results_moderate.get('current_price', np.nan)
    fp_c = results_conserv.get('fair_price', np.nan)
    fp_m = results_moderate.get('fair_price', np.nan)
    mos_m = results_moderate.get('margin_of_safety', np.nan)
    irr_m = results_moderate.get('irr', np.nan)

    upside_c = (fp_c / cp - 1) if (fp_c and cp and cp > 0) else np.nan
    upside_m = (fp_m / cp - 1) if (fp_m and cp and cp > 0) else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Preço Atual",       f"R$ {cp:.2f}" if not np.isnan(cp) else "N/D")
    k2.metric("Preço Justo Conser.", f"R$ {fp_c:.2f}" if not np.isnan(fp_c) else "N/D",
              delta=f"{upside_c:.1%}" if not np.isnan(upside_c) else None)
    k3.metric("Preço Justo Moder.", f"R$ {fp_m:.2f}" if not np.isnan(fp_m) else "N/D",
              delta=f"{upside_m:.1%}" if not np.isnan(upside_m) else None)
    k4.metric("Margem de Seg. (M)", f"{mos_m:.1%}" if not np.isnan(mos_m) else "N/D")
    k5.metric("TIR Implícita (M)", f"{irr_m:.1%}" if not np.isnan(irr_m) else "N/D")

    st.markdown("---")
    col_l, col_r = st.columns([1, 1])

    # ── Parâmetros & Premissas ────────────────────────────────────────────────
    with col_l:
        st.markdown("**📋 Premissas do Modelo**")
        params_data = {
            'Parâmetro': ['SELIC / Rf', 'IPCA Projetado', 'ERP Brasil', 'Beta (setorial)',
                          'WACC Conserv.', 'WACC Moderado', 'g Perpetuidade Conserv.',
                          'g Perpetuidade Moder.', 'Crescimento Receita Conserv.',
                          'Crescimento Receita Moder.', 'Margem EBITDA Base',
                          'Dívida Líquida (R$ bi)'],
            'Valor': [
                f"{MACRO_PARAMS['selic']:.2%}",
                f"{MACRO_PARAMS['ipca_proj']:.2%}",
                f"{MACRO_PARAMS['erp_brazil']:.2%}",
                f"{co.get('beta', 1.0):.2f}",
                f"{results_conserv.get('wacc', np.nan):.2%}" if not err_c else "—",
                f"{results_moderate.get('wacc', np.nan):.2%}" if not err_m else "—",
                f"{results_conserv.get('terminal_g', np.nan):.2%}" if not err_c else "—",
                f"{results_moderate.get('terminal_g', np.nan):.2%}" if not err_m else "—",
                f"{results_conserv.get('rev_growth_used', np.nan):.2%}" if not err_c else "—",
                f"{results_moderate.get('rev_growth_used', np.nan):.2%}" if not err_m else "—",
                f"{results_moderate.get('ebitda_margin_base', np.nan):.2%}" if not err_m else "—",
                f"{fin_data.get('net_debt', 0) / 1e9:.2f}" if fin_data.get('net_debt') else "0.00",
            ]
        }
        st.dataframe(pd.DataFrame(params_data).set_index('Parâmetro'),
                     use_container_width=True, height=420)

    # ── Projeção de Receita & EBITDA ─────────────────────────────────────────
    with col_r:
        st.markdown("**📈 Projeção 5 Anos — Moderado**")
        if not err_m:
            years = list(range(1, 6))
            rev_proj  = results_moderate.get('revenue_projection', [])
            ebit_proj = results_moderate.get('ebitda_projection', [])
            fcff_proj = results_moderate.get('fcff_projection', [])
            if rev_proj:
                proj_df = pd.DataFrame({
                    'Ano': [f"Ano {y}" for y in years],
                    'Receita (R$M)':  [r / 1e6 for r in rev_proj],
                    'EBITDA (R$M)':   [e / 1e6 for e in ebit_proj],
                    'FCFF (R$M)':     [f / 1e6 for f in fcff_proj],
                }).set_index('Ano')
                st.dataframe(proj_df.style.format('R$ {:,.1f}'), use_container_width=True)

                fig_proj = go.Figure()
                fig_proj.add_trace(go.Bar(x=proj_df.index, y=proj_df['Receita (R$M)'],
                                          name='Receita', marker_color='#1f77b4', opacity=0.7))
                fig_proj.add_trace(go.Bar(x=proj_df.index, y=proj_df['EBITDA (R$M)'],
                                          name='EBITDA', marker_color='#2ca02c', opacity=0.8))
                fig_proj.add_trace(go.Scatter(x=proj_df.index, y=proj_df['FCFF (R$M)'],
                                              name='FCFF', mode='lines+markers',
                                              line=dict(color='#ff7f0e', width=2)))
                fig_proj.update_layout(
                    title='Projeção de Receita, EBITDA e FCFF', barmode='overlay',
                    height=280, margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation='h', y=-0.25)
                )
                st.plotly_chart(fig_proj, use_container_width=True)
        else:
            st.warning("Dados insuficientes para projeção.")

    # ── Composição do EV ─────────────────────────────────────────────────────
    st.markdown("**🏗️ Composição do Valor — Cenário Moderado**")
    if not err_m:
        pv_f = results_moderate.get('pv_fcffs', 0)
        pv_t = results_moderate.get('pv_terminal', 0)
        nd   = results_moderate.get('net_debt', 0)
        ev   = results_moderate.get('enterprise_value', 0)
        eq   = results_moderate.get('equity_value', 0)

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            fig_bridge = go.Figure(go.Waterfall(
                orientation='v',
                measure=['relative', 'relative', 'total', 'relative', 'total'],
                x=['PV FCFFs', 'PV Terminal', 'Enterprise Value', '(-) Dívida Líquida', 'Equity Value'],
                y=[pv_f / 1e9, pv_t / 1e9, 0, -nd / 1e9, 0],
                connector={'line': {'color': 'rgb(63,63,63)'}},
                increasing={'marker': {'color': '#2ca02c'}},
                decreasing={'marker': {'color': '#d62728'}},
                totals={'marker': {'color': '#1f77b4'}},
                text=[f"R${v/1e9:.1f}Bi" for v in [pv_f, pv_t, ev, nd, eq]],
                textposition='outside',
            ))
            fig_bridge.update_layout(title='Bridge EV → Equity (R$ Bilhões)',
                                     height=320, margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig_bridge, use_container_width=True)

        with comp_col2:
            tv_pct = results_moderate.get('tv_pct_ev', 0.5)
            fig_pie = px.pie(
                values=[1 - tv_pct, tv_pct],
                names=['FCFFs Explícitos', 'Valor Terminal'],
                title='Decomposição do EV',
                color_discrete_sequence=['#1f77b4', '#ff7f0e'],
                hole=0.45
            )
            fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)


def render_sensitivity_chart(df_sens: pd.DataFrame, ticker: str, current_price: float):
    """Renderiza heatmap de sensibilidade."""
    if df_sens.empty:
        st.warning("Sensibilidade indisponível.")
        return

    # Numérico para o heatmap
    z_vals = df_sens.values.astype(float)
    x_vals = df_sens.columns.tolist()
    y_vals = df_sens.index.tolist()

    fig = go.Figure(data=go.Heatmap(
        z=z_vals, x=x_vals, y=y_vals,
        colorscale='RdYlGn',
        zmid=current_price if not np.isnan(current_price) else None,
        text=[[f"R${v:.2f}" if not np.isnan(v) else "—" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont={'size': 10},
        colorbar=dict(title='Preço Justo (R$)'),
    ))

    # Destaque: linha com preço atual
    if not np.isnan(current_price):
        fig.add_annotation(
            text=f"Preço atual: R${current_price:.2f}",
            xref='paper', yref='paper', x=1.0, y=1.05,
            showarrow=False, font=dict(color='navy', size=11),
        )

    fig.update_layout(
        title=f'Sensibilidade do Preço Justo — {ticker} (Eixo X: g  |  Eixo Y: WACC)',
        xaxis_title='Taxa de Crescimento Perpétuo (g)',
        yaxis_title='WACC',
        height=420,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Pro v4.0 | DCF + Momentum")
    st.markdown("""
    **Plataforma Integrada: Motor Quant Multifator + Valuation Fundamentalista (DCF)**
    * **Quant Engine:** Residual Momentum + Value + Quality → DCA Backtest + Monte Carlo.
    * **DCF Engine:** FCFF / Gordon | Cenários Conservador × Moderado | Sensibilidade WACC × g.
    * **Dados:** Brapi.dev (fundamentos) + Yahoo Finance (fallback e demonstrativos).
    """)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    st.sidebar.header("1. Universo e Dados")
    default_univ = "ITUB3, TOTS3, MDIA3, TAEE3, BBSE3, WEGE3, PSSA3, EGIE3, B3SA3, VIVT3, AGRO3, PRIO3, BBAS3, BPAC11, SBSP3, SAPR4, CMIG3, UNIP6, FRAS3, CPFE3"
    ticker_input = st.sidebar.text_area("Tickers Quant (sem .SA)", default_univ, height=100)
    raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    yf_tickers = [f"{t}.SA" for t in raw_tickers]

    st.sidebar.header("2. Pesos Fatoriais")
    w_rm   = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_val  = st.sidebar.slider("Value (P/L, P/VP, EBITDA)", 0.0, 1.0, 0.40)
    w_qual = st.sidebar.slider("Quality (ROE, Margem)", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Gestão de Carteira")
    top_n         = st.sidebar.number_input("Número de Ativos", 4, 30, 10)
    use_vol_target = st.sidebar.checkbox("Risk Parity (Inv Vol)", True)

    st.sidebar.markdown("---")
    st.sidebar.header("4. Backtest & Monte Carlo")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 100000, 2000)
    dca_years  = st.sidebar.slider("Anos de Histórico", 2, 10, 5)
    mc_years   = st.sidebar.slider("Projeção Futura (Anos)", 1, 20, 5)

    st.sidebar.markdown("---")
    st.sidebar.header("5. 💎 Valuation DCF")
    dcf_tickers_default = list(DCF_UNIVERSE.keys())
    dcf_selected = st.sidebar.multiselect(
        "Ativos para Valuation",
        options=dcf_tickers_default,
        default=dcf_tickers_default[:8],
    )
    dcf_proj_years = st.sidebar.slider("Horizonte de Projeção (Anos)", 3, 7, 5)

    st.sidebar.markdown("**Macro Override (opcional)**")
    selic_override = st.sidebar.number_input("SELIC / Rf (%)", 0.0, 30.0,
                                             MACRO_PARAMS['selic'] * 100, step=0.25) / 100
    ipca_override  = st.sidebar.number_input("IPCA Proj. (%)", 0.0, 20.0,
                                             MACRO_PARAMS['ipca_proj'] * 100, step=0.25) / 100
    MACRO_PARAMS['selic']     = selic_override
    MACRO_PARAMS['ipca_proj'] = ipca_override

    run_btn_quant = st.sidebar.button("🚀 Executar Análise Quant", type="primary")
    run_btn_dcf   = st.sidebar.button("💎 Executar Valuation DCF",  type="secondary")

    # ── BLOCO QUANT ───────────────────────────────────────────────────────────
    if run_btn_quant:
        if not raw_tickers:
            st.error("Insira pelo menos um ticker.")
            return

        with st.status("Processando Pipeline Quantitativo...", expanded=True) as status:
            end_date   = datetime.now()
            start_date_total    = end_date - timedelta(days=365 * (dca_years + 3))
            start_date_backtest = end_date - timedelta(days=365 * dca_years)

            status.write("📥 Baixando Histórico de Preços...")
            prices = fetch_price_data(yf_tickers, start_date_total, end_date)
            if prices.empty:
                st.error("Falha ao baixar preços.")
                status.update(label="Erro", state="error")
                return

            status.write("🔍 Consultando Fundamentos (Brapi + Yahoo Fallback)...")
            fundamentals = fetch_fundamentals_hybrid(raw_tickers, BRAPI_TOKEN)
            if not fundamentals.empty:
                status.write(f"✅ Fundamentos: {len(fundamentals)} ativos.")
            else:
                status.write("⚠️ Fundamentos indisponíveis. Usando apenas Momentum.")

            status.write("🧮 Calculando Scores Atuais...")
            curr_mom = compute_residual_momentum_enhanced(prices)
            curr_val  = compute_value_robust(fundamentals) if not fundamentals.empty else pd.Series(0, index=prices.columns)
            curr_qual = compute_quality_score(fundamentals) if not fundamentals.empty else pd.Series(0, index=prices.columns)

            df_master = pd.DataFrame(index=prices.columns)
            df_master['Res_Mom'] = curr_mom
            df_master['Value']   = curr_val
            df_master['Quality'] = curr_qual
            if not fundamentals.empty and 'sector' in fundamentals.columns:
                df_master['Sector'] = fundamentals['sector']
            df_master.dropna(thresh=1, inplace=True)

            cols_map = {'Res_Mom': w_rm, 'Value': w_val, 'Quality': w_qual}
            df_master['Composite_Score'] = 0.0
            for col, weight in cols_map.items():
                if col in df_master.columns:
                    z = robust_zscore(df_master[col])
                    df_master[f'{col}_Z'] = z
                    df_master['Composite_Score'] += z * weight
            df_master = df_master.sort_values('Composite_Score', ascending=False)

            status.write("⚙️ Rodando Backtest Robusto...")
            dca_curve, dca_transactions, dca_holdings = run_dca_backtest_robust(
                prices, top_n, dca_amount, use_vol_target, start_date_backtest, end_date
            )
            status.update(label="Análise Quant Concluída!", state="complete", expanded=False)

        # ── Benchmarks ──────────────────────────────────────────────────────
        bench_curves = {}
        if not dca_transactions.empty:
            dca_dates = sorted(list(set(pd.to_datetime(dca_transactions['Date']).tolist())))
        else:
            dca_dates = []
        if dca_dates:
            for bench_ticker in ['BOVA11.SA', 'DIVO11.SA']:
                if bench_ticker in prices.columns:
                    bench_curve = run_benchmark_dca(prices[bench_ticker], dca_dates, dca_amount)
                    common_idx = dca_curve.index.intersection(bench_curve.index)
                    if not common_idx.empty:
                        bench_curves[bench_ticker] = bench_curve.loc[common_idx]

        tab1, tab2, tab6, tab3, tab4, tab5 = st.tabs([
            "🏆 Ranking Atual",
            "📈 Performance DCA",
            "🆚 Comparativo Benchmarks",
            "💰 Histórico & Custódia",
            "🔮 Monte Carlo",
            "📋 Dados Brutos"
        ])

        with tab1:
            st.subheader("🎯 Carteira Recomendada (Hoje)")
            top_picks = df_master.head(top_n).copy()
            latest_prices = prices.iloc[-1]
            top_picks['Preço Atual'] = latest_prices.reindex(top_picks.index)
            risk_window = prices.tail(90)
            sug_weights = construct_portfolio(top_picks, risk_window, top_n, use_vol_target)
            top_picks['Peso (%)']     = sug_weights * 100
            top_picks['Alocação (R$)'] = sug_weights * dca_amount
            top_picks['Qtd Sugerida'] = top_picks['Alocação (R$)'] / top_picks['Preço Atual']
            cols_show = ['Sector', 'Preço Atual', 'Composite_Score', 'Peso (%)', 'Alocação (R$)', 'Qtd Sugerida', 'Value', 'Quality']
            cols_final = [c for c in cols_show if c in top_picks.columns]
            display_df = top_picks[cols_final].style.format({
                'Preço Atual': 'R$ {:.2f}', 'Composite_Score': '{:.2f}',
                'Value': '{:.2f}', 'Quality': '{:.2f}',
                'Peso (%)': '{:.1f}%', 'Alocação (R$)': 'R$ {:.0f}', 'Qtd Sugerida': '{:.0f}'
            }).background_gradient(subset=['Composite_Score'], cmap='Greens')
            st.dataframe(display_df, use_container_width=True, height=400)
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.plotly_chart(px.pie(values=sug_weights, names=sug_weights.index, title="Alocação Sugerida"), use_container_width=True)
            with col_chart2:
                if 'Sector' in top_picks.columns:
                    st.plotly_chart(px.pie(top_picks, names='Sector', values='Peso (%)', title="Exposição Setorial"), use_container_width=True)

        with tab2:
            st.subheader("Simulação de Acumulação (DCA)")
            if not dca_curve.empty:
                end_val = dca_curve.iloc[-1, 0]
                unique_months = pd.to_datetime(dca_transactions['Date']).dt.to_period('M').nunique()
                total_invested_real = unique_months * dca_amount
                profit = end_val - total_invested_real
                roi = (profit / total_invested_real) if total_invested_real > 0 else 0
                m1, m2, m3 = st.columns(3)
                m1.metric("Patrimônio Final", f"R$ {end_val:,.2f}")
                m2.metric("Total Investido", f"R$ {total_invested_real:,.2f}")
                m3.metric("Lucro Líquido", f"R$ {profit:,.2f}", delta=f"{roi:.1%}")
                st.plotly_chart(px.line(dca_curve, title="Curva de Patrimônio (Estratégia)"), use_container_width=True)
                st.markdown("### Análise de Risco")
                st.json(calculate_advanced_metrics(dca_curve['Strategy_DCA']))

        with tab6:
            st.subheader("🆚 Estratégia vs Benchmarks")
            if not dca_curve.empty and bench_curves:
                df_compare = dca_curve.copy()
                for b_name, b_series in bench_curves.items():
                    df_compare[b_name] = b_series
                df_compare = df_compare.ffill().dropna()
                st.plotly_chart(px.line(df_compare, title="Evolução Patrimonial Comparativa"), use_container_width=True)
                comp_metrics = []
                m_strat = calculate_advanced_metrics(df_compare['Strategy_DCA'])
                m_strat['Asset'] = '🚀 Estratégia'
                m_strat['Saldo Final'] = df_compare['Strategy_DCA'].iloc[-1]
                comp_metrics.append(m_strat)
                for b_name in bench_curves.keys():
                    if b_name in df_compare.columns:
                        m_bench = calculate_advanced_metrics(df_compare[b_name])
                        m_bench['Asset'] = b_name
                        m_bench['Saldo Final'] = df_compare[b_name].iloc[-1]
                        comp_metrics.append(m_bench)
                df_comp_metrics = pd.DataFrame(comp_metrics).set_index('Asset')
                cols_order = ['Saldo Final', 'Retorno Total', 'CAGR', 'Volatilidade', 'Sharpe', 'Max Drawdown']
                st.dataframe(
                    df_comp_metrics[cols_order].style.format({
                        'Saldo Final': 'R$ {:,.2f}', 'Retorno Total': '{:.1%}',
                        'CAGR': '{:.1%}', 'Volatilidade': '{:.1%}',
                        'Sharpe': '{:.2f}', 'Max Drawdown': '{:.1%}'
                    }).highlight_max(subset=['Saldo Final'], color='#d4edda'),
                    use_container_width=True
                )
            else:
                st.warning("Dados insuficientes para comparação.")

        with tab3:
            col_h1, col_h2 = st.columns([1, 1])
            with col_h1:
                st.subheader("💰 Posição Final (Backtest)")
                if dca_holdings:
                    final_df = pd.DataFrame.from_dict(dca_holdings, orient='index', columns=['Qtd'])
                    last_date_idx = dca_curve.index[-1]
                    if last_date_idx in prices.index:
                        last_prices = prices.loc[last_date_idx]
                        final_df['Preço Fechamento'] = last_prices.reindex(final_df.index)
                        final_df['Valor Total (R$)'] = final_df['Qtd'] * final_df['Preço Fechamento']
                        total_nav = final_df['Valor Total (R$)'].sum()
                        final_df['Peso (%)'] = (final_df['Valor Total (R$)'] / total_nav) * 100
                        final_df = final_df.sort_values('Peso (%)', ascending=False)
                        st.dataframe(final_df.style.format({
                            'Qtd': '{:.0f}', 'Preço Fechamento': 'R$ {:.2f}',
                            'Valor Total (R$)': 'R$ {:,.2f}', 'Peso (%)': '{:.1f}%'
                        }), use_container_width=True)
                        st.metric("Patrimônio em Custódia", f"R$ {total_nav:,.2f}")
                else:
                    st.info("Nenhuma posição mantida.")
            with col_h2:
                st.subheader("📊 Alocação Final")
                if dca_holdings:
                    st.plotly_chart(px.pie(final_df, values='Valor Total (R$)', names=final_df.index, hole=0.4), use_container_width=True)
            st.divider()
            if not dca_transactions.empty:
                st.subheader("Histórico de Transações")
                st.dataframe(pd.DataFrame(dca_transactions).sort_values('Date', ascending=False), use_container_width=True)

        with tab4:
            st.subheader("Projeção Probabilística")
            if not dca_curve.empty:
                daily_rets = dca_curve['Strategy_DCA'].pct_change().dropna()
                if not daily_rets.empty:
                    mu = daily_rets.mean() * 252
                    sigma = daily_rets.std() * np.sqrt(252)
                    sim_df = run_monte_carlo(dca_curve.iloc[-1, 0], dca_amount, mu, sigma, mc_years)
                    if not sim_df.empty:
                        st.plotly_chart(px.line(sim_df, title=f"Cone de Probabilidade - {mc_years} Anos"), use_container_width=True)
                    else:
                        st.warning("Dados insuficientes para Monte Carlo.")

        with tab5:
            st.subheader("Dados Fundamentais (Brapi + YF)")
            if not fundamentals.empty:
                st.dataframe(fundamentals)
                st.caption("Dados via Brapi.dev com fallback automático para Yahoo Finance.")
            else:
                st.error("Falha na recuperação de fundamentos.")

    # ── BLOCO DCF ─────────────────────────────────────────────────────────────
    if run_btn_dcf:
        if not dcf_selected:
            st.warning("Selecione ao menos um ativo para o valuation.")
            return

        st.markdown("---")
        st.header("💎 Valuation Fundamentalista — DCF + Gordon")

        macro_info_cols = st.columns(4)
        macro_info_cols[0].metric("SELIC (Rf)", f"{MACRO_PARAMS['selic']:.2%}")
        macro_info_cols[1].metric("IPCA Projetado", f"{MACRO_PARAMS['ipca_proj']:.2%}")
        macro_info_cols[2].metric("ERP Brasil", f"{MACRO_PARAMS['erp_brazil']:.2%}")
        macro_info_cols[3].metric("Alíquota IR/CSLL", f"{MACRO_PARAMS['tax_rate']:.0%}")

        st.info(f"📊 **Metodologia:** FCFF = NOPAT + D&A − CapEx − ΔCGL | "
                f"Valor Terminal (Gordon): TV = FCFF_n+1 / (WACC − g) | "
                f"Horizonte: {dcf_proj_years} anos | {len(dcf_selected)} ativos selecionados")

        # Tabela-resumo comparativa
        st.subheader("📊 Painel Consolidado de Valuation")
        summary_rows = []

        with st.spinner("Baixando demonstrativos financeiros e calculando DCF..."):
            all_results = {}
            all_fin_data = {}
            progress = st.progress(0)

            for idx, t in enumerate(dcf_selected):
                fin = fetch_dcf_financials(t)
                res_c = run_dcf_model(t, fin, scenario='conservative', projection_years=dcf_proj_years)
                res_m = run_dcf_model(t, fin, scenario='moderate',     projection_years=dcf_proj_years)
                all_fin_data[t]  = fin
                all_results[t]   = {'conservative': res_c, 'moderate': res_m}

                cp    = fin.get('current_price', np.nan)
                fp_c  = res_c.get('fair_price', np.nan)
                fp_m  = res_m.get('fair_price', np.nan)
                mos   = res_m.get('margin_of_safety', np.nan)
                irr   = res_m.get('irr', np.nan)
                wacc  = res_m.get('wacc', np.nan)
                g_    = res_m.get('terminal_g', np.nan)
                upside_m = (fp_m / cp - 1) if (fp_m and cp and cp > 0) else np.nan

                def verdict(mos_val):
                    if np.isnan(mos_val): return "⬜ N/D"
                    if mos_val > 0.25:    return "🟢 COMPRA"
                    if mos_val > 0.05:    return "🟡 NEUTRO+"
                    if mos_val > -0.10:   return "🟡 NEUTRO"
                    return "🔴 CARO"

                summary_rows.append({
                    'Ticker': t,
                    'Empresa': DCF_UNIVERSE[t]['name'],
                    'Setor': DCF_UNIVERSE[t]['sector'],
                    'Preço Atual': cp,
                    'PJ Conserv.': fp_c,
                    'PJ Moderado': fp_m,
                    'Upside Mod.': upside_m,
                    'Margem Seg.': mos,
                    'TIR Impl.': irr,
                    'WACC': wacc,
                    'g (Term.)': g_,
                    'Veredicto': verdict(mos),
                })
                progress.progress((idx + 1) / len(dcf_selected))

            progress.empty()

        df_summary = pd.DataFrame(summary_rows).set_index('Ticker')

        def color_verdict(val):
            if '🟢' in str(val): return 'background-color: #d4edda; color: #155724'
            if '🔴' in str(val): return 'background-color: #f8d7da; color: #721c24'
            if '🟡' in str(val): return 'background-color: #fff3cd; color: #856404'
            return ''

        def color_upside(val):
            try:
                v = float(val)
                if v > 0.25:  return 'color: #155724; font-weight: bold'
                if v > 0.05:  return 'color: #856404'
                if v < -0.10: return 'color: #721c24'
            except:
                pass
            return ''

        fmt_map = {
            'Preço Atual': 'R$ {:.2f}', 'PJ Conserv.': 'R$ {:.2f}', 'PJ Moderado': 'R$ {:.2f}',
            'Upside Mod.': '{:.1%}', 'Margem Seg.': '{:.1%}', 'TIR Impl.': '{:.1%}',
            'WACC': '{:.2%}', 'g (Term.)': '{:.2%}',
        }

        st.dataframe(
            df_summary.style
                .format(fmt_map, na_rep='N/D')
                .applymap(color_verdict, subset=['Veredicto'])
                .applymap(color_upside, subset=['Upside Mod.']),
            use_container_width=True, height=480
        )

        # ── Detalhe por ativo ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Análise Detalhada por Ativo")

        if len(dcf_selected) <= 8:
            cols_per_row = 4
            ticker_tabs = st.tabs([f"{t}" for t in dcf_selected])
            for tab_obj, t in zip(ticker_tabs, dcf_selected):
                with tab_obj:
                    res_c = all_results[t]['conservative']
                    res_m = all_results[t]['moderate']
                    fin   = all_fin_data[t]
                    render_valuation_summary(res_c, res_m, fin, t)

                    # Sensibilidade
                    st.markdown("**🎛️ Análise de Sensibilidade — Preço Justo (Cenário Moderado)**")
                    with st.spinner("Calculando sensibilidade..."):
                        df_sens = build_sensitivity_table(t, fin, res_m, 'moderate')
                    render_sensitivity_chart(df_sens, t, fin.get('current_price', np.nan))

                    with st.expander("Ver tabela numérica de sensibilidade"):
                        if not df_sens.empty:
                            st.dataframe(df_sens.style.format('R$ {:.2f}', na_rep='N/D')
                                         .background_gradient(cmap='RdYlGn', axis=None),
                                         use_container_width=True)
        else:
            selected_detail = st.selectbox("Selecionar ativo para detalhe:", dcf_selected)
            if selected_detail:
                res_c = all_results[selected_detail]['conservative']
                res_m = all_results[selected_detail]['moderate']
                fin   = all_fin_data[selected_detail]
                render_valuation_summary(res_c, res_m, fin, selected_detail)
                st.markdown("**🎛️ Análise de Sensibilidade — Preço Justo (Cenário Moderado)**")
                with st.spinner("Calculando sensibilidade..."):
                    df_sens = build_sensitivity_table(selected_detail, fin, res_m, 'moderate')
                render_sensitivity_chart(df_sens, selected_detail, fin.get('current_price', np.nan))

        # ── Gráfico comparativo: Preço Atual vs Preços Justos ─────────────
        st.markdown("---")
        st.subheader("📊 Comparativo: Preço Atual × Preços Justos")

        chart_data_rows = []
        for t in dcf_selected:
            res_c = all_results[t]['conservative']
            res_m = all_results[t]['moderate']
            fin   = all_fin_data[t]
            chart_data_rows.append({
                'Ticker': t,
                'Preço Atual':       fin.get('current_price', np.nan),
                'PJ Conservador':    res_c.get('fair_price', np.nan),
                'PJ Moderado':       res_m.get('fair_price', np.nan),
            })

        df_chart = pd.DataFrame(chart_data_rows).set_index('Ticker').dropna(how='all')

        if not df_chart.empty:
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=df_chart.index, y=df_chart['PJ Moderado'],
                name='Preço Justo Moderado', marker_color='#2ca02c', opacity=0.75
            ))
            fig_compare.add_trace(go.Bar(
                x=df_chart.index, y=df_chart['PJ Conservador'],
                name='Preço Justo Conservador', marker_color='#1f77b4', opacity=0.75
            ))
            fig_compare.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart['Preço Atual'],
                name='Preço Atual', mode='markers',
                marker=dict(color='red', size=10, symbol='diamond')
            ))
            fig_compare.update_layout(
                barmode='group', height=450,
                title='Preço Atual vs Preços Justos por Cenário',
                yaxis_title='R$',
                legend=dict(orientation='h', y=-0.2),
                margin=dict(l=30, r=30, t=50, b=50)
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        # ── Notas metodológicas ────────────────────────────────────────────
        with st.expander("📚 Notas Metodológicas e Limitações"):
            st.markdown(f"""
**Estrutura do Modelo DCF:**
- **FCFF** = NOPAT + D&A − CapEx − ΔCapital de Giro Líquido
- **NOPAT** = EBIT × (1 − Alíquota IR/CSLL de {MACRO_PARAMS['tax_rate']:.0%})
- **Valor Terminal (Gordon):** TV = FCFF_n+1 ÷ (WACC − g)
- **Equity Value** = PV(FCFFs) + PV(TV) − Dívida Líquida

**Custo de Capital (CAPM Brasil):**
- Ke = SELIC + β × ERP_Brasil + Prêmio de Risco do Cenário
- Kd = SELIC + Spread Setorial
- WACC = Ke × %PL + Kd × (1−IR) × %Dívida

**Cenário Conservador:** Crescimento = IPCA, compressão de margem, prêmio de risco +150-250bps, g = {MACRO_PARAMS['terminal_g_conserv']:.1%}

**Cenário Moderado:** Crescimento = IPCA + pipeline setorial, margens estáveis, WACC histórico, g = {MACRO_PARAMS['terminal_g_moderate']:.1%}

**⚠️ Limitações:**
- Dados de demonstrativos financeiros obtidos via Yahoo Finance (TTM). Divergências em relação a relatórios oficiais são esperadas.
- Premissas setoriais são médias históricas; empresas individuais podem diferir substancialmente.
- Bancos e seguradoras possuem estrutura de capital distinta — o modelo FCFF é uma aproximação.
- O modelo não captura eventos extraordinários, mudanças regulatórias abruptas ou riscos geopolíticos.
- **Não constitui recomendação de investimento.** Use como ferramenta de suporte à decisão.
""")


if __name__ == "__main__":
    main()
