"""
╔══════════════════════════════════════════════════════════════════╗
║   QUANT FACTOR LAB PRO v4.0  |  DCF VALUATION ENGINE INTEGRADO  ║
║   Quantitative + Fundamental Analysis · Brazilian Equities       ║
╚══════════════════════════════════════════════════════════════════╝
"""

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
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Factor Lab Pro v4.0 | DCF Valuation",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Quant Factor Lab Pro v4.0 — Motor Híbrido + Valuation DCF"}
)

# ──────────────────────────────────────────────────────────────────
# CONSTANTES GLOBAIS
# ──────────────────────────────────────────────────────────────────
BRAPI_TOKEN = "5gVedSQ928pxhFuTvBFPfr"

# Universe fixo para valuation
VALUATION_UNIVERSE = {
    "EGIE3":  {"name": "Engie Brasil",       "sector": "Utilities",  "model": "dcf", "beta": 0.55},
    "ITUB3":  {"name": "Itaú Unibanco",      "sector": "Banks",      "model": "ddm", "beta": 0.85},
    "PSSA3":  {"name": "Porto Seguro",        "sector": "Insurance",  "model": "dcf", "beta": 0.70},
    "WEGE3":  {"name": "WEG",                "sector": "Industrial", "model": "dcf", "beta": 0.90},
    "CXSE3":  {"name": "Caixa Seguridade",   "sector": "Insurance",  "model": "ddm", "beta": 0.60},
    "SBSP3":  {"name": "Sabesp",             "sector": "Sanitation", "model": "dcf", "beta": 0.65},
    "TAEE3":  {"name": "Taesa",              "sector": "Utilities",  "model": "dcf", "beta": 0.50},
    "VIVT3":  {"name": "Telefônica Vivo",    "sector": "Telecom",    "model": "dcf", "beta": 0.75},
    "CPFE3":  {"name": "CPFL Energia",       "sector": "Utilities",  "model": "dcf", "beta": 0.60},
    "SAPR3":  {"name": "Sanepar",            "sector": "Sanitation", "model": "dcf", "beta": 0.60},
    "BBAS3":  {"name": "Banco do Brasil",    "sector": "Banks",      "model": "ddm", "beta": 0.90},
    "PRIO3":  {"name": "PetroRio",           "sector": "Oil & Gas",  "model": "dcf", "beta": 1.30},
    "TOTS3":  {"name": "TOTVS",              "sector": "Technology", "model": "dcf", "beta": 0.95},
    "BPAC3":  {"name": "BTG Pactual",        "sector": "Banks",      "model": "ddm", "beta": 1.10},
    "ALUP3":  {"name": "Alupar",             "sector": "Utilities",  "model": "dcf", "beta": 0.50},
    "BMOB3":  {"name": "Bemobi",             "sector": "Technology", "model": "dcf", "beta": 1.10},
}

# Parâmetros WACC/Ke por setor — cenários Conservador e Moderado
SECTOR_PARAMS = {
    "Utilities":  {
        "wacc_cons": 0.110, "wacc_mod": 0.095,
        "g_cons": 0.035,    "g_mod": 0.045,
        "tax": 0.34, "capex_rev": 0.15,
        "rev_growth_cons": [0.050, 0.050, 0.050, 0.050, 0.050],
        "rev_growth_mod":  [0.070, 0.075, 0.080, 0.075, 0.065],
        "description": "Concessões reguladas com RAP reajustado por IPCA/IGP-M. Alta previsibilidade de fluxo de caixa."
    },
    "Sanitation": {
        "wacc_cons": 0.115, "wacc_mod": 0.100,
        "g_cons": 0.040,    "g_mod": 0.050,
        "tax": 0.34, "capex_rev": 0.22,
        "rev_growth_cons": [0.050, 0.050, 0.055, 0.055, 0.055],
        "rev_growth_mod":  [0.100, 0.105, 0.095, 0.090, 0.080],
        "description": "Receita regulada com expansão de cobertura. Ciclo de CapEx intenso pós-novo marco do saneamento."
    },
    "Banks": {
        "ke_cons": 0.185, "ke_mod": 0.165,
        "g_cons": 0.040,  "g_mod": 0.055,
        "payout": 0.40,
        "eps_growth_cons": 0.05, "eps_growth_mod": 0.10,
        "description": "Modelo DDM — bancos têm estrutura de capital regulada, FCFF não aplicável. Avaliação por Dividendos Descontados."
    },
    "Insurance": {
        "ke_cons": 0.175, "ke_mod": 0.155,
        "g_cons": 0.040,  "g_mod": 0.055,
        "tax": 0.34, "capex_rev": 0.02,
        "rev_growth_cons": [0.055, 0.055, 0.050, 0.050, 0.050],
        "rev_growth_mod":  [0.120, 0.115, 0.110, 0.100, 0.090],
        "payout": 0.50,
        "eps_growth_cons": 0.05, "eps_growth_mod": 0.10,
        "description": "Receita de prêmios crescendo acima do PIB. Combinado operacional como principal driver de qualidade."
    },
    "Industrial": {
        "wacc_cons": 0.125, "wacc_mod": 0.110,
        "g_cons": 0.040,    "g_mod": 0.060,
        "tax": 0.34, "capex_rev": 0.07,
        "rev_growth_cons": [0.050, 0.055, 0.055, 0.055, 0.050],
        "rev_growth_mod":  [0.120, 0.130, 0.125, 0.115, 0.100],
        "description": "Crescimento via expansão internacional e novos produtos. Margem EBITDA defendida por pricing power."
    },
    "Telecom": {
        "wacc_cons": 0.120, "wacc_mod": 0.105,
        "g_cons": 0.035,    "g_mod": 0.045,
        "tax": 0.34, "capex_rev": 0.20,
        "rev_growth_cons": [0.040, 0.045, 0.045, 0.040, 0.040],
        "rev_growth_mod":  [0.065, 0.070, 0.065, 0.060, 0.055],
        "description": "Setor maduro com crescimento em fibra e 5G. CapEx elevado em infraestrutura de rede."
    },
    "Oil & Gas": {
        "wacc_cons": 0.150, "wacc_mod": 0.130,
        "g_cons": 0.020,    "g_mod": 0.035,
        "tax": 0.34, "capex_rev": 0.30,
        "rev_growth_cons": [0.050, 0.050, 0.030, 0.020, 0.020],
        "rev_growth_mod":  [0.180, 0.150, 0.120, 0.090, 0.060],
        "description": "E&P com crescimento de produção (barris/dia). Alta volatilidade atrelada ao preço do Brent e câmbio."
    },
    "Technology": {
        "wacc_cons": 0.140, "wacc_mod": 0.120,
        "g_cons": 0.045,    "g_mod": 0.065,
        "tax": 0.34, "capex_rev": 0.05,
        "rev_growth_cons": [0.070, 0.070, 0.065, 0.060, 0.055],
        "rev_growth_mod":  [0.160, 0.155, 0.145, 0.130, 0.110],
        "description": "Receita recorrente (SaaS/assinaturas). Expansão de margens via alavancagem operacional digital."
    },
}

# ══════════════════════════════════════════════════════════════════
# MÓDULO 1 — PRICE DATA (HÍBRIDO, CORRIGIDO)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600 * 12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados via YFinance com tratamento robusto."""
    t_list = list(tickers)
    for bench in ["BOVA11.SA", "DIVO11.SA"]:
        if bench not in t_list:
            t_list.append(bench)

    try:
        # auto_adjust=True retorna preços já ajustados em 'Close'
        raw = yf.download(
            t_list,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        # Normaliza MultiIndex → plano
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw["Close"].copy()
        elif "Close" in raw.columns:
            data = raw[["Close"]].copy()
        else:
            data = raw.copy()

        if isinstance(data, pd.Series):
            data = data.to_frame()

        data = data.dropna(axis=1, how="all")
        return data

    except Exception as e:
        st.error(f"Erro crítico ao baixar preços (YF): {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# MÓDULO 2 — FUNDAMENTOS (HÍBRIDO: BRAPI + YFINANCE FALLBACK)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600 * 4)
def fetch_fundamentals_hybrid(tickers: list, token: str) -> pd.DataFrame:
    """Fundamentos via Brapi com fallback automático para Yahoo Finance."""
    clean_tickers = [
        t.replace(".SA", "")
        for t in tickers
        if "BOVA11" not in t and "DIVO11" not in t
    ]
    if not clean_tickers:
        return pd.DataFrame()

    fundamental_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(clean_tickers)

    def safe_float(val):
        if val is None or str(val).lower() in ("", "nan", "none"):
            return np.nan
        try:
            return float(val)
        except Exception:
            return np.nan

    def nested_val(item, keys):
        for k in keys:
            if k in item and item[k] is not None:
                return item[k]
        for mod in ["defaultKeyStatistics", "financialData", "summaryProfile", "price"]:
            if mod in item and isinstance(item[mod], dict):
                for k in keys:
                    if k in item[mod]:
                        return item[mod][k]
        return None

    for i, ticker in enumerate(clean_tickers):
        status_text.text(f"Analisando: {ticker} ({i + 1}/{total}) — Brapi...")
        price = market_cap = pe = p_vp = ev_ebitda = roe = net_margin = np.nan
        sector = "Outros"

        # 1. Brapi
        try:
            r = requests.get(
                f"https://brapi.dev/api/quote/{ticker}",
                params={"token": token, "fundamental": "true"},
                timeout=10,
            )
            if r.status_code == 200:
                res = r.json().get("results", [])
                if res:
                    it = res[0]
                    price = safe_float(it.get("regularMarketPrice"))
                    market_cap = safe_float(it.get("marketCap"))
                    sector = it.get("sector") or "Outros"
                    pe = safe_float(nested_val(it, ["priceEarnings", "trailingPE"]))
                    p_vp = safe_float(nested_val(it, ["priceToBook", "priceToBookRatio", "p_vp"]))
                    ev_ebitda = safe_float(nested_val(it, ["enterpriseToEbitda", "ev_ebitda"]))
                    roe = safe_float(nested_val(it, ["returnOnEquity", "roe"]))
                    net_margin = safe_float(nested_val(it, ["profitMargin", "netMargin"]))
        except Exception:
            pass

        # 2. YFinance fallback
        if any(np.isnan(v) for v in [p_vp, roe, ev_ebitda]):
            status_text.text(f"Complementando: {ticker} — Yahoo Finance...")
            try:
                yft = yf.Ticker(f"{ticker}.SA")
                info = yft.info
                if np.isnan(price):
                    price = info.get("currentPrice") or info.get("previousClose")
                if np.isnan(market_cap):
                    market_cap = info.get("marketCap")
                if sector == "Outros":
                    sector = info.get("sector", "Outros")
                if np.isnan(pe):
                    pe = info.get("trailingPE")
                if np.isnan(p_vp):
                    p_vp = info.get("priceToBook")
                if np.isnan(ev_ebitda):
                    ev_ebitda = info.get("enterpriseToEbitda")
                if np.isnan(roe):
                    roe = info.get("returnOnEquity")
                if np.isnan(net_margin):
                    net_margin = info.get("profitMargins")
            except Exception:
                pass

        fundamental_data.append(
            {
                "ticker": f"{ticker}.SA",
                "sector": sector,
                "currentPrice": price,
                "marketCap": market_cap,
                "PE": pe,
                "P_VP": p_vp,
                "EV_EBITDA": ev_ebitda,
                "ROE": roe,
                "Net_Margin": net_margin,
            }
        )
        progress_bar.progress((i + 1) / total)
        time.sleep(0.4)

    progress_bar.empty()
    status_text.empty()

    df = pd.DataFrame(fundamental_data)
    if not df.empty:
        df = df.drop_duplicates(subset=["ticker"]).set_index("ticker")
        for col in ["PE", "P_VP", "EV_EBITDA", "ROE", "Net_Margin"]:
            if col in df.columns:
                df[col] = df[col].replace([0, 0.0], np.nan)
    return df


# ══════════════════════════════════════════════════════════════════
# MÓDULO 3 — CÁLCULO DE FATORES
# ══════════════════════════════════════════════════════════════════

def compute_residual_momentum_enhanced(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Residual Momentum (Blitz) com Volatility Scaling."""
    df = price_df.copy()
    monthly = df.resample("ME").last()
    rets = monthly.pct_change().dropna()
    if "BOVA11.SA" not in rets.columns:
        return pd.Series(dtype=float)
    market = rets["BOVA11.SA"]
    scores = {}
    regression_window = 36
    for ticker in rets.columns:
        if ticker in ["BOVA11.SA", "DIVO11.SA"]:
            continue
        y_full = rets[ticker].tail(regression_window + skip)
        x_full = market.tail(regression_window + skip)
        if len(y_full) < 12:
            continue
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
        except Exception:
            scores[ticker] = 0
    return pd.Series(scores, name="Residual_Momentum")


def compute_value_robust(fund_df: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=fund_df.index)

    def invert(s):
        return 1.0 / s.replace(0, np.nan)

    if "PE" in fund_df:
        scores["Earnings_Yield"] = invert(fund_df["PE"])
    if "P_VP" in fund_df:
        scores["Book_Yield"] = invert(fund_df["P_VP"])
    if "EV_EBITDA" in fund_df:
        scores["EBITDA_Yield"] = invert(fund_df["EV_EBITDA"])
    if scores.empty or scores.dropna(how="all").empty:
        return pd.Series(0, index=fund_df.index, name="Value_Score")
    for col in scores.columns:
        filled = scores[col].fillna(scores[col].median())
        scores[col] = (filled - filled.mean()) / filled.std() if filled.std() > 0 else 0
    return scores.mean(axis=1).rename("Value_Score")


def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=fund_df.index)
    if "ROE" in fund_df:
        scores["ROE"] = fund_df["ROE"]
    if "Net_Margin" in fund_df:
        scores["Margin"] = fund_df["Net_Margin"]
    if scores.empty or scores.dropna(how="all").empty:
        return pd.Series(0, index=fund_df.index, name="Quality_Score")
    for col in scores.columns:
        filled = scores[col].fillna(scores[col].median())
        scores[col] = (filled - filled.mean()) / filled.std() if filled.std() > 0 else 0
    return scores.mean(axis=1).rename("Quality_Score")


# ══════════════════════════════════════════════════════════════════
# MÓDULO 4 — MÉTRICAS AVANÇADAS
# ══════════════════════════════════════════════════════════════════

def robust_zscore(series: pd.Series) -> pd.Series:
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or mad < 1e-6:
        return series - median
    return ((series - median) / (mad * 1.4826)).clip(-3, 3)


def calculate_advanced_metrics(prices_series: pd.Series, rf_annual: float = 0.105):
    if prices_series.empty or len(prices_series) < 2:
        return {}
    daily_rets = prices_series.pct_change().dropna()
    if daily_rets.empty:
        return {}
    total_ret = (prices_series.iloc[-1] / prices_series.iloc[0]) - 1
    days = (prices_series.index[-1] - prices_series.index[0]).days
    cagr = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
    vol_ann = daily_rets.std() * np.sqrt(252)
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = daily_rets - rf_daily
    sharpe = (excess.mean() * 252) / vol_ann if vol_ann > 0 else 0
    down_std = excess[excess < 0].std() * np.sqrt(252)
    sortino = (excess.mean() * 252) / down_std if down_std > 0 else 0
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        "Retorno Total": total_ret, "CAGR": cagr, "Volatilidade": vol_ann,
        "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "Max Drawdown": max_dd, "Ulcer Index": float(np.sqrt((dd ** 2).mean())),
    }


# ══════════════════════════════════════════════════════════════════
# MÓDULO 5 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════

def run_monte_carlo(initial_balance, monthly_contrib, mu_annual, sigma_annual, years, simulations=1000):
    if np.isnan(mu_annual) or np.isnan(sigma_annual):
        return pd.DataFrame()
    months = int(years * 12)
    dt = 1 / 12
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    sigma_annual = max(sigma_annual, 0.01)
    shock = sigma_annual * np.sqrt(dt) * np.random.normal(0, 1, (months, simulations))
    monthly_returns = np.exp(drift + shock) - 1
    paths = np.zeros((months + 1, simulations))
    paths[0] = initial_balance
    for t in range(1, months + 1):
        paths[t] = paths[t - 1] * (1 + monthly_returns[t - 1]) + monthly_contrib
    pcts = np.percentile(paths, [5, 50, 95], axis=1)
    dates = [datetime.now() + timedelta(days=30 * i) for i in range(months + 1)]
    return pd.DataFrame(
        {"Pessimista (5%)": pcts[0], "Base (50%)": pcts[1], "Otimista (95%)": pcts[2]},
        index=dates,
    )


# ══════════════════════════════════════════════════════════════════
# MÓDULO 6 — BACKTEST & CONSTRUÇÃO DE PORTFÓLIO
# ══════════════════════════════════════════════════════════════════

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: bool = False):
    selected = ranked_df.head(top_n).index.tolist()
    if not selected:
        return pd.Series()
    if vol_target:
        valid = [s for s in selected if s in prices.columns]
        if not valid:
            return pd.Series()
        rets = prices[valid].pct_change().tail(63)
        vols = rets.std() * 252 ** 0.5
        vols = vols.replace(0, 1e-6)
        inv = 1 / vols
        weights = inv / inv.sum() if inv.sum() > 0 else pd.Series(1 / len(valid), index=valid)
    else:
        weights = pd.Series(1.0 / len(selected), index=selected)
    return weights.sort_values(ascending=False)


def run_dca_backtest_robust(all_prices, top_n, dca_amount, use_vol_target, start_date, end_date):
    all_prices = all_prices.ffill()
    dca_start = start_date + timedelta(days=30)
    market_cal = pd.Series(all_prices.index, index=all_prices.index)
    dates_series = market_cal.loc[dca_start:end_date].resample("MS").first()
    dates = dates_series.dropna().tolist()
    if not dates or len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    portfolio_value = pd.Series(0.0, index=all_prices.index)
    portfolio_holdings = {}
    monthly_transactions = []
    cash = 0.0

    for i, month_start in enumerate(dates):
        eval_date = month_start - timedelta(days=1)
        mom_start = month_start - timedelta(days=365 * 3)
        prices_hist = all_prices.loc[:eval_date]
        prices_win = prices_hist.loc[mom_start:]
        if prices_win.empty:
            continue
        res_mom = compute_residual_momentum_enhanced(prices_win, lookback=12, skip=1)
        if res_mom.empty:
            continue
        df_rank = pd.DataFrame({"Score": robust_zscore(res_mom)}).sort_values("Score", ascending=False)
        risk_window = prices_hist.tail(90)
        target_weights = construct_portfolio(df_rank, risk_window, top_n, use_vol_target)
        try:
            if month_start not in all_prices.index:
                nxt = all_prices.index[all_prices.index > month_start]
                if nxt.empty:
                    break
                exec_date = nxt[0]
            else:
                exec_date = month_start
            curr_prices = all_prices.loc[exec_date]
        except KeyError:
            continue

        curr_val = cash + sum(
            q * curr_prices[t]
            for t, q in portfolio_holdings.items()
            if t in curr_prices and not np.isnan(curr_prices[t])
        )
        total_val = curr_val + dca_amount
        new_holdings = {}
        for ticker, weight in target_weights.items():
            if ticker in curr_prices and not np.isnan(curr_prices[ticker]) and curr_prices[ticker] > 0:
                qty = (total_val * weight) / curr_prices[ticker]
                new_holdings[ticker] = qty
                monthly_transactions.append({
                    "Date": exec_date, "Ticker": ticker,
                    "Action": "Rebalance/Buy", "Price": curr_prices[ticker], "Weight": weight,
                })
        portfolio_holdings = new_holdings

        next_rb = dates[i + 1] if i < len(dates) - 1 else end_date
        valid_end = min(next_rb, all_prices.index[-1])
        if exec_date > valid_end:
            continue
        for d in all_prices.loc[exec_date:valid_end].index:
            val = sum(
                q * all_prices.at[d, t]
                for t, q in portfolio_holdings.items()
                if not np.isnan(all_prices.at[d, t])
            )
            portfolio_value[d] = val

    portfolio_value = portfolio_value[portfolio_value > 0].sort_index()
    equity_curve = pd.DataFrame({"Strategy_DCA": portfolio_value})
    transactions_df = pd.DataFrame(monthly_transactions)
    return equity_curve, transactions_df, portfolio_holdings


def run_benchmark_dca(price_series: pd.Series, dates: list, dca_amount: float):
    if price_series.empty:
        return pd.Series()
    price_series = price_series.dropna()
    df_flow = pd.DataFrame(index=price_series.index)
    df_flow["Price"] = price_series
    df_flow["Add_Units"] = 0.0
    for d in sorted(dates):
        idx_loc = price_series.index.asof(d)
        if idx_loc is not None and price_series.loc[idx_loc] > 0:
            if idx_loc in df_flow.index:
                df_flow.at[idx_loc, "Add_Units"] += dca_amount / price_series.loc[idx_loc]
    df_flow["Cumulative_Units"] = df_flow["Add_Units"].cumsum()
    equity = df_flow["Cumulative_Units"] * df_flow["Price"]
    return equity[equity > 0]


# ══════════════════════════════════════════════════════════════════
# MÓDULO 7 — VALUATION ENGINE (DCF + DDM)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600 * 6)
def fetch_valuation_data(ticker_clean: str) -> dict:
    """Coleta dados fundamentais via yfinance para DCF/DDM."""
    yf_ticker = f"{ticker_clean}.SA"
    out = {k: np.nan for k in [
        "price", "ebitda", "revenue", "ebit", "net_income", "da",
        "capex", "total_debt", "total_cash", "shares", "eps",
        "dps", "payout", "roe", "book_value", "market_cap",
        "pe", "ev_ebitda", "net_margin", "op_margin", "op_cash_flow", "fcf",
    ]}
    out["ticker"] = ticker_clean
    out["error"] = None

    try:
        t = yf.Ticker(yf_ticker)
        info = t.info

        out["price"]      = _safe(info.get("currentPrice") or info.get("previousClose") or info.get("regularMarketPrice"))
        out["ebitda"]     = _safe(info.get("ebitda"))
        out["revenue"]    = _safe(info.get("totalRevenue"))
        out["total_debt"] = _safe(info.get("totalDebt"))
        out["total_cash"] = _safe(info.get("totalCash"))
        out["shares"]     = _safe(info.get("sharesOutstanding"))
        out["eps"]        = _safe(info.get("trailingEps"))
        out["roe"]        = _safe(info.get("returnOnEquity"))
        out["payout"]     = _safe(info.get("payoutRatio"))
        out["dps"]        = _safe(info.get("dividendRate"))
        out["book_value"] = _safe(info.get("bookValue"))
        out["market_cap"] = _safe(info.get("marketCap"))
        out["pe"]         = _safe(info.get("trailingPE"))
        out["ev_ebitda"]  = _safe(info.get("enterpriseToEbitda"))
        out["net_margin"] = _safe(info.get("profitMargins"))
        out["op_margin"]  = _safe(info.get("operatingMargins"))

        # Cash flow statement
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                col = cf.columns[0]
                for idx in cf.index:
                    idx_str = str(idx).lower()
                    if ("capital expenditure" in idx_str or "purchase of ppe" in idx_str
                            or "capital" in idx_str and "expenditure" in idx_str):
                        out["capex"] = abs(_safe(cf.loc[idx, col]))
                    if "operating" in idx_str and ("cash" in idx_str or "activities" in idx_str):
                        out["op_cash_flow"] = _safe(cf.loc[idx, col])
                    if "depreciation" in idx_str or ("amortization" in idx_str and "depletion" not in idx_str):
                        out["da"] = abs(_safe(cf.loc[idx, col]))
        except Exception:
            pass

        # Income statement
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                col = fin.columns[0]
                for idx in fin.index:
                    idx_str = str(idx).lower()
                    if "ebit" == idx_str or ("ebit" in idx_str and "ebitda" not in idx_str and "margin" not in idx_str):
                        out["ebit"] = _safe(fin.loc[idx, col])
                    if "net income" in idx_str or ("net" in idx_str and "income" in idx_str):
                        out["net_income"] = _safe(fin.loc[idx, col])
        except Exception:
            pass

        # Derivadas
        if not np.isnan(out["op_cash_flow"]) and not np.isnan(out["capex"]):
            out["fcf"] = out["op_cash_flow"] - out["capex"]
        if np.isnan(out["da"]) and not np.isnan(out["ebitda"]) and not np.isnan(out["ebit"]):
            out["da"] = max(out["ebitda"] - out["ebit"], 0)
        if np.isnan(out["ebit"]) and not np.isnan(out["ebitda"]) and not np.isnan(out["da"]):
            out["ebit"] = out["ebitda"] - out["da"]

    except Exception as e:
        out["error"] = str(e)

    return out


def _safe(val):
    """Converte para float seguro ou NaN."""
    if val is None:
        return np.nan
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan


def run_dcf(ticker: str, data: dict, scenario: str = "moderate",
            override: dict = None) -> dict:
    """
    DCF via FCFF + Gordon Growth Model para perpetuidade.
    Retorna preço justo, TIR, margem de segurança e projeções.
    """
    info  = VALUATION_UNIVERSE.get(ticker, {})
    sector = info.get("sector", "Industrial")
    p = dict(SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Industrial"]))
    if override:
        p.update(override)

    is_cons = scenario == "conservative"
    wacc   = p["wacc_cons"] if is_cons else p["wacc_mod"]
    g_perp = p["g_cons"]    if is_cons else p["g_mod"]
    tax    = p.get("tax", 0.34)
    capex_rev = p.get("capex_rev", 0.10)
    rev_growth = p.get("rev_growth_cons" if is_cons else "rev_growth_mod", [0.05]*5)

    ebitda   = _safe(data.get("ebitda"))
    revenue  = _safe(data.get("revenue"))
    da       = _safe(data.get("da"))
    capex    = _safe(data.get("capex"))
    debt     = _safe(data.get("total_debt")) or 0
    cash     = _safe(data.get("total_cash")) or 0
    shares   = _safe(data.get("shares"))
    price    = _safe(data.get("price"))

    if any(np.isnan(x) or x <= 0 for x in [ebitda, revenue, shares]):
        return {"error": "Dados financeiros insuficientes para DCF", "scenario": scenario}

    # Fallbacks
    if np.isnan(capex) or capex <= 0:
        capex = revenue * capex_rev
    if np.isnan(da) or da <= 0:
        da = ebitda * 0.12
    ebit = ebitda - da
    ebitda_margin = ebitda / revenue

    # Compressão de margem no conservador: -0.5pp por ano (acumulativo)
    margin_adj_annual = -0.005 if is_cons else 0.0

    projecoes = []
    rev_t = revenue
    margin_t = ebitda_margin

    for yr, g_rev in enumerate(rev_growth, 1):
        rev_t = rev_t * (1 + g_rev)
        margin_t = max(margin_t + margin_adj_annual, 0.05)  # acumula -0.5pp/ano
        ebitda_t = rev_t * margin_t
        da_t = da * ((rev_t / revenue) ** 0.65)
        ebit_t = ebitda_t - da_t
        nopat = ebit_t * (1 - tax)
        capex_t = rev_t * capex_rev
        fcff = nopat + da_t - capex_t
        projecoes.append({
            "Ano": f"Ano {yr}", "Receita": rev_t, "EBITDA": ebitda_t,
            "EBIT": ebit_t, "NOPAT": nopat, "D&A": da_t,
            "CapEx": capex_t, "FCFF": fcff,
        })

    if wacc <= g_perp:
        return {"error": f"WACC ({wacc:.1%}) ≤ g ({g_perp:.1%}). Ajuste os parâmetros.", "scenario": scenario}

    fcff_5 = projecoes[-1]["FCFF"]
    tv = fcff_5 * (1 + g_perp) / (wacc - g_perp)

    pv_fcff = sum(proj["FCFF"] / (1 + wacc) ** (i + 1) for i, proj in enumerate(projecoes))
    pv_tv   = tv / (1 + wacc) ** 5
    ev      = pv_fcff + pv_tv
    net_debt = debt - cash
    eq_val  = ev - net_debt
    fair_price = eq_val / shares if shares > 0 else np.nan

    irr = np.nan
    if not np.isnan(price) and not np.isnan(fair_price) and price > 0 and fair_price > 0:
        irr = (fair_price / price) ** (1 / 5) - 1

    mos = (fair_price - price) / price if not np.isnan(fair_price) and not np.isnan(price) and price > 0 else np.nan

    return {
        "error": None, "scenario": scenario, "ticker": ticker,
        "wacc": wacc, "g": g_perp, "tax": tax,
        "ebitda_base": ebitda, "revenue_base": revenue,
        "projecoes": projecoes,
        "pv_fcff": pv_fcff, "terminal_value": tv, "pv_tv": pv_tv,
        "enterprise_value": ev, "net_debt": net_debt, "equity_value": eq_val,
        "fair_price": fair_price, "current_price": price,
        "irr": irr, "margin_of_safety": mos,
        "tv_pct": pv_tv / ev if ev > 0 else np.nan,
    }


def run_ddm(ticker: str, data: dict, scenario: str = "moderate",
            override: dict = None) -> dict:
    """
    DDM (Dividendos Descontados) para bancos e seguradoras.
    Usa Gordon multi-estágio com crescimento de EPS projetado.
    """
    info   = VALUATION_UNIVERSE.get(ticker, {})
    sector = info.get("sector", "Banks")
    p = dict(SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Banks"]))
    if override:
        p.update(override)

    is_cons = scenario == "conservative"
    ke       = p["ke_cons"]  if is_cons else p["ke_mod"]
    g_perp   = p["g_cons"]   if is_cons else p["g_mod"]
    payout   = p.get("payout", 0.40)
    eps_g    = p.get("eps_growth_cons" if is_cons else "eps_growth_mod", 0.07)
    if is_cons:
        payout *= 0.85

    eps    = _safe(data.get("eps"))
    price  = _safe(data.get("price"))
    roe    = _safe(data.get("roe"))
    bv     = _safe(data.get("book_value"))
    data_payout = _safe(data.get("payout"))

    # Recalcula EPS se necessário
    if (np.isnan(eps) or eps <= 0) and not np.isnan(roe) and not np.isnan(bv) and bv > 0:
        eps = roe * bv

    if np.isnan(eps) or eps <= 0:
        return {"error": "EPS insuficiente para DDM. Verifique os dados.", "scenario": scenario}

    if not np.isnan(data_payout) and 0.05 < data_payout < 1.0:
        payout = data_payout

    projecoes = []
    eps_t = eps
    for yr in range(1, 6):
        eps_t = eps_t * (1 + eps_g)
        dps_t = eps_t * payout
        projecoes.append({"Ano": f"Ano {yr}", "EPS": eps_t, "DPS": dps_t})

    if ke <= g_perp:
        return {"error": f"Ke ({ke:.1%}) ≤ g ({g_perp:.1%}). Ajuste os parâmetros.", "scenario": scenario}

    dps_term = projecoes[-1]["DPS"] * (1 + g_perp)
    tv       = dps_term / (ke - g_perp)
    pv_dps   = sum(p_["DPS"] / (1 + ke) ** (i + 1) for i, p_ in enumerate(projecoes))
    pv_tv    = tv / (1 + ke) ** 5
    fair_price = pv_dps + pv_tv

    irr = np.nan
    if not np.isnan(price) and price > 0 and fair_price > 0:
        irr = (fair_price / price) ** (1 / 5) - 1

    mos = (fair_price - price) / price if not np.isnan(price) and price > 0 else np.nan

    return {
        "error": None, "scenario": scenario, "ticker": ticker,
        "ke": ke, "g": g_perp, "payout": payout, "eps_base": eps,
        "projecoes": projecoes,
        "pv_dps": pv_dps, "terminal_value": tv, "pv_tv": pv_tv,
        "fair_price": fair_price, "current_price": price,
        "irr": irr, "margin_of_safety": mos,
        "tv_pct": pv_tv / fair_price if fair_price > 0 else np.nan,
    }


def run_valuation(ticker: str, data: dict, scenario: str = "moderate",
                  override: dict = None) -> dict:
    """Despacha para DCF ou DDM conforme o modelo do ativo."""
    model = VALUATION_UNIVERSE.get(ticker, {}).get("model", "dcf")
    if model == "ddm":
        return run_ddm(ticker, data, scenario, override)
    return run_dcf(ticker, data, scenario, override)


def compute_sensitivity(ticker: str, data: dict) -> pd.DataFrame:
    """
    Análise de sensibilidade: WACC/Ke (linhas) × g (colunas).
    Retorna DataFrame com preços justos para combinações.
    """
    info   = VALUATION_UNIVERSE.get(ticker, {})
    sector = info.get("sector", "Industrial")
    model  = info.get("model", "dcf")
    p      = SECTOR_PARAMS.get(sector, SECTOR_PARAMS["Industrial"])

    if model == "dcf":
        base_r = p.get("wacc_mod", 0.10)
        base_g = p.get("g_mod", 0.045)
        r_label = "WACC"
        r_key   = "wacc_mod"
    else:
        base_r = p.get("ke_mod", 0.165)
        base_g = p.get("g_mod", 0.055)
        r_label = "Ke"
        r_key   = "ke_mod"

    r_range = np.round(np.arange(base_r - 0.02, base_r + 0.025, 0.005), 4)
    g_range = np.round(np.arange(base_g - 0.015, base_g + 0.020, 0.005), 4)

    matrix = {}
    for r in r_range:
        row = {}
        for g in g_range:
            if r <= g:
                row[f"{g:.1%}"] = np.nan
                continue
            ov = {r_key: float(r), "g_mod": float(g)}
            res = run_valuation(ticker, data, "moderate", ov)
            row[f"{g:.1%}"] = round(res.get("fair_price", np.nan), 2)
        matrix[f"{r:.1%}"] = row

    df = pd.DataFrame(matrix).T
    df.index.name = f"{r_label} →"
    return df


def build_summary_table(all_results: dict) -> pd.DataFrame:
    """Constrói tabela-resumo com todos os tickers e cenários."""
    rows = []
    for ticker, res_dict in all_results.items():
        cons = res_dict.get("conservative", {})
        mod  = res_dict.get("moderate", {})
        info = VALUATION_UNIVERSE[ticker]

        if cons.get("error") or mod.get("error"):
            continue

        price  = mod.get("current_price", np.nan)
        fp_c   = cons.get("fair_price", np.nan)
        fp_m   = mod.get("fair_price", np.nan)
        irr    = mod.get("irr", np.nan)
        mos    = mod.get("margin_of_safety", np.nan)

        # Recomendação
        if not np.isnan(mos):
            if mos > 0.25:
                rec = "🟢 COMPRA FORTE"
            elif mos > 0.10:
                rec = "🔵 COMPRA"
            elif mos > -0.05:
                rec = "🟡 NEUTRO"
            elif mos > -0.20:
                rec = "🟠 VENDA PARCIAL"
            else:
                rec = "🔴 VENDA"
        else:
            rec = "⚪ N/D"

        rows.append({
            "Ticker": ticker,
            "Empresa": info["name"],
            "Setor": info["sector"],
            "Modelo": info["model"].upper(),
            "Preço Atual": price,
            "P. Justo Conservador": fp_c,
            "P. Justo Moderado": fp_m,
            "Upside / Downside": mos,
            "TIR Implícita (5a)": irr,
            "Recomendação": rec,
        })

    return pd.DataFrame(rows).set_index("Ticker")


# ══════════════════════════════════════════════════════════════════
# MÓDULO 8 — RENDERIZAÇÃO DA ABA VALUATION
# ══════════════════════════════════════════════════════════════════

def render_valuation_tab():
    st.markdown("""
    <style>
    .val-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1a1f2e; border: 1px solid #2d3348;
        border-radius: 10px; padding: 1rem;
        text-align: center; color: white;
    }
    .badge-buy    { background: #0d6e3f; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.8em; }
    .badge-sell   { background: #8b1a1a; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.8em; }
    .badge-neutral{ background: #5a4e0d; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.8em; }
    </style>
    <div class="val-header">
        <h2 style="margin:0">💼 Valuation DCF / DDM — Análise Fundamentalista</h2>
        <p style="margin:0.3rem 0 0 0; opacity:0.8; font-size:0.9em">
        FCFF descontado (DCF) para não-financeiros · Dividendos descontados (DDM) para bancos/seguros ·
        Gordon Growth Model para perpetuidade · Análise de sensibilidade WACC × g
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar controls para valuation ──────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.header("5. Parâmetros Valuation")
        ipca_assumption = st.slider("IPCA Projetado (%)", 3.0, 8.0, 4.5, 0.5) / 100
        selic_base      = st.slider("Selic / Rf (%)", 8.0, 16.0, 10.75, 0.25) / 100
        erp_brazil      = st.slider("Prêmio Risco Brasil (ERP %)", 5.0, 12.0, 7.5, 0.5) / 100
        proj_years      = st.selectbox("Horizonte Projeção", [3, 5], index=1)
        refresh_val     = st.button("🔄 Atualizar Dados Valuation", type="secondary")

    if refresh_val:
        # Limpa cache para forçar re-fetch
        fetch_valuation_data.clear()

    # ── Carregamento de dados ─────────────────────────────────────
    tickers_list = list(VALUATION_UNIVERSE.keys())

    with st.spinner("Carregando dados fundamentais dos 16 ativos..."):
        prog = st.progress(0)
        all_data = {}
        for i, tk in enumerate(tickers_list):
            all_data[tk] = fetch_valuation_data(tk)
            prog.progress((i + 1) / len(tickers_list))
        prog.empty()

    # ── Computação de valuation ───────────────────────────────────
    all_results = {}
    for tk in tickers_list:
        all_results[tk] = {
            "conservative": run_valuation(tk, all_data[tk], "conservative"),
            "moderate":     run_valuation(tk, all_data[tk], "moderate"),
        }

    summary_df = build_summary_table(all_results)

    # ══════════════════════════════════════════════════════════════
    # PAINEL GERAL — tabela resumo de todos os ativos
    # ══════════════════════════════════════════════════════════════
    sub1, sub2 = st.tabs(["📊 Painel Geral", "🔍 Análise Individual"])

    with sub1:
        st.subheader("Resumo de Valuation — 16 Ativos")
        st.caption(
            f"Premissas macro: IPCA {ipca_assumption:.1%} · Selic {selic_base:.2%} · "
            f"ERP Brasil {erp_brazil:.1%}. Dados atualizados em tempo real via YFinance."
        )

        if not summary_df.empty:
            def _color_upside(val):
                if isinstance(val, float):
                    if val > 0.25:     return "background-color: #0d3d22; color: #7fffc1"
                    elif val > 0.10:   return "background-color: #1a4a2e; color: #adffd8"
                    elif val > -0.05:  return "background-color: #3a3a10; color: #ffd97d"
                    elif val > -0.20:  return "background-color: #4a2010; color: #ffb07d"
                    else:              return "background-color: #3d0d0d; color: #ff8888"
                return ""

            styled = (
                summary_df.style
                .format({
                    "Preço Atual": "R$ {:.2f}",
                    "P. Justo Conservador": "R$ {:.2f}",
                    "P. Justo Moderado": "R$ {:.2f}",
                    "Upside / Downside": "{:+.1%}",
                    "TIR Implícita (5a)": "{:.1%}",
                }, na_rep="N/D")
                .applymap(_color_upside, subset=["Upside / Downside"])
                .background_gradient(subset=["TIR Implícita (5a)"], cmap="RdYlGn", vmin=-0.10, vmax=0.25)
            )
            st.dataframe(styled, use_container_width=True, height=520)

            # ── Gráfico de barras: upside por ativo ───────────────
            plot_df = summary_df.reset_index()[
                ["Ticker", "Empresa", "Upside / Downside", "Recomendação"]
            ].dropna()
            plot_df["Upside %"] = plot_df["Upside / Downside"] * 100
            plot_df["Cor"] = plot_df["Upside %"].apply(
                lambda x: "#22c55e" if x > 10 else ("#ef4444" if x < -5 else "#eab308")
            )

            fig_up = go.Figure(go.Bar(
                x=plot_df["Ticker"],
                y=plot_df["Upside %"],
                marker_color=plot_df["Cor"],
                text=[f"{v:+.1f}%" for v in plot_df["Upside %"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Upside: %{y:.1f}%<extra></extra>",
            ))
            fig_up.update_layout(
                title="Upside / Downside vs Preço Justo Moderado (%)",
                xaxis_title="Ticker",
                yaxis_title="Upside (%)",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font_color="white",
                height=380,
                yaxis=dict(zeroline=True, zerolinecolor="#555", gridcolor="#1f2535"),
            )
            st.plotly_chart(fig_up, use_container_width=True)

            # ── Scatter: TIR × Upside ─────────────────────────────
            sc_df = summary_df.reset_index().dropna(subset=["TIR Implícita (5a)", "Upside / Downside"])
            if not sc_df.empty:
                fig_sc = px.scatter(
                    sc_df, x="Upside / Downside", y="TIR Implícita (5a)",
                    text="Ticker", color="Setor",
                    title="TIR Implícita (5 anos) × Upside Moderado",
                    labels={"Upside / Downside": "Upside/Downside", "TIR Implícita (5a)": "TIR (5a)"},
                    template="plotly_dark",
                )
                fig_sc.update_traces(textposition="top center")
                fig_sc.update_layout(height=420)
                st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.error("Não foi possível computar o valuation. Verifique a conexão com os dados.")

    # ══════════════════════════════════════════════════════════════
    # ANÁLISE INDIVIDUAL
    # ══════════════════════════════════════════════════════════════
    with sub2:
        sel_ticker = st.selectbox(
            "Selecione o ativo para análise detalhada:",
            tickers_list,
            format_func=lambda t: f"{t} — {VALUATION_UNIVERSE[t]['name']}",
        )

        info    = VALUATION_UNIVERSE[sel_ticker]
        data_t  = all_data[sel_ticker]
        res_c   = all_results[sel_ticker]["conservative"]
        res_m   = all_results[sel_ticker]["moderate"]
        sp      = SECTOR_PARAMS.get(info["sector"], {})

        # ── Diagnóstico ───────────────────────────────────────────
        st.markdown(f"### 🏢 {info['name']} ({sel_ticker}) — {info['sector']}")
        st.info(sp.get("description", ""), icon="ℹ️")

        # Métricas de cabeçalho
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        price_cur = _safe(data_t.get("price"))
        fp_mod    = res_m.get("fair_price", np.nan)
        fp_con    = res_c.get("fair_price", np.nan)
        irr_mod   = res_m.get("irr", np.nan)
        mos_mod   = res_m.get("margin_of_safety", np.nan)
        ev_ebitda_t = _safe(data_t.get("ev_ebitda"))

        c1.metric("Preço Atual", f"R$ {price_cur:.2f}" if not np.isnan(price_cur) else "N/D")
        c2.metric("P. Justo Moderado",
                  f"R$ {fp_mod:.2f}" if not np.isnan(fp_mod) else "N/D",
                  delta=f"{mos_mod:+.1%}" if not np.isnan(mos_mod) else None)
        c3.metric("P. Justo Conservador", f"R$ {fp_con:.2f}" if not np.isnan(fp_con) else "N/D")
        c4.metric("TIR Implícita (5a)", f"{irr_mod:.1%}" if not np.isnan(irr_mod) else "N/D")
        c5.metric("EV/EBITDA", f"{ev_ebitda_t:.1f}x" if not np.isnan(ev_ebitda_t) else "N/D")

        wacc_or_ke = res_m.get("wacc") or res_m.get("ke")
        c6.metric("WACC / Ke", f"{wacc_or_ke:.1%}" if wacc_or_ke else "N/D")

        # Erros
        if res_m.get("error"):
            st.error(f"⚠️ Cenário Moderado: {res_m['error']}")
        if res_c.get("error"):
            st.warning(f"⚠️ Cenário Conservador: {res_c['error']}")

        # ── Dados Financeiros Brutos ──────────────────────────────
        with st.expander("📋 Dados Financeiros (TTM / Último Exercício)", expanded=False):
            def fmt_brl(v, div=1e9, sfx="B"):
                if np.isnan(v): return "N/D"
                return f"R$ {v/div:.2f}{sfx}"

            fin_cols = st.columns(4)
            fin_cols[0].metric("Receita Líquida",  fmt_brl(_safe(data_t.get("revenue"))))
            fin_cols[1].metric("EBITDA",            fmt_brl(_safe(data_t.get("ebitda"))))
            fin_cols[2].metric("Lucro Líquido",     fmt_brl(_safe(data_t.get("net_income"))))
            fin_cols[3].metric("FCF",               fmt_brl(_safe(data_t.get("fcf"))))

            fin_cols2 = st.columns(4)
            fin_cols2[0].metric("Dívida Bruta",     fmt_brl(_safe(data_t.get("total_debt"))))
            fin_cols2[1].metric("Caixa",            fmt_brl(_safe(data_t.get("total_cash"))))
            nd = (_safe(data_t.get("total_debt")) or 0) - (_safe(data_t.get("total_cash")) or 0)
            fin_cols2[2].metric("Dívida Líquida",   fmt_brl(nd))
            fin_cols2[3].metric("CapEx",            fmt_brl(_safe(data_t.get("capex"))))

            fin_cols3 = st.columns(4)
            roe_v = _safe(data_t.get("roe"))
            nm_v  = _safe(data_t.get("net_margin"))
            om_v  = _safe(data_t.get("op_margin"))
            fin_cols3[0].metric("ROE",             f"{roe_v:.1%}" if not np.isnan(roe_v) else "N/D")
            fin_cols3[1].metric("Margem Líquida",  f"{nm_v:.1%}"  if not np.isnan(nm_v)  else "N/D")
            fin_cols3[2].metric("Margem EBIT",     f"{om_v:.1%}"  if not np.isnan(om_v)  else "N/D")
            ebitda_v = _safe(data_t.get("ebitda"))
            rev_v    = _safe(data_t.get("revenue"))
            ebitda_m = ebitda_v / rev_v if not any(np.isnan(x) for x in [ebitda_v, rev_v]) and rev_v > 0 else np.nan
            fin_cols3[3].metric("Margem EBITDA",   f"{ebitda_m:.1%}" if not np.isnan(ebitda_m) else "N/D")

        # ── Projeções Financeiras ─────────────────────────────────
        st.subheader("📈 Projeções — Cenário Moderado vs Conservador")

        def _make_proj_df(res):
            if res.get("error") or not res.get("projecoes"):
                return pd.DataFrame()
            projs = res["projecoes"]
            model = VALUATION_UNIVERSE[sel_ticker]["model"]
            rows  = []
            for p_ in projs:
                if model == "ddm":
                    rows.append({"Período": p_["Ano"], "EPS (R$)": p_["EPS"], "DPS (R$)": p_["DPS"]})
                else:
                    rows.append({
                        "Período": p_["Ano"],
                        "Receita (R$ M)": p_["Receita"] / 1e6,
                        "EBITDA (R$ M)":  p_["EBITDA"]  / 1e6,
                        "FCFF (R$ M)":    p_["FCFF"]    / 1e6,
                        "CapEx (R$ M)":   p_["CapEx"]   / 1e6,
                    })
            return pd.DataFrame(rows).set_index("Período")

        col_mod, col_con = st.columns(2)
        df_mod = _make_proj_df(res_m)
        df_con = _make_proj_df(res_c)

        with col_mod:
            st.caption("🔵 Cenário Moderado")
            if not df_mod.empty:
                st.dataframe(df_mod.style.format("{:.1f}"), use_container_width=True)
        with col_con:
            st.caption("🟠 Cenário Conservador")
            if not df_con.empty:
                st.dataframe(df_con.style.format("{:.1f}"), use_container_width=True)

        # ── Waterfall DCF ─────────────────────────────────────────
        if not res_m.get("error") and info["model"] == "dcf" and res_m.get("projecoes"):
            st.subheader("🏗️ Decomposição do Valor — Waterfall (Moderado)")
            pv_list = [
                proj["FCFF"] / (1 + res_m["wacc"]) ** (i + 1)
                for i, proj in enumerate(res_m["projecoes"])
            ]
            wf_labels = [p_["Ano"] for p_ in res_m["projecoes"]] + ["Valor Terminal", "(-) Dívida Líquida"]
            wf_vals   = [v / 1e9 for v in pv_list] + [
                res_m["pv_tv"] / 1e9,
                -res_m["net_debt"] / 1e9,
            ]
            colors = ["#3b82f6"] * len(pv_list) + ["#8b5cf6", "#ef4444"]

            fig_wf = go.Figure(go.Bar(
                x=wf_labels, y=wf_vals,
                marker_color=colors,
                text=[f"R$ {v:.1f}B" for v in wf_vals],
                textposition="outside",
            ))
            fig_wf.update_layout(
                title="Componentes do Valor da Firma (R$ Bilhões)",
                yaxis_title="R$ Bilhões",
                template="plotly_dark",
                height=380,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        # ── Comparativo de Cenários ───────────────────────────────
        st.subheader("🆚 Comparativo de Preços Justos")
        comp_data = {
            "Cenário": ["Conservador", "Moderado", "Preço Atual"],
            "Preço (R$)": [
                fp_con if not np.isnan(fp_con) else 0,
                fp_mod if not np.isnan(fp_mod) else 0,
                price_cur if not np.isnan(price_cur) else 0,
            ],
            "Cor": ["#f59e0b", "#3b82f6", "#10b981"],
        }
        fig_comp = go.Figure(go.Bar(
            x=comp_data["Cenário"],
            y=comp_data["Preço (R$)"],
            marker_color=comp_data["Cor"],
            text=[f"R$ {v:.2f}" for v in comp_data["Preço (R$)"]],
            textposition="outside",
        ))
        fig_comp.update_layout(
            title="Preço Justo por Cenário vs Cotação",
            yaxis_title="Preço (R$)",
            template="plotly_dark",
            height=360,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # ── Análise de Sensibilidade ──────────────────────────────
        st.subheader("🎯 Stress Test — Sensibilidade WACC/Ke × g")
        st.caption("Preços justos (R$) para diferentes combinações de taxa de desconto e crescimento na perpetuidade.")

        with st.spinner("Calculando mapa de sensibilidade..."):
            df_sens = compute_sensitivity(sel_ticker, data_t)

        if not df_sens.empty:
            # Colore baseado na distância do preço atual
            def _heat_color(val):
                if np.isnan(val) or np.isnan(price_cur) or price_cur == 0:
                    return ""
                ratio = val / price_cur
                if ratio > 1.30:     return "background-color: #0d3d22; color: #7fffc1"
                elif ratio > 1.10:   return "background-color: #1a4a2e; color: #adffd8"
                elif ratio > 0.95:   return "background-color: #3a3a10; color: #ffd97d"
                elif ratio > 0.80:   return "background-color: #4a2010; color: #ffb07d"
                else:                return "background-color: #3d0d0d; color: #ff8888"

            st.dataframe(
                df_sens.style
                .format("R$ {:.2f}", na_rep="—")
                .applymap(_heat_color),
                use_container_width=True,
            )
            st.caption(
                "🟢 > +30% upside · 🔵 > +10% · 🟡 ±5% (neutro) · 🟠 < -5% · 🔴 < -20% (downside) "
                f"| Preço atual: R$ {price_cur:.2f}"
            )

            # Heatmap Plotly
            try:
                z = df_sens.values.astype(float)
                fig_hm = go.Figure(go.Heatmap(
                    z=z,
                    x=df_sens.columns.tolist(),
                    y=df_sens.index.tolist(),
                    colorscale="RdYlGn",
                    zmin=price_cur * 0.6 if not np.isnan(price_cur) else None,
                    zmax=price_cur * 1.6 if not np.isnan(price_cur) else None,
                    text=[[f"R$ {v:.2f}" if not np.isnan(v) else "—" for v in row] for row in z],
                    texttemplate="%{text}",
                    colorbar=dict(title="Preço Justo"),
                ))
                r_label = "Ke" if info["model"] == "ddm" else "WACC"
                fig_hm.update_layout(
                    title=f"Heatmap: Preço Justo por {r_label} × g",
                    xaxis_title="Taxa de Crescimento Perpetuidade (g)",
                    yaxis_title=r_label,
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            except Exception:
                pass

        # ── Premissas e Notas ─────────────────────────────────────
        with st.expander("📌 Premissas Utilizadas", expanded=False):
            model_t = info["model"].upper()
            if model_t == "DCF":
                st.markdown(f"""
                **Modelo:** Fluxo de Caixa Livre para a Firma (FCFF)  
                **WACC Moderado:** {res_m.get('wacc', 0):.1%} · **WACC Conservador:** {res_c.get('wacc', 0):.1%}  
                **g (Moderado):** {res_m.get('g', 0):.1%} · **g (Conservador):** {res_c.get('g', 0):.1%}  
                **Alíquota IR/CSLL:** {sp.get('tax', 0.34):.0%}  
                **CapEx / Receita:** {sp.get('capex_rev', 0):.0%}  
                **Perpetuidade:** Gordon Growth → TV = FCFF₅ × (1+g) / (WACC − g)  
                **Conservador:** crescimento de receita = IPCA only, margem comprimindo  
                **Moderado:** crescimento setorial + pipeline de projetos anunciados  
                """)
            else:
                st.markdown(f"""
                **Modelo:** Dividendos Descontados (DDM) — aplicável a bancos e seguradoras  
                **Ke Moderado:** {res_m.get('ke', 0):.1%} · **Ke Conservador:** {res_c.get('ke', 0):.1%}  
                **g (Moderado):** {res_m.get('g', 0):.1%} · **g (Conservador):** {res_c.get('g', 0):.1%}  
                **Payout Ratio utilizado:** {res_m.get('payout', 0):.0%}  
                **Justificativa:** FCFF não aplicável a financeiros; capital regulado pelo Banco Central/SUSEP  
                """)


# ══════════════════════════════════════════════════════════════════
# MÓDULO 9 — MAIN (UNIFICADO)
# ══════════════════════════════════════════════════════════════════

def main():
    # CSS customizado global
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: #0e1117; }
    [data-testid="stSidebar"] { background: #161b27; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9em; font-weight: 600; }
    .stMetric label { font-size: 0.78em !important; color: #94a3b8 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧪 Quant Factor Lab Pro v4.0 | DCF Valuation Engine")
    st.markdown(
        "**Motor Híbrido Quant + Fundamental** — "
        "Ranking multifator institucional · DCF/DDM detalhado · Análise de sensibilidade · Monte Carlo"
    )

    # ── Sidebar: Parâmetros Quant ─────────────────────────────────
    st.sidebar.header("1. Universo e Dados")
    default_univ = (
        "ITUB3, TOTS3, TAEE3, BBSE3, WEGE3, PSSA3, EGIE3, VIVT3, "
        "PRIO3, BBAS3, BPAC3, SBSP3, SAPR3, CPFE3, ALUP3, BMOB3, "
        "CXSE3, B3SA3, MDIA3, AGRO3"
    )
    ticker_input = st.sidebar.text_area("Tickers (sem .SA)", default_univ, height=100)
    raw_tickers  = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    yf_tickers   = [f"{t}.SA" for t in raw_tickers]

    st.sidebar.header("2. Pesos do Ranking")
    w_rm   = st.sidebar.slider("Residual Momentum",      0.0, 1.0, 0.40)
    w_val  = st.sidebar.slider("Value (P/L, P/VP, EV)", 0.0, 1.0, 0.40)
    w_qual = st.sidebar.slider("Quality (ROE, Margem)",  0.0, 1.0, 0.20)

    st.sidebar.header("3. Parâmetros de Gestão")
    top_n          = st.sidebar.number_input("Número de Ativos", 4, 30, 10)
    use_vol_target = st.sidebar.checkbox("Risk Parity (Inv Vol)", True)

    st.sidebar.header("4. Backtest & Monte Carlo")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 100000, 2000)
    dca_years  = st.sidebar.slider("Anos de Histórico", 2, 10, 5)
    mc_years   = st.sidebar.slider("Projeção Futura (Anos)", 1, 20, 5)

    run_btn = st.sidebar.button("🚀 Executar Análise Quant", type="primary")

    # ── Tabs principais ───────────────────────────────────────────
    tab_rank, tab_dca, tab_bench, tab_hist, tab_mc, tab_raw, tab_val = st.tabs([
        "🏆 Ranking Atual",
        "📈 Performance DCA",
        "🆚 Benchmarks",
        "💰 Custódia",
        "🔮 Monte Carlo",
        "📋 Dados Brutos",
        "💼 Valuation DCF",
    ])

    # ── Aba Valuation (sempre disponível) ────────────────────────
    with tab_val:
        render_valuation_tab()

    # ── Análise Quant (requer botão) ──────────────────────────────
    if not run_btn:
        for tab in [tab_rank, tab_dca, tab_bench, tab_hist, tab_mc, tab_raw]:
            with tab:
                st.info("▶ Configure os parâmetros na sidebar e clique em **Executar Análise Quant**.")
        return

    if not raw_tickers:
        st.error("Insira pelo menos um ticker.")
        return

    with st.status("Processando Pipeline Quantitativo...", expanded=True) as status:
        end_date   = datetime.now()
        start_total = end_date - timedelta(days=365 * (dca_years + 3))
        start_bt    = end_date - timedelta(days=365 * dca_years)

        status.write("📥 Baixando preços históricos (YFinance)...")
        prices = fetch_price_data(yf_tickers, start_total.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if prices.empty:
            st.error("Falha ao baixar preços.")
            status.update(label="Erro", state="error")
            return

        status.write("🔍 Consultando fundamentos (Brapi + YF fallback)...")
        fundamentals = fetch_fundamentals_hybrid(raw_tickers, BRAPI_TOKEN)
        if not fundamentals.empty:
            status.write(f"✅ Fundamentos para {len(fundamentals)} ativos.")
        else:
            status.write("⚠️ Sem fundamentos — usando somente Momentum.")

        status.write("🧮 Calculando scores...")
        curr_mom  = compute_residual_momentum_enhanced(prices)
        curr_val  = compute_value_robust(fundamentals) if not fundamentals.empty else pd.Series(0, index=prices.columns)
        curr_qual = compute_quality_score(fundamentals) if not fundamentals.empty else pd.Series(0, index=prices.columns)

        df_master = pd.DataFrame(index=prices.columns)
        df_master["Res_Mom"] = curr_mom
        df_master["Value"]   = curr_val
        df_master["Quality"] = curr_qual
        if not fundamentals.empty and "sector" in fundamentals.columns:
            df_master["Sector"] = fundamentals["sector"]
        df_master.dropna(thresh=1, inplace=True)

        df_master["Composite_Score"] = 0.0
        for col, weight in [("Res_Mom", w_rm), ("Value", w_val), ("Quality", w_qual)]:
            if col in df_master.columns:
                z = robust_zscore(df_master[col])
                df_master[f"{col}_Z"] = z
                df_master["Composite_Score"] += z * weight
        df_master = df_master.sort_values("Composite_Score", ascending=False)

        status.write("⚙️ Rodando backtest robusto...")
        dca_curve, dca_transactions, dca_holdings = run_dca_backtest_robust(
            prices, top_n, dca_amount, use_vol_target, start_bt, end_date
        )

        status.update(label="✅ Análise Concluída!", state="complete", expanded=False)

    # ── Benchmarks ────────────────────────────────────────────────
    bench_curves = {}
    if not dca_transactions.empty and not dca_curve.empty:
        dca_dates = sorted(set(pd.to_datetime(dca_transactions["Date"]).tolist()))
        for bench in ["BOVA11.SA", "DIVO11.SA"]:
            if bench in prices.columns:
                bc = run_benchmark_dca(prices[bench], dca_dates, dca_amount)
                common = dca_curve.index.intersection(bc.index)
                if not common.empty:
                    bench_curves[bench] = bc.loc[common]

    # ── Tab: Ranking ──────────────────────────────────────────────
    with tab_rank:
        st.subheader("🎯 Carteira Recomendada (Hoje)")
        top_picks = df_master.head(top_n).copy()
        latest    = prices.iloc[-1]
        top_picks["Preço Atual"] = latest.reindex(top_picks.index)
        sug_w = construct_portfolio(top_picks, prices.tail(90), top_n, use_vol_target)
        top_picks["Peso (%)"]      = sug_w * 100
        top_picks["Alocação (R$)"] = sug_w * dca_amount
        top_picks["Qtd Sugerida"]  = top_picks["Alocação (R$)"] / top_picks["Preço Atual"]

        cols_show = ["Sector", "Preço Atual", "Composite_Score", "Peso (%)", "Alocação (R$)", "Qtd Sugerida", "Value", "Quality"]
        st.dataframe(
            top_picks[[c for c in cols_show if c in top_picks.columns]].style.format({
                "Preço Atual": "R$ {:.2f}", "Composite_Score": "{:.2f}",
                "Value": "{:.2f}", "Quality": "{:.2f}",
                "Peso (%)": "{:.1f}%", "Alocação (R$)": "R$ {:.0f}", "Qtd Sugerida": "{:.0f}",
            }).background_gradient(subset=["Composite_Score"], cmap="Greens"),
            use_container_width=True, height=400,
        )
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.plotly_chart(
                px.pie(values=sug_w, names=sug_w.index, title="Alocação Sugerida"),
                use_container_width=True,
            )
        with col_p2:
            if "Sector" in top_picks.columns:
                st.plotly_chart(
                    px.pie(top_picks, names="Sector", values="Peso (%)", title="Exposição Setorial"),
                    use_container_width=True,
                )

    # ── Tab: DCA ──────────────────────────────────────────────────
    with tab_dca:
        st.subheader("Simulação de Acumulação (DCA)")
        if not dca_curve.empty:
            end_val = dca_curve.iloc[-1, 0]
            u_months = pd.to_datetime(dca_transactions["Date"]).dt.to_period("M").nunique() if not dca_transactions.empty else 0
            total_inv = u_months * dca_amount
            profit    = end_val - total_inv
            roi       = profit / total_inv if total_inv > 0 else 0
            m1, m2, m3 = st.columns(3)
            m1.metric("Patrimônio Final", f"R$ {end_val:,.2f}")
            m2.metric("Total Investido",  f"R$ {total_inv:,.2f}")
            m3.metric("Lucro Líquido",    f"R$ {profit:,.2f}", delta=f"{roi:.1%}")
            st.plotly_chart(px.line(dca_curve, title="Curva de Patrimônio (Estratégia)"), use_container_width=True)
            metrics = calculate_advanced_metrics(dca_curve["Strategy_DCA"])
            st.subheader("Métricas de Risco")
            m_cols = st.columns(4)
            for i, (k, v) in enumerate(metrics.items()):
                fmt = f"{v:.1%}" if k not in ("Sharpe", "Sortino", "Calmar", "Ulcer Index") else f"{v:.2f}"
                m_cols[i % 4].metric(k, fmt)

    # ── Tab: Benchmarks ───────────────────────────────────────────
    with tab_bench:
        st.subheader("🆚 Estratégia vs Benchmarks")
        if not dca_curve.empty and bench_curves:
            df_cmp = dca_curve.copy()
            for b, s in bench_curves.items():
                df_cmp[b] = s
            df_cmp = df_cmp.ffill().dropna()
            st.plotly_chart(px.line(df_cmp, title="Evolução Patrimonial Comparativa"), use_container_width=True)
            comp_rows = []
            m_s = calculate_advanced_metrics(df_cmp["Strategy_DCA"])
            m_s["Asset"] = "🚀 Estratégia"
            m_s["Saldo Final"] = df_cmp["Strategy_DCA"].iloc[-1]
            comp_rows.append(m_s)
            for bn in bench_curves:
                if bn in df_cmp.columns:
                    m_b = calculate_advanced_metrics(df_cmp[bn])
                    m_b["Asset"] = bn
                    m_b["Saldo Final"] = df_cmp[bn].iloc[-1]
                    comp_rows.append(m_b)
            df_cm = pd.DataFrame(comp_rows).set_index("Asset")
            st.dataframe(
                df_cm[["Saldo Final", "Retorno Total", "CAGR", "Volatilidade", "Sharpe", "Max Drawdown"]].style.format({
                    "Saldo Final": "R$ {:,.2f}", "Retorno Total": "{:.1%}",
                    "CAGR": "{:.1%}", "Volatilidade": "{:.1%}",
                    "Sharpe": "{:.2f}", "Max Drawdown": "{:.1%}",
                }).highlight_max(subset=["Saldo Final"], color="#d4edda"),
                use_container_width=True,
            )
        else:
            st.warning("Dados insuficientes para comparação.")

    # ── Tab: Custódia ─────────────────────────────────────────────
    with tab_hist:
        h1, h2 = st.columns([1, 1])
        final_df = pd.DataFrame()
        with h1:
            st.subheader("💰 Posição Final (Backtest)")
            if dca_holdings:
                final_df = pd.DataFrame.from_dict(dca_holdings, orient="index", columns=["Qtd"])
                if not dca_curve.empty:
                    last_d = dca_curve.index[-1]
                    if last_d in prices.index:
                        final_df["Preço"] = prices.loc[last_d].reindex(final_df.index)
                        final_df["Valor (R$)"] = final_df["Qtd"] * final_df["Preço"]
                        nav = final_df["Valor (R$)"].sum()
                        final_df["Peso (%)"] = final_df["Valor (R$)"] / nav * 100
                        final_df = final_df.sort_values("Peso (%)", ascending=False)
                        st.dataframe(
                            final_df.style.format({"Qtd": "{:.0f}", "Preço": "R$ {:.2f}",
                                                   "Valor (R$)": "R$ {:,.2f}", "Peso (%)": "{:.1f}%"}),
                            use_container_width=True,
                        )
                        st.metric("Patrimônio em Custódia", f"R$ {nav:,.2f}")
        with h2:
            st.subheader("📊 Alocação Final")
            if not final_df.empty and "Valor (R$)" in final_df.columns:
                st.plotly_chart(
                    px.pie(final_df, values="Valor (R$)", names=final_df.index, hole=0.4),
                    use_container_width=True,
                )
        if not dca_transactions.empty:
            st.divider()
            st.subheader("Histórico de Transações")
            st.dataframe(dca_transactions.sort_values("Date", ascending=False), use_container_width=True)

    # ── Tab: Monte Carlo ──────────────────────────────────────────
    with tab_mc:
        st.subheader("Projeção Probabilística")
        if not dca_curve.empty:
            daily_r = dca_curve["Strategy_DCA"].pct_change().dropna()
            if not daily_r.empty:
                mu    = daily_r.mean() * 252
                sigma = daily_r.std() * np.sqrt(252)
                sim   = run_monte_carlo(dca_curve.iloc[-1, 0], dca_amount, mu, sigma, mc_years)
                if not sim.empty:
                    st.plotly_chart(
                        px.line(sim, title=f"Cone de Probabilidade — {mc_years} Anos"),
                        use_container_width=True,
                    )

    # ── Tab: Dados Brutos ─────────────────────────────────────────
    with tab_raw:
        st.subheader("Dados Fundamentais (Brapi + YF)")
        if not fundamentals.empty:
            st.dataframe(fundamentals)
            st.caption("Fonte: Brapi.dev com fallback automático para Yahoo Finance.")
        else:
            st.error("Falha nos fundamentos. Verifique token ou conexão.")


if __name__ == "__main__":
    main()
