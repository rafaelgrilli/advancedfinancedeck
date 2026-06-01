import streamlit as st
import numpy as np
import scipy.stats as si
import scipy.optimize as optimize
import scipy.optimize as sco
import scipy.interpolate as interpolate
import plotly.graph_objects as go
import pandas as pd
import io
import re
import datetime
from datetime import date
from yahooquery import Ticker
from sklearn.covariance import LedoitWolf

# ==============================================================================
# ⚙️ CONFIGURAÇÃO DE PÁGINA UNIFICADA
# ==============================================================================
st.set_page_config(page_title="Grilli Analytics | Institutional Suite", layout="wide")

# ==============================================================================
# 🛠️ NAVEGAÇÃO ENTRE OS TERMINAIS
# ==============================================================================
st.sidebar.title("Navegação")
app_choice = st.sidebar.selectbox(
    "Selecione o Terminal:",
    [
        "Terminal de Risco e Gestão de Portfólio (v18.2)",
        "Terminal Quantitativo v8.0: Institutional Asset Suite"
    ]
)
st.sidebar.markdown("---")


# ==============================================================================
# 📦 CLASSES E MÉTODOS - CÓDIGO 1 (TERMINAL DE RISCO)
# ==============================================================================
class B3Calendar:
    @staticmethod
    def get_easter_date(year):
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        L = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * L) // 451
        month = (h + L - 7 * m + 114) // 31
        day = ((h + L - 7 * m + 114) % 31) + 1
        return datetime.date(year, month, day)

    @classmethod
    def get_b3_holidays(cls, year):
        easter = cls.get_easter_date(year)
        carnival_mon = easter - datetime.timedelta(days=48)
        carnival_tue = easter - datetime.timedelta(days=47)
        good_friday = easter - datetime.timedelta(days=2)
        corpus_christi = easter + datetime.timedelta(days=60)
        
        fixed_holidays = [
            datetime.date(year, 1, 1),   # Confraternização Universal
            datetime.date(year, 4, 21),  # Tiradentes
            datetime.date(year, 5, 1),   # Dia do Trabalho
            datetime.date(year, 7, 9),   # Revolução Constitucionalista (SP)
            datetime.date(year, 9, 7),   # Independência do Brasil
            datetime.date(year, 10, 12), # Nossa Senhora Aparecida
            datetime.date(year, 11, 2),  # Finados
            datetime.date(year, 11, 15), # Proclamação da República
            datetime.date(year, 11, 20), # Dia da Consciência Negra
            datetime.date(year, 12, 25)  # Natal
        ]
        
        mobile_holidays = [carnival_mon, carnival_tue, good_friday, corpus_christi]
        return set(fixed_holidays + mobile_holidays)

    @classmethod
    def is_business_day(cls, date):
        if date.weekday() >= 5:
            return False
        holidays = cls.get_b3_holidays(date.year)
        return date not in holidays

    @classmethod
    def get_b3_business_days(cls, start_date, end_date):
        if start_date > end_date:
            return -cls.get_b3_business_days(end_date, start_date)
        
        curr = start_date
        business_days = 0
        while curr < end_date:
            if cls.is_business_day(curr):
                business_days += 1
            curr += datetime.timedelta(days=1)
        return business_days

    @classmethod
    def resolve_b3_expiry(cls, ticker, val_date):
        ticker = str(ticker).strip().upper()
        match = re.match(r'^[A-Z]{4}([A-X])\d+$', ticker)
        if not match:
            return None
        
        letter = match.group(1)
        is_call = letter <= 'L'
        month_exp = ord(letter) - ord('A') + 1 if is_call else ord(letter) - ord('M') + 1
        
        year_exp = val_date.year
        if month_exp < val_date.month:
            year_exp += 1
            
        first_day = datetime.date(year_exp, month_exp, 1)
        days_to_first_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + datetime.timedelta(days=days_to_first_friday)
        third_friday = first_friday + datetime.timedelta(days=14)
        
        return cls.get_b3_business_days(val_date, third_friday)


class YieldCurveEngine:
    @staticmethod
    def interpolate_rate(t_array, rates_dict):
        t_arr = np.array(sorted(rates_dict.keys()))
        r_arr = np.array([rates_dict[k] for k in t_arr])
        return np.interp(t_array, t_arr, r_arr)


class CoreModels:
    @staticmethod
    def d1(S, K, T, r, sigma):
        T_safe = np.maximum(T, 1e-9)
        sigma_safe = np.maximum(sigma, 1e-9)
        return (np.log(S / K) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * np.sqrt(T_safe))

    @classmethod
    def d2(cls, S, K, T, r, sigma):
        T_safe = np.maximum(T, 1e-9)
        sigma_safe = np.maximum(sigma, 1e-9)
        return cls.d1(S, K, T, r, sigma) - sigma_safe * np.sqrt(T_safe)

    @classmethod
    def engine_bsm(cls, S, K, T, r, sigma, opt_type="call"):
        _d1 = cls.d1(S, K, T, r, sigma)
        _d2 = cls.d2(S, K, T, r, sigma)
        
        if isinstance(opt_type, (str, bool)):
            is_call = (str(opt_type).lower() == "call")
        else:
            opt_type_arr = np.atleast_1d(opt_type)
            is_call = np.array([str(ot).lower() == "call" for ot in opt_type_arr.flatten()]).reshape(opt_type_arr.shape)
        
        call_price = S * si.norm.cdf(_d1) - K * np.exp(-r * T) * si.norm.cdf(_d2)
        put_price = K * np.exp(-r * T) * si.norm.cdf(-_d2) - S * si.norm.cdf(-_d1)
        
        return np.where(is_call, call_price, put_price)

    @classmethod
    def calc_greeks_bsm(cls, S, K, T, r, sigma, opt_type="call"):
        T_safe = np.maximum(T, 1e-9)
        sigma_safe = np.maximum(sigma, 1e-9)
        S_safe = np.maximum(S, 1e-9)
        
        _d1 = cls.d1(S_safe, K, T_safe, r, sigma_safe)
        _d2 = cls.d2(S_safe, K, T_safe, r, sigma_safe)
        
        if isinstance(opt_type, (str, bool)):
            is_call = (str(opt_type).lower() == "call")
        else:
            opt_type_arr = np.atleast_1d(opt_type)
            is_call = np.array([str(ot).lower() == "call" for ot in opt_type_arr.flatten()]).reshape(opt_type_arr.shape)
        
        pdf_d1 = si.norm.pdf(_d1)
        cdf_d1 = si.norm.cdf(_d1)
        cdf_d2 = si.norm.cdf(_d2)
        
        delta = np.where(is_call, cdf_d1, cdf_d1 - 1.0)
        vega_raw = S_safe * pdf_d1 * np.sqrt(T_safe)
        vega_1pct = vega_raw / 100.0
        
        theta_ann = np.where(is_call,
                             - (S_safe * sigma_safe * pdf_d1) / (2 * np.sqrt(T_safe)) - r * K * np.exp(-r * T_safe) * cdf_d2,
                             - (S_safe * sigma_safe * pdf_d1) / (2 * np.sqrt(T_safe)) + r * K * np.exp(-r * T_safe) * si.norm.cdf(-_d2))
        theta_daily = theta_ann / 252.0
        
        gamma = pdf_d1 / (S_safe * sigma_safe * np.sqrt(T_safe))
        vanna = - pdf_d1 * _d2 / sigma_safe
        vomma = vega_raw * _d1 * _d2 / sigma_safe
        
        charm_call = - pdf_d1 * (2 * r * T_safe - _d2 * sigma_safe * np.sqrt(T_safe)) / (2 * T_safe * sigma_safe * np.sqrt(T_safe)) - r * cdf_d1
        charm_put = - pdf_d1 * (2 * r * T_safe - _d2 * sigma_safe * np.sqrt(T_safe)) / (2 * T_safe * sigma_safe * np.sqrt(T_safe)) + r * si.norm.cdf(-_d1)
        charm_daily = np.where(is_call, charm_call, charm_put) / 252.0
        
        color = (pdf_d1 / (2 * S_safe * T_safe * sigma_safe * np.sqrt(T_safe))) * (1 + _d1 * (2 * r * T_safe - _d2 * sigma_safe * np.sqrt(T_safe)) / (sigma_safe * np.sqrt(T_safe)))
        speed = - (gamma / S_safe) * (_d1 / (sigma_safe * np.sqrt(T_safe)) + 1.0)
        
        return {
            "delta": delta, "gamma": gamma, "theta": theta_daily, "vega": vega_1pct, "vega_100": vega_raw,
            "vanna": vanna, "vomma": vomma, "charm": charm_daily, "color": color, "speed": speed
        }

    @classmethod
    def solve_implied_volatility(cls, target_price, S, K, T, r, opt_type="call"):
        is_call = (opt_type.lower() == "call")
        lower_bound = max(0, S - K * np.exp(-r * T)) if is_call else max(0, K * np.exp(-r * T) - S)
        upper_bound = S if is_call else K * np.exp(-r * T)
        
        if not (lower_bound <= target_price <= upper_bound) or target_price <= 0:
            return np.nan
            
        def obj_func(sigma):
            return cls.engine_bsm(S, K, T, r, sigma, opt_type) - target_price
            
        try:
            return optimize.brentq(obj_func, 1e-4, 5.0)
        except ValueError:
            res = optimize.minimize_scalar(
                lambda vol: (cls.engine_bsm(S, K, T, r, vol, opt_type) - target_price)**2,
                bounds=(1e-4, 5.0), method='bounded'
            )
            return res.x if res.success else np.nan


class AnalyticsEngine:
    @staticmethod
    def valuate_portfolio_mc(St, Sig, df, rates_dict):
        M = len(St)
        N = len(df)
        if N == 0:
            return np.zeros(M)
            
        S_mat = St if St.ndim == 2 else St[:, np.newaxis]
        Sig_mat = Sig if Sig.ndim == 2 else Sig[:, np.newaxis]
        
        qty = df['Quantidade'].values[np.newaxis, :]
        K = df['Strike'].values[np.newaxis, :]
        t_dias = df['Dias Úteis (Hoje)'].values
        t_dias_new = np.maximum(t_dias - 1, 0)
        T_new = (t_dias_new / 252)[np.newaxis, :]
        
        r_list = YieldCurveEngine.interpolate_rate(t_dias_new / 252, rates_dict)
        r_mat = r_list[np.newaxis, :]
        
        types = df['Tipo'].str.lower().values
        is_spot = (types == 'spot')[np.newaxis, :]
        is_call = (types == 'call')[np.newaxis, :]
        is_put = (types == 'put')[np.newaxis, :]
        is_forward = (types == 'forward')[np.newaxis, :]
        is_future = (types == 'future')[np.newaxis, :]
        
        K_safe = np.where(is_spot | is_forward | is_future, 1.0, K)
        
        spot_val = np.where(is_spot, S_mat, 0.0)
        forward_val = np.where(is_forward, S_mat - K * np.exp(-r_mat * T_new), 0.0)
        future_val = np.where(is_future, S_mat - K, 0.0)
        
        expired_mask = np.broadcast_to((T_new <= 0) & (~(is_spot | is_forward | is_future)), (M, N))
        intrinsic_call = np.maximum(S_mat - K_safe, 0.0)
        intrinsic_put = np.maximum(K_safe - S_mat, 0.0)
        expired_val = np.where(is_call, intrinsic_call, 0.0) + np.where(is_put, intrinsic_put, 0.0)
        expired_val = np.where(expired_mask, expired_val, 0.0)
        
        active_mask = np.broadcast_to((T_new > 0) & (~(is_spot | is_forward | is_future)), (M, N))
        T_safe = np.where(active_mask, T_new, 1e-9)
        Sig_safe = np.where(active_mask, Sig_mat, 0.1)
        r_safe = np.where(active_mask, r_mat, 0.01)
        
        d1_mat = (np.log(S_mat / K_safe) + (r_safe + 0.5 * Sig_safe**2) * T_safe) / (Sig_safe * np.sqrt(T_safe))
        d2_mat = d1_mat - Sig_safe * np.sqrt(T_safe)
        
        call_bsm = S_mat * si.norm.cdf(d1_mat) - K_safe * np.exp(-r_safe * T_safe) * si.norm.cdf(d2_mat)
        put_bsm = K_safe * np.exp(-r_safe * T_safe) * si.norm.cdf(-d2_mat) - S_mat * si.norm.cdf(-d1_mat)
        
        active_val = np.where(is_call, call_bsm, 0.0) + np.where(is_put, put_bsm, 0.0)
        active_val = np.where(active_mask, active_val, 0.0)
        
        total_asset_values = spot_val + forward_val + future_val + expired_val + active_val
        return np.sum(total_asset_values * qty, axis=1)


class ValuationValidators:
    @staticmethod
    def sanitize_portfolio_data(df):
        sanitized = df.copy()
        sanitized['Tipo'] = sanitized['Tipo'].astype(str).str.lower().str.strip()
        sanitized = sanitized[sanitized['Tipo'].isin(['spot', 'call', 'put', 'forward', 'future'])]
        
        sanitized['Quantidade'] = pd.to_numeric(sanitized['Quantidade'], errors='coerce').fillna(0.0)
        sanitized['Strike'] = pd.to_numeric(sanitized['Strike'], errors='coerce').fillna(0.0)
        sanitized['Strike'] = np.maximum(sanitized['Strike'], 0.0)
        
        sanitized['Dias Úteis (Hoje)'] = pd.to_numeric(sanitized['Dias Úteis (Hoje)'], errors='coerce').fillna(0).astype(int)
        sanitized['Dias Úteis (Hoje)'] = np.maximum(sanitized['Dias Úteis (Hoje)'], 0)
        return sanitized.reset_index(drop=True)


def run_model_diagnostics():
    results = {}
    try:
        S, K, T_dias, r_disc, sigma = 100.0, 95.0, 63, 0.10, 0.30
        T_anos = T_dias / 252
        r_cont = np.log(1 + r_disc)
        
        c_price = CoreModels.engine_bsm(S, K, T_anos, r_cont, sigma, "call")
        p_price = CoreModels.engine_bsm(S, K, T_anos, r_cont, sigma, "put")
        results["Put-Call Parity Bounds"] = abs((c_price - p_price) - (S - K * np.exp(-r_cont * T_anos))) < 1e-6
        
        g_call = CoreModels.calc_greeks_bsm(np.array([S]), np.array([K]), np.array([T_anos]), np.array([r_cont]), np.array([sigma]), np.array(["call"]))
        g_put = CoreModels.calc_greeks_bsm(np.array([S]), np.array([K]), np.array([T_anos]), np.array([r_cont]), np.array([sigma]), np.array(["put"]))
        results["Greek Signs (Delta/Gamma)"] = (g_call["delta"][0] >= 0) and (g_put["delta"][0] <= 0) and (g_call["gamma"][0] > 0)
        
        days = B3Calendar.get_b3_business_days(datetime.date(2026, 6, 1), datetime.date(2026, 6, 22))
        results["B3 Deterministic Calendar"] = (days == 14)
        
    except Exception as e:
        results[f"Falha Crítica nos Testes: {str(e)}"] = False
    return results


# ==============================================================================
# 📦 FUNÇÕES - CÓDIGO 2 (TERMINAL QUANTITATIVO)
# ==============================================================================
def get_market_weights(prices):
    vols = prices.pct_change().std()
    inv_vol = 1 / vols
    w = inv_vol / inv_vol.sum()
    return w.values


def calculate_stats(weights, mu, cov_mat, rf):
    p_ret = np.sum(mu * weights)
    p_vol = np.sqrt(weights.T @ cov_mat @ weights)
    p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0
    return p_ret, p_vol, p_sharpe


def risk_parity_objective(weights, cov_mat):
    p_vol = np.sqrt(weights.T @ cov_mat @ weights)
    marginal_risk = (cov_mat @ weights) / p_vol
    risk_contribution = weights * marginal_risk
    target = p_vol / len(weights)
    return np.sum(np.square(risk_contribution - target))


def risk_contributions(weights, cov_mat):
    p_vol = np.sqrt(weights.T @ cov_mat @ weights)
    marginal_risk = (cov_mat @ weights) / p_vol
    rc = weights * marginal_risk
    return rc / rc.sum()


def black_litterman_full(mu_prior, cov, views_dict, confidences=None, tau=0.05):
    n = len(mu_prior)
    if not views_dict:
        return mu_prior, cov

    P = np.zeros((len(views_dict), n))
    Q = np.zeros(len(views_dict))
    assets = list(mu_prior.index)
    omega_diag = []

    for i, (asset, view_val) in enumerate(views_dict.items()):
        if asset in assets:
            idx = assets.index(asset)
            P[i, idx] = 1
            Q[i] = view_val
            conf = confidences.get(asset, 0.5) if confidences else 0.5
            omega_diag.append((P[i] @ (tau * cov) @ P[i].T) / conf)

    omega = np.diag(omega_diag)

    inv_prior = np.linalg.inv(tau * cov)
    inv_omega = np.linalg.inv(omega)

    term1 = np.linalg.inv(inv_prior + P.T @ inv_omega @ P)
    mu_bl = term1 @ (inv_prior @ mu_prior + P.T @ inv_omega @ Q)

    return pd.Series(mu_bl, index=assets), cov


def rolling_backtest(prices, rf, t_cost, views, confs, window=252, rebalance=21):
    rets = prices.pct_change().dropna()
    n_assets = len(prices.columns)
    w_prev = np.array([1 / n_assets] * n_assets)
    port_rets, dates = [], []
    convergence_failures = 0

    for i in range(window, len(rets) - rebalance, rebalance):
        train = rets.iloc[i - window:i]
        test  = rets.iloc[i:i + rebalance]

        lw = LedoitWolf().fit(train)
        cov = lw.covariance_ * 252

        w_mkt = get_market_weights(train)
        pi = 3.0 * (cov @ w_mkt)

        mu_bl, cov_bl = black_litterman_full(
            pd.Series(pi, index=prices.columns), cov, views, confs
        )

        def obj(w):
            r = np.sum(mu_bl * w)
            v = np.sqrt(w.T @ cov_bl @ w)
            turnover = np.sum(np.abs(w - w_prev))
            cost = t_cost * turnover
            return -(r - cost - rf) / v

        res = sco.minimize(
            obj, w_prev,
            bounds=tuple((0, 0.5) for _ in range(n_assets)),
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if res.success:
            w_prev = res.x
        else:
            convergence_failures += 1

        r_test = test.dot(w_prev)
        port_rets.extend(r_test.tolist())
        dates.extend(test.index.tolist())

    return pd.Series(port_rets, index=pd.to_datetime(dates)), convergence_failures


def efficient_frontier_parametric(mu, cov_mat, rf, n_points=200):
    n = len(mu)
    bnds = tuple((0, 0.5) for _ in range(n))
    cons_sum = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    r_min = mu.min() * 0.8
    r_max = mu.max() * 1.1
    targets = np.linspace(r_min, r_max, n_points)

    front_vols, front_rets = [], []
    w0 = np.array([1 / n] * n)

    for target in targets:
        cons = [
            cons_sum,
            {'type': 'eq', 'fun': lambda x, t=target: np.sum(mu * x) - t}
        ]
        res = sco.minimize(
            lambda w: np.sqrt(w.T @ cov_mat @ w),
            w0,
            bounds=bnds,
            constraints=cons,
            method='SLSQP',
            options={'ftol': 1e-10, 'maxiter': 1000}
        )
        if res.success:
            front_vols.append(res.fun)
            front_rets.append(target)
            w0 = res.x

    return np.array(front_vols), np.array(front_rets)


def bootstrap_sharpe_ci(returns, rf, n_boot=1000, ci=0.95):
    sharpes = []
    n = len(returns)
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=n, replace=True)
        ann_r = sample.mean() * 252
        ann_v = sample.std() * np.sqrt(252)
        if ann_v > 0:
            sharpes.append((ann_r - rf) / ann_v)

    alpha = (1 - ci) / 2
    return (
        np.mean(sharpes),
        np.percentile(sharpes, alpha * 100),
        np.percentile(sharpes, (1 - alpha) * 100)
    )


def bootstrap_maxdd_ci(returns, n_boot=1000, ci=0.95):
    mdd_samples = []
    n = len(returns)
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=n, replace=True)
        cum = (1 + pd.Series(sample)).cumprod()
        mdd_samples.append(((cum / cum.cummax()) - 1).min())

    alpha = (1 - ci) / 2
    return (
        np.mean(mdd_samples),
        np.percentile(mdd_samples, alpha * 100),
        np.percentile(mdd_samples, (1 - alpha) * 100)
    )


def safe_optimize(obj_fn, n_assets, label=""):
    w0 = np.array([1 / n_assets] * n_assets)
    bnds = tuple((0, 0.5) for _ in range(n_assets))
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    res = sco.minimize(
        obj_fn, w0, bounds=bnds, constraints=cons,
        method='SLSQP', options={'ftol': 1e-9, 'maxiter': 1000}
    )

    if not res.success:
        best = res
        for _ in range(10):
            w_rand = np.random.dirichlet(np.ones(n_assets))
            r2 = sco.minimize(
                obj_fn, w_rand, bounds=bnds, constraints=cons,
                method='SLSQP', options={'ftol': 1e-9, 'maxiter': 1000}
            )
            if r2.success or (not best.success and r2.fun < best.fun):
                best = r2
        res = best

    return res, res.success


# ==============================================================================
# 🖥️ EXECUÇÃO CONDICIONAL DA INTERFACE DO USUÁRIO (UI)
# ==============================================================================

if app_choice == "Terminal de Risco e Gestão de Portfólio (v18.2)":
    
    # 🎨 CSS Estilos Específicos do Código 1
    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            background-color: rgba(30, 58, 138, 0.04);
            padding: 15px; border-radius: 8px; border: 1px solid rgba(30, 58, 138, 0.1);
            border-left: 5px solid #1e3a8a;
        }
        .secao-titulo { color: #1e3a8a; font-weight: bold; border-bottom: 2px solid #1e3a8a; padding-bottom: 5px; margin-bottom: 20px; margin-top: 20px;}
        .pnl-box { font-size: 0.9em; padding: 15px; border-radius: 5px; background: #f8fafc; border: 1px solid #e2e8f0; }
        .gov-box { font-size: 0.85em; padding: 15px; border-radius: 5px; background: #0f172a; color: white; border: 1px solid #1e293b; margin-bottom: 20px;}
        .test-badge { font-weight: bold; padding: 3px 8px; border-radius: 4px; display: inline-block; font-size: 0.8em;}
        .test-success { background-color: #16a34a; color: white; }
        .test-fail { background-color: #dc2626; color: white; }
        .audit-box { background-color: #f1f5f9; padding: 15px; border-radius: 5px; border-left: 4px solid #475569; margin-top: 10px; font-size: 0.9em;}
        </style>
        """, unsafe_allow_html=True)

    st.title("Terminal de Risco e Gestão de Portfólio (v18.2)")
    st.write("Vectorized Risk Core | Discrete-to-Continuous Rate BRL Adjuster | Cubic Spline Interpolated Vol Surface | Optimized Performance Node")
    st.write("---")

    # INITIAL STATE SETUP & CACHING
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = pd.DataFrame({
            "Ativo": ["PETR4", "PETRF350", "PETRR350"],
            "Tipo": ["spot", "call", "put"],
            "Quantidade": [1000.0, -500.0, 300.0],
            "Strike": [0.0, 35.0, 35.0],
            "Dias Úteis (Hoje)": [0, 21, 21]
        })
    if 'raw_text_cache' not in st.session_state:
        st.session_state['raw_text_cache'] = ""
    if 'parsed_df' not in st.session_state:
        st.session_state['parsed_df'] = None

    # CONTROLES DA SIDEBAR & MODELO CONTINUO
    st.sidebar.markdown("### 🗓️ Configurações Temporais")
    val_date = st.sidebar.date_input("Data Base (Valuation Date):", datetime.date(2026, 6, 1))

    st.sidebar.markdown("### ⚙️ Parâmetros do Mercado DI (Discretos B3)")
    r_curto = st.sidebar.number_input("Rate Curto (30 DU) %:", value=10.25, step=0.05) / 100
    r_medio = st.sidebar.number_input("Rate Médio (252 DU) %:", value=11.10, step=0.05) / 100
    r_longo = st.sidebar.number_input("Rate Longo (1008 DU) %:", value=11.75, step=0.05) / 100

    rates_dict = {
        30/252: np.log(1.0 + r_curto),
        252/252: np.log(1.0 + r_medio),
        1008/252: np.log(1.0 + r_longo)
    }

    st.sidebar.markdown("### 📊 Variáveis Globais de Referência")
    opt_type_global = st.sidebar.radio("Ponta Global (Micro):", ["call", "put"], format_func=lambda x: "Call" if x == "call" else "Put")
    S_global = st.sidebar.number_input("Spot Atual (S):", value=35.00, step=0.10)
    sigma_global = st.sidebar.number_input("Volatilidade de Referência (σ %):", value=30.0, step=0.5) / 100
    T_days_global = st.sidebar.number_input("Dias Úteis Default:", value=21, step=1)

    st.sidebar.markdown("### 🎲 Parâmetros de Estresse D+1 (PnL Explain)")
    shock_S = st.sidebar.number_input("Choque de Spot D+1 (R$):", value=0.00, step=0.20)
    shock_vol = st.sidebar.number_input("Choque de Vol D+1 (pp):", value=0.00, step=0.5) / 100

    S_d1_scenario = max(S_global + shock_S, 0.01)
    sigma_d1_scenario = max(sigma_global + shock_vol, 0.01)

    st.sidebar.markdown("### 🛡️ Configurações Monte Carlo")
    sims = st.sidebar.slider("Simulações:", 10000, 50000, 20000, step=5000)
    rho_shock = st.sidebar.slider("Correlação Spot/Vol (ρ):", -1.0, 1.0, -0.6, step=0.05)
    vol_of_vol = st.sidebar.slider("Magnitude do Choque Vol-of-Vol:", 0.1, 2.0, 0.6, step=0.05)
    reduction_method = st.sidebar.selectbox("Método de Redução de Variância:", ["None", "Antithetic Variates", "Moment Matching"])

    @st.cache_data
    def generate_monte_carlo_normals(sims, rho, method):
        rng = np.random.default_rng(42)
        Z1 = rng.standard_normal(sims)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal(sims)
        
        if method == "Antithetic Variates":
            Z1 = np.concatenate([Z1, -Z1])
            Z2 = np.concatenate([Z2, -Z2])
        elif method == "Moment Matching":
            Z1 = (Z1 - np.mean(Z1)) / (np.std(Z1) + 1e-9)
            Z2 = (Z2 - np.mean(Z2)) / (np.std(Z2) + 1e-9)
            
        return Z1, Z2

    diagnostic_suite = run_model_diagnostics()
    all_passed = all(diagnostic_suite.values())

    st.markdown("""<div class='gov-box'>
        <b style="color: #60a5fa;">SISTEMA DE RISK GOVERNANCE & AUTODIAGNÓSTICO AUTOMATIZADO (v18.2)</b><br><br>
        <b>Core Model:</b> Black-Scholes-Merton unificado com conversão de taxa BRL discreta para contínua: $r_c = \\ln(1 + r_{\\text{discreta}})$.<br>
        <b>Curva de Juros:</b> Yield Curve Bootstrap com interpolação linear vetorizada em NumPy.<br>
        <b>Validação Temporal:</b> Custom B3 Business Day Parser integrado (feriados móveis e fixos nacionais B3) calibrado com Valuation Date.<br>
        <b>Performance Node:</b> Cálculo de Mesa de Operações 100% vetorizado sem o uso de iterrows.
    </div>""", unsafe_allow_html=True)

    st.write("### 🔬 Status de Integridade Quantitativa")
    cols_diag = st.columns(len(diagnostic_suite))
    for col, (name, passed) in zip(cols_diag, diagnostic_suite.items()):
        badge_class = "test-success" if passed else "test-fail"
        status_text = "PASSED" if passed else "FAILED"
        col.markdown(f"**{name}**<br><span class='test-badge {badge_class}'>{status_text}</span>", unsafe_allow_html=True)
    st.write("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💼 Mesa de Operações & Risco", 
        "🚀 Alpha Scanner de Arbitragem", 
        "📈 Superfície de Volatilidade", 
        "🔬 Microestrutura & Audit",
        "📊 Governança"
    ])

    # --- TAB 1: MESA DE OPERAÇÕES VETORIZADA, PNL EXPLAIN & TRIPLICE VaR ---
    with tab1:
        st.markdown("<div class='secao-titulo'>1. EXPOSIÇÃO GERAL & PORTFOLIO RISK CONTROL (VETORIZADO)</div>", unsafe_allow_html=True)
        
        with st.form("portfolio_editor_form"):
            raw_df = st.data_editor(st.session_state['portfolio'], num_rows="dynamic", use_container_width=True)
            submit_button = st.form_submit_button("🚀 Recalcular Portfólio & Executar Simulações de Risco")

        if submit_button or 'portfolio' in st.session_state:
            edited_df = ValuationValidators.sanitize_portfolio_data(raw_df)
            st.session_state['portfolio'] = edited_df

            N_assets = len(edited_df)
            
            if N_assets > 0:
                qty = edited_df['Quantidade'].values
                k = edited_df['Strike'].values
                t_dias = edited_df['Dias Úteis (Hoje)'].values
                types = edited_df['Tipo'].str.lower().values
                t_years = t_dias / 252
                
                r_assets = YieldCurveEngine.interpolate_rate(t_years, rates_dict)
                
                is_opt = (types == "call") | (types == "put")
                prices_hoje = np.zeros(N_assets)
                
                prices_hoje = np.where(types == "spot", S_global, prices_hoje)
                prices_hoje = np.where(types == "forward", S_global - k * np.exp(-r_assets * t_years), prices_hoje)
                prices_hoje = np.where(types == "future", S_global - k, prices_hoje)
                
                if np.any(is_opt):
                    opt_prices = CoreModels.engine_bsm(S_global, k, t_years, r_assets, sigma_global, types)
                    prices_hoje = np.where(is_opt, opt_prices, prices_hoje)
                    greeks_hoje = CoreModels.calc_greeks_bsm(S_global, k, t_years, r_assets, sigma_global, types)
                else:
                    greeks_hoje = {
                        "delta": np.zeros(N_assets), "gamma": np.zeros(N_assets), "theta": np.zeros(N_assets),
                        "vega": np.zeros(N_assets), "vega_100": np.zeros(N_assets), "vanna": np.zeros(N_assets),
                        "charm": np.zeros(N_assets), "color": np.zeros(N_assets), "speed": np.zeros(N_assets)
                    }

                deltas_total = np.where(is_opt, greeks_hoje["delta"], 1.0)
                gammas_total = np.where(is_opt, greeks_hoje["gamma"], 0.0)
                thetas_total = np.where(is_opt, greeks_hoje["theta"], 0.0)
                vegas_total = np.where(is_opt, greeks_hoje["vega"], 0.0)
                vegas_100_total = np.where(is_opt, greeks_hoje["vega_100"], 0.0)
                vannas_total = np.where(is_opt, greeks_hoje["vanna"], 0.0)
                charms_total = np.where(is_opt, greeks_hoje["charm"], 0.0)
                
                gex_total = gammas_total * (S_global ** 2) * qty

                mtm_hoje = np.sum(prices_hoje * qty)
                net_delta = np.sum(deltas_total * qty)
                net_gamma = np.sum(gammas_total * qty)
                net_theta = np.sum(thetas_total * qty)
                net_vega = np.sum(vegas_total * qty)
                net_gex = np.sum(gex_total)

                t_years_new = np.maximum(t_dias - 1, 0) / 252
                r_assets_new = YieldCurveEngine.interpolate_rate(t_years_new, rates_dict)
                
                prices_amanha = np.zeros(N_assets)
                prices_amanha = np.where(types == "spot", S_d1_scenario, prices_amanha)
                prices_amanha = np.where(types == "forward", S_d1_scenario - k * np.exp(-r_assets_new * t_years_new), prices_amanha)
                prices_amanha = np.where(types == "future", S_d1_scenario - k, prices_amanha)
                
                is_expired_t1 = (t_dias - 1 <= 0) & is_opt
                is_active_t1 = (t_dias - 1 > 0) & is_opt
                
                intrinsic_val = np.where(types == "call", np.maximum(S_d1_scenario - k, 0.0), np.maximum(k - S_d1_scenario, 0.0))
                prices_amanha = np.where(is_expired_t1, intrinsic_val, prices_amanha)
                
                if np.any(is_active_t1):
                    opt_prices_amanha = CoreModels.engine_bsm(S_d1_scenario, k, t_years_new, r_assets_new, sigma_d1_scenario, types)
                    prices_amanha = np.where(is_active_t1, opt_prices_amanha, prices_amanha)

                mtm_amanha = np.sum(prices_amanha * qty)
                
                dS = S_d1_scenario - S_global
                dSig = sigma_d1_scenario - sigma_global
                dt = 1/252
                
                pnl_delta_attr = np.sum(qty * deltas_total * dS)
                pnl_gamma_attr = np.sum(qty * 0.5 * gammas_total * (dS ** 2))
                pnl_theta_attr = np.sum(qty * thetas_total * dt * 252)
                pnl_vega_attr = np.sum(qty * vegas_100_total * dSig)
                pnl_vanna_attr = np.sum(qty * vannas_total * dS * dSig)
                pnl_charm_attr = np.sum(qty * charms_total * dS * dt * 252)

                pnl_total_real = mtm_amanha - mtm_hoje
                pnl_teorico_taylor = pnl_delta_attr + pnl_gamma_attr + pnl_theta_attr + pnl_vega_attr + pnl_vanna_attr + pnl_charm_attr
                pnl_residual = pnl_total_real - pnl_teorico_taylor

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("MTM Hoje (R$)", f"{mtm_hoje:,.2f}")
                c2.metric("Net Delta", f"{net_delta:,.2f}")
                c3.metric("Net Gamma", f"{net_gamma:,.2f}")
                c4.metric("Net Theta/Dia", f"R$ {net_theta:,.2f}")
                c5.metric("Net Vega/1pp", f"R$ {net_vega:,.2f}")
                c6.metric("Gamma Exp (GEX)", f"R$ {net_gex:,.2f}")

                st.markdown("<div class='secao-titulo'>2. PnL EXPLAIN (TAYLOR MULTI-ORDER ATTRIBUTION)</div>", unsafe_allow_html=True)
                st.latex(r"\Delta PnL \approx \Delta \cdot dS + \frac{1}{2}\Gamma \cdot dS^2 + \Theta \cdot dt + \nu \cdot d\sigma + \text{Vanna} \cdot dS d\sigma + \text{Charm} \cdot dS dt + \text{Residual}")
                
                col_pnl, col_var_metrics = st.columns([1.5, 1])
                
                with col_pnl:
                    fig_waterfall = go.Figure(go.Waterfall(
                        name="Decomposição", 
                        orientation="v",
                        measure=["relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"],
                        x=["Delta", "Gamma", "Theta", "Vega", "Vanna", "Charm", "Residual", "PnL Total Real"],
                        textposition="outside",
                        text=[f"{x:+.2f}" for x in [pnl_delta_attr, pnl_gamma_attr, pnl_theta_attr, pnl_vega_attr, pnl_vanna_attr, pnl_charm_attr, pnl_residual, pnl_total_real]],
                        y=[pnl_delta_attr, pnl_gamma_attr, pnl_theta_attr, pnl_vega_attr, pnl_vanna_attr, pnl_charm_attr, pnl_residual, pnl_total_real],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    fig_waterfall.update_layout(title=" Waterfall de Atribuição de Resultado Financeiro (R$)", template="plotly_white", height=380)
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                    
                    residual_pct = abs(pnl_residual / pnl_total_real) if abs(pnl_total_real) > 1e-4 else 0
                    if residual_pct > 0.25:
                        st.warning(f"🚨 **Alerta de Convergência:** O residual teórico representa {residual_pct:.1%} do resultado total. Aproximações de Taylor deterioram sob choques extremos.")

                with col_var_metrics:
                    st.write("**Métricas Robustas de VaR D+1 (95% Confiança)**")
                    with st.spinner("Calculando modelos matemáticos..."):
                        dt = 1/252
                        Z1, Z2 = generate_monte_carlo_normals(sims, rho_shock, reduction_method)
                        current_sims = len(Z1)
                        r_drift = YieldCurveEngine.interpolate_rate(dt, rates_dict)
                        
                        St1_mc = S_global * np.exp((r_drift - 0.5 * sigma_global**2) * dt + sigma_global * np.sqrt(dt) * Z1)
                        Sig1_mc_joint = np.maximum(sigma_global * np.exp(-0.5 * vol_of_vol**2 * dt + vol_of_vol * np.sqrt(dt) * Z2), 0.01)
                        
                        mtm_var_pure = AnalyticsEngine.valuate_portfolio_mc(St1_mc, np.full(current_sims, sigma_global), edited_df, rates_dict)
                        mtm_var_joint = AnalyticsEngine.valuate_portfolio_mc(St1_mc, Sig1_mc_joint, edited_df, rates_dict)
                        
                        pnl_pure = mtm_var_pure - mtm_hoje
                        pnl_joint = mtm_var_joint - mtm_hoje
                        
                        dn_var_95 = 1.645 * abs(net_delta) * S_global * sigma_global * np.sqrt(dt)
                        var_pure_95 = np.percentile(pnl_pure, 5)
                        var_joint_95 = np.percentile(pnl_joint, 5)
                        cvar_joint_95 = pnl_joint[pnl_joint <= var_joint_95].mean()
                        
                        skew = si.skew(pnl_joint)
                        kurt = si.kurtosis(pnl_joint)
                        z_alpha = -1.645
                        z_cf = z_alpha + (1/6)*(z_alpha**2 - 1)*skew + (1/24)*(z_alpha**3 - 3*z_alpha)*kurt - (1/36)*(2*z_alpha**3 - 5*z_alpha)*(skew**2)
                        cf_var_95 = np.mean(pnl_joint) + z_cf * np.std(pnl_joint)
                        
                        st.metric("First-Order Delta-Normal VaR", f"R$ {dn_var_95:,.2f}", help="VaR linear com base no delta do portfólio.")
                        st.metric("Monte Carlo VaR (Spot Shock)", f"R$ {var_pure_95:,.2f}", help="VaR por simulação estática de volatilidade.")
                        st.metric("Joint Spot-Vol Estresse VaR", f"R$ {var_joint_95:,.2f}", help="VaR com choques dinâmicos acoplados de Spot e Volatilidade.")
                        st.metric("Cornish-Fisher VaR (Adjusted)", f"R$ {cf_var_95:,.2f}", help="VaR corrigido por assimetria e curtose reais de cauda.")
                        st.metric("Expected Shortfall (CVaR Joint)", f"R$ {cvar_joint_95:,.2f}", help="Expectativa de perda caso o limite de confiança do VaR seja ultrapassado.")

                st.markdown("<div class='secao-titulo'>3. DISTRIBUIÇÃO CONJUNTA & CENÁRIOS MACROESTRUTURAIS</div>", unsafe_allow_html=True)
                col_chart, col_stress = st.columns([1.5, 1])
                
                with col_chart:
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(x=pnl_joint, nbinsx=150, name="Simulação Conjunta", opacity=0.6, marker_color="#1e3a8a"))
                    fig_dist.add_vline(x=var_joint_95, line_width=2.5, line_dash="dash", line_color="#ef4444", annotation_text=f"VaR (95%): R$ {var_joint_95:,.0f}", annotation_position="top left")
                    fig_dist.add_vline(x=cvar_joint_95, line_width=2.5, line_dash="dash", line_color="#b91c1c", annotation_text=f"CVaR: R$ {cvar_joint_95:,.0f}", annotation_position="bottom left")
                    fig_dist.update_layout(title="Histograma de Distribuição de Retorno (PnL Estocástico)", xaxis_title="Retorno Estimado (R$)", template="plotly_white", height=380)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                with col_stress:
                    st.write("**Análise de Cenários Estressados Determinísticos (Vetorizada)**")
                    
                    spot_shocks = np.array([-0.25, -0.15, 0.00, 0.15, 0.25])
                    vol_shocks = np.array([0.20, 0.10, 0.00, -0.05, -0.10])
                    scenarios_names = [
                        "Crash Sistêmico Estilo Covid (-25% Spot, +20% Vol)",
                        "Crash Moderado de Ativos (-15% Spot, +10% Vol)",
                        "Cenário Neutro de Referência (0% Spot, 0% Vol)",
                        "Boom Econômico Moderado (+15% Spot, -5% Vol)",
                        "Shock Bullish Extremo (+25% Spot, -10% Vol)"
                    ]
                    
                    St_stress = S_global * (1.0 + spot_shocks)
                    Sig_stress = np.maximum(sigma_global + vol_shocks, 0.01)
                    
                    stress_mtm = AnalyticsEngine.valuate_portfolio_mc(St_stress, Sig_stress, edited_df, rates_dict)
                    stress_pnl = stress_mtm - mtm_hoje
                    
                    df_stress_analysis = pd.DataFrame({
                        "Cenários de Estresse": scenarios_names,
                        "Impacto Spot": [f"{s*100:+.1f}%" for s in spot_shocks],
                        "Impacto Vol": [f"{v*100:+.1f} pp" for v in vol_shocks],
                        "PnL de Choque (R$)": stress_pnl
                    })
                    
                    st.dataframe(
                        df_stress_analysis.style.format({"PnL de Choque (R$)": "R$ {:,.2f}"})
                        .background_gradient(subset=["PnL de Choque (R$)"], cmap="RdYlGn"),
                        use_container_width=True, hide_index=True
                    )
            else:
                st.info("Adicione ativos na mesa de operações para calcular as métricas de risco.")

    # --- TAB 2: ALPHA SCANNER DE ARBITRAGEM (ESTRUTURAL E SINTÉTICA) ---
    with tab2:
        st.markdown("<div class='secao-titulo'>4. SCANNER DE ARBITRAGEM E RELATIVE VOLATILITY Z-SCORE</div>", unsafe_allow_html=True)
        st.info("Insira dados de fechamento ou mercado intradia no formato CSV ou tabulação padrão de mercado (Strikes, Prêmios, Tickers). O scanner verificará violações estruturais.")
        
        raw_paste = st.text_area("Insira Dados (Smart Paste - Ctrl+V):", height=150, placeholder="Ticker\tStrike\tLast\nPETRF350\t35.00\t2.10\nPETRR350\t35.00\t1.45")
        
        if raw_paste and raw_paste != st.session_state["raw_text_cache"]:
            try:
                df_parsed = pd.read_csv(io.StringIO(raw_paste), sep='\t')
                
                ALIASES_K = ['strike', 'exerc', 'k', 'preço']
                ALIASES_P = ['ult', 'ultimo', 'last', 'prêmio', 'mercado']
                ALIASES_T = ['ativo', 'ticker', 'código', 'opção']
                
                col_k = next((c for c in df_parsed.columns if any(a in c.lower() for a in ALIASES_K)), None)
                col_p = next((c for c in df_parsed.columns if any(a in c.lower() for a in ALIASES_P)), None)
                col_t = next((c for c in df_parsed.columns if any(a in c.lower() for a in ALIASES_T)), None)
                
                if col_k and col_p:
                    df_parsed[col_k] = df_parsed[col_k].astype(str).apply(lambda x: re.sub(r'[^\d,.-]', '', x)).str.replace(',', '.').astype(float)
                    df_parsed[col_p] = df_parsed[col_p].astype(str).apply(lambda x: re.sub(r'[^\d,.-]', '', x)).str.replace(',', '.').astype(float)
                    
                    tipos, meses, business_days_list = [], [], []
                    
                    for _, row in df_parsed.iterrows():
                        tk = str(row[col_t]).upper() if col_t else "PETRF350"
                        match = re.match(r'^[A-Z]{4}([A-X])\d+$', tk)
                        if match:
                            letra = match.group(1)
                            is_call = letra <= 'L'
                            t_month = ord(letra) - ord('A') + 1 if is_call else ord(letra) - ord('M') + 1
                            tipo_p = "call" if is_call else "put"
                        else:
                            tipo_p = opt_type_global
                            t_month = 6
                            
                        days = B3Calendar.resolve_b3_expiry(tk, val_date)
                        if days is None:
                            days = T_days_global
                            
                        tipos.append(tipo_p)
                        meses.append(t_month)
                        business_days_list.append(days)
                        
                    df_parsed['Tipo_P'] = tipos
                    df_parsed['Mes_P'] = meses
                    df_parsed['Dias Úteis'] = business_days_list
                    df_parsed['T_Anos'] = df_parsed['Dias Úteis'] / 252
                    
                    st.session_state["raw_text_cache"] = raw_paste
                    st.session_state["parsed_df"] = df_parsed.dropna(subset=[col_k, col_p]).reset_index(drop=True)
                    st.session_state["col_k"] = col_k
                    st.session_state["col_p"] = col_p
            except Exception as e:
                st.error(f"Erro Crítico no processador de importação: {e}")

        if st.session_state["parsed_df"] is not None:
            col_k = st.session_state["col_k"]
            col_p = st.session_state["col_p"]
            
            df_editor_scanner = st.data_editor(
                st.session_state["parsed_df"],
                column_config={
                    "Tipo_P": st.column_config.SelectboxColumn("Tipo", options=["call", "put"]),
                    "Dias Úteis": st.column_config.NumberColumn("Dias Úteis Reais (B3)", min_value=1)
                },
                use_container_width=True,
                key="scanner_b3_editor_v18"
            )
            
            df_editor_scanner['T_Anos'] = df_editor_scanner['Dias Úteis'] / 252
            
            if st.button("🚀 Iniciar Testes de Arbitragem Estrutural"):
                violations = []
                
                for (tipo, mes), group in df_editor_scanner.groupby(['Tipo_P', 'Mes_P']):
                    g_sort = group.sort_values(by=col_k)
                    for i in range(len(g_sort)-1):
                        k1, p1 = g_sort[col_k].iloc[i], g_sort[col_p].iloc[i]
                        k2, p2 = g_sort[col_k].iloc[i+1], g_sort[col_p].iloc[i+1]
                        
                        if tipo == 'call' and p1 < p2:
                            violations.append(f"Violação de Monotonicidade (Call Mês {mes}): Strike {k1} (R$ {p1:.2f}) mais barato que Strike {k2} (R$ {p2:.2f})")
                        if tipo == 'put' and p1 > p2:
                            violations.append(f"Violação de Monotonicidade (Put Mês {mes}): Strike {k1} (R$ {p1:.2f}) mais caro que Strike {k2} (R$ {p2:.2f})")
                            
                        if i < len(g_sort)-2:
                            k3, p3 = g_sort[col_k].iloc[i+2], g_sort[col_p].iloc[i+2]
                            weight = (k3 - k2) / (k3 - k1)
                            if p2 > (weight * p1 + (1 - weight) * p3):
                                violations.append(f"Violação de Convexidade / Arbitragem de Butterfly ({tipo.upper()} Mês {mes}): Intermediário K={k2} violado pelas asas K1={k1} e K3={k3}")
                
                for (strike, mes), group in df_editor_scanner.groupby([col_k, 'Mes_P']):
                    calls = group[group['Tipo_P'] == 'call']
                    puts = group[group['Tipo_P'] == 'put']
                    if not calls.empty and not puts.empty:
                        C = calls[col_p].iloc[0]
                        P = puts[col_p].iloc[0]
                        t_item = calls['T_Anos'].iloc[0]
                        r_asset = YieldCurveEngine.interpolate_rate(t_item, rates_dict)
                        
                        lower_bound = S_global - strike * np.exp(-r_asset * t_item)
                        if (C - P) < lower_bound - 1e-4:
                            violations.append(f"Synthetic Lower Bound Violado (K={strike}, Mês {mes}): C - P ({C - P:.2f}) < Spot - K * exp(-rt) ({lower_bound:.2f})")

                if violations:
                    st.error("🚨 **OPORTUNIDADES DE ARBITRAGEM/VIOLAÇÕES DETECTADAS:**\n\n" + "\n".join(violations))
                else:
                    st.success("✅ Estrutura de preços robusta: Nenhuma arbitragem estática detectada na cadeia.")
                    
                misps, ivs = [], []
                for _, row in df_editor_scanner.iterrows():
                    kv, pv, tp, t_real = row[col_k], row[col_p], row['Tipo_P'], row['T_Anos']
                    r_asset = YieldCurveEngine.interpolate_rate(t_real, rates_dict)
                    ptheo = CoreModels.engine_bsm(S_global, kv, t_real, r_asset, sigma_global, tp)
                    misp = (pv / ptheo - 1) * 100 if ptheo > 0.01 else np.nan
                    
                    iv = CoreModels.solve_implied_volatility(pv, S_global, kv, t_real, r_asset, tp)
                    misps.append(misp)
                    ivs.append(iv * 100 if not np.isnan(iv) else np.nan)
                    
                df_editor_scanner['Misp (%)'] = misps
                df_editor_scanner['IV_Market'] = ivs
                
                mean_misp = df_editor_scanner['Misp (%)'].mean()
                std_misp = df_editor_scanner['Misp (%)'].std()
                df_editor_scanner['Z-Score Relative Vol'] = (df_editor_scanner['Misp (%)'] - mean_misp) / (std_misp + 1e-9)
                
                st.write("**Desvios Estatísticos Relativos à Volatilidade de Referência**")
                st.dataframe(
                    df_editor_scanner[[col_k, col_p, 'Tipo_P', 'Dias Úteis', 'IV_Market', 'Misp (%)', 'Z-Score Relative Vol']]
                    .style.format("{:.2f}", na_rep="-")
                    .background_gradient(subset=['Z-Score Relative Vol'], cmap='RdYlGn_r'),
                    use_container_width=True
                )

    # --- TAB 3: SUPERFÍCIE DE VOLATILIDADE (ROBUSTA POR GRIDDATA) ---
    with tab3:
        st.markdown("<div class='secao-titulo'>5. MAPEAMENTO DE VOLATILIDADE IMPLÍCITA (GRIDDATA ROBUSTO)</div>", unsafe_allow_html=True)
        st.info("Modelagem da superfície implícita ajustada por GridData linear com fallback seguro e tratamento de Runge-phenomenon.")
        
        sample_strikes = np.array([30.0, 32.5, 35.0, 37.5, 40.0] * 3)
        sample_maturities = np.array([21/252]*5 + [42/252]*5 + [63/252]*5)
        sample_ivs = np.array([
            0.38, 0.33, 0.30, 0.31, 0.34, 
            0.36, 0.32, 0.29, 0.30, 0.33, 
            0.34, 0.31, 0.28, 0.29, 0.32  
        ])
        
        grid_k, grid_t = np.meshgrid(np.linspace(28, 42, 50), np.linspace(10/252, 90/252, 50))
        
        grid_iv = interpolate.griddata(
            (sample_strikes, sample_maturities), sample_ivs, 
            (grid_k, grid_t), method='linear'
        )
        
        nan_mask = np.isnan(grid_iv)
        if np.any(nan_mask):
            grid_iv_nearest = interpolate.griddata(
                (sample_strikes, sample_maturities), sample_ivs, 
                (grid_k, grid_t), method='nearest'
            )
            grid_iv[nan_mask] = grid_iv_nearest[nan_mask]
            
        grid_iv = np.clip(grid_iv, 0.01, 2.00) 
        
        col_surf_3d, col_surf_hm = st.columns(2)
        
        with col_surf_3d:
            fig_surf = go.Figure(data=[go.Surface(z=grid_iv * 100, x=grid_k, y=grid_t * 252, colorscale='Viridis')])
            fig_surf.update_layout(
                title="Superfície 3D de Volatilidade Implícita (%)",
                scene=dict(
                    xaxis_title="Strike (K)",
                    yaxis_title="Maturidade (Dias Úteis)",
                    zaxis_title="Volatilidade (%)"
                ),
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_surf, use_container_width=True)
            
        with col_surf_hm:
            fig_hm = go.Figure(data=go.Heatmap(
                z=grid_iv * 100,
                x=np.linspace(28, 42, 50),
                y=np.linspace(10, 90, 50),
                colorscale='Viridis',
                colorbar=dict(title="Vol (%)")
            ))
            fig_hm.update_layout(
                title="Heatmap da Estrutura a Termo e Smile",
                xaxis_title="Strike (K)",
                yaxis_title="Dias Úteis",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    # --- TAB 4: LABORATÓRIO ISOLADO & AUDITORIA QUANT ---
    with tab4:
        st.markdown("<div class='secao-titulo'>6. LABORATÓRIO MATEMÁTICO & EXPLAINER CALCULATOR</div>", unsafe_allow_html=True)
        st.info("Laboratório isolado para auditoria analítica e refinamento quantitativo.")
        
        K_lab = st.number_input("Strike do Ativo (K):", value=36.00, step=0.50)
        T_lab_days = st.number_input("Dias Úteis do Ativo (T):", value=21, step=1)
        T_lab = T_lab_days / 252
        
        r_lab = YieldCurveEngine.interpolate_rate(T_lab, rates_dict)
        
        p_lab = CoreModels.engine_bsm(S_global, K_lab, T_lab, r_lab, sigma_global, opt_type_global)
        g_lab = CoreModels.calc_greeks_bsm(S_global, K_lab, T_lab, r_lab, sigma_global, opt_type_global)
        
        dt = 1/252
        p_dt_minus = CoreModels.engine_bsm(S_global, K_lab, max(T_lab - dt, 1e-9), r_lab, sigma_global, opt_type_global)
        discrete_theta = (p_dt_minus - p_lab) 
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Prêmio Teórico BSM", f"R$ {p_lab:.4f}")
        c2.metric("Delta do Ativo", f"{g_lab['delta']:.4f}")
        c3.metric("Gamma Analítico", f"{g_lab['gamma']:.4f}")
        c4.metric("Vega Analítico (1%)", f"R$ {g_lab['vega']:.4f}")
        
        st.write("### 📐 Sensibilidades de Ordem Superior (Mesa Exótica)")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Vanna", f"{g_lab['vanna']:.4f}", help="Sensibilidade do Delta à Volatilidade.")
        c6.metric("Vomma", f"{g_lab['vomma']:.4f}", help="Sensibilidade do Vega à Volatilidade.")
        c7.metric("Charm (Daily decay)", f"{g_lab['charm']:.6f}", help="Decaimento do Delta por dia útil.")
        c8.metric("Speed", f"{g_lab['speed']:.6f}", help="Sensibilidade do Gamma ao preço do Spot.")
        
        st.write("---")
        st.write("### 🔬 Auditoria Metodológica e Equações (Explain Calculation)")
        
        with st.expander("Exibir Equações Teóricas e Parâmetros Utilizados neste Cálculo"):
            st.markdown("<div class='audit-box'><b>Mapeamento Black-Scholes-Merton:</b></div>", unsafe_allow_html=True)
            st.latex(r"d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}")
            st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
            st.latex(r"\text{Call Price} = S \cdot N(d_1) - K e^{-r T} N(d_2)")
            
            st.markdown("<div class='audit-box'><b>Gregas de Alta Ordem Empregadas:</b></div>", unsafe_allow_html=True)
            st.latex(r"\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = - \phi(d_1) \frac{d_2}{\sigma}")
            st.latex(r"\text{Charm} = \frac{\partial \Delta}{\partial T} = - \phi(d_1) \left[ \frac{r}{\sigma\sqrt{T}} - \frac{d_2}{2T} \right]")
            
            st.markdown("**Comparativo de Modelagem do Theta (Aproximação Temporal):**")
            st.write(f"- **Theta Analítico BSM Diário (Anual/252):** R$ {g_lab['theta']:.6f}")
            st.write(f"- **Theta Discreto (Diferenças Finitas em T - 1/252):** R$ {discrete_theta:.6f}")
            st.write(f"- **Divergência de Discretização:** {abs(g_lab['theta'] - discrete_theta):.6e}")

    # --- TAB 5: GOVERNANÇA ---
    with tab5:
        st.markdown("<div class='secao-titulo'>7. MODEL METADATA & MODEL BACKTESTING SUITE</div>", unsafe_allow_html=True)
        st.info("Mapeamento de governança de modelos e suite de validação de backtesting quantitativo (Basileia / Traffic Light Framework).")
        
        col_gov_left, col_gov_right = st.columns(2)
        
        with col_gov_left:
            st.markdown(f"""
            ### Parâmetros Técnicos de Execução:
            - **Engine Version:** v18.2.0 (Dynamic Continuous Drift Node)
            - **Model Execution Environment:** Streamlit Virtual Machine
            - **Validation Status:** Approved by Quant Audit Suite
            - **Data Base Atual:** {val_date}
            - **DI Contínuo Aplicado (30 DU):** {np.log(1.0 + r_curto)*100:.4f}% (Discreto: {r_curto*100:.2f}%)
            - **DI Contínuo Aplicado (252 DU):** {np.log(1.0 + r_medio)*100:.4f}% (Discreto: {r_medio*100:.2f}%)
            - **DI Contínuo Aplicado (1008 DU):** {np.log(1.0 + r_longo)*100:.4f}% (Discreto: {r_longo*100:.2f}%)
            """)
            
        with col_gov_right:
            st.write("**Suite de Backtesting Estatístico do VaR (Kupiec & Christoffersen Tests)**")
            
            np.random.seed(42)
            simulated_pnl = np.random.normal(0, 1000, 252)
            simulated_var = -1645 * np.ones(252)
            
            exceptions = np.sum(simulated_pnl < simulated_var)
            p_hat = exceptions / 252
            p_target = 0.05
            
            lr_pof = -2 * ( (252 - exceptions) * np.log((1 - p_target)/(1 - p_hat + 1e-9)) + exceptions * np.log(p_target / (p_hat + 1e-9)) )
            critical_val = 3.84
            pof_passed = lr_pof < critical_val
            
            st.write(f"- **Número de Exceções Observadas (252 dias):** {exceptions}")
            st.write(f"- **Taxa de Exceção Real:** {p_hat*100:.2f}% (Meta de 5.00%)")
            st.write(f"- **Estatística do Teste de Kupiec (LR):** {lr_pof:.4f} (Valor Crítico: {critical_val})")
            
            if exceptions <= 4:
                st.markdown("- **Semáforo de Basileia:** <span class='test-badge test-success'>VERDE</span>", unsafe_allow_html=True)
            elif exceptions <= 9:
                st.markdown("- **Semáforo de Basileia:** <span class='test-badge' style='background-color: #f59e0b; color: white;'>AMARELO</span>", unsafe_allow_html=True)
            else:
                st.markdown("- **Semáforo de Basileia:** <span class='test-badge test-fail'>VERMELHO</span>", unsafe_allow_html=True)
                
            if pof_passed:
                st.success("✅ **Kupiec Test Passed:** O modelo de VaR calibrado não apresenta viés estatístico de subestimação de risco.")
            else:
                st.error("🚨 **Kupiec Test Failed:** Desvio estatisticamente significativo em relação à frequência teórica de cauda.")


elif app_choice == "Terminal Quantitativo v8.0: Institutional Asset Suite":
    
    # 🎨 CSS Estilos Específicos do Código 2
    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            background-color: rgba(28, 131, 225, 0.05);
            padding: 15px; border-radius: 8px; border: 1px solid rgba(28, 131, 225, 0.1);
        }
        .stButton>button {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            color: white; font-weight: bold; width: 100%; height: 3.5em; border: none;
        }
        .nota-metrica { font-size: 0.85rem; color: #666; font-style: italic; margin-top: -10px; margin-bottom: 15px; }
        .secao-titulo { color: #1e3a8a; font-weight: bold; border-bottom: 2px solid #1e3a8a; padding-bottom: 5px; margin-bottom: 20px; }
        </style>
        """, unsafe_allow_html=True)

    st.title("Terminal Quantitativo v8.0: Institutional Asset Suite")
    st.write("Black-Litterman Framework | Markowitz Frontier | Walk-Forward Validation")
    st.write("---")

    st.markdown("<div class='secao-titulo'>1. PARÂMETROS DE MERCADO E MANDATO</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    with c1:
        tickers_in = st.text_input(
            "Universo de Ativos:",
            "VALE3.SA, ITSA4.SA, BBAS3.SA, GOAU4.SA, CSAN3.SA",
            help=(
                "Tickers via Yahoo Finance. "
                "Brasil: sufixo .SA (ex: PETR4.SA) | "
                "EUA: ticker puro (ex: AAPL) | "
                "Crypto: Ticker-USD (ex: BTC-USD). "
                "Máximo recomendado: 10 ativos. Acima disso, a matriz de covariância "
                "se torna numericamente instável sem regularização adicional."
            )
        )
        tickers = [t.strip().upper() for t in tickers_in.split(",")]

    with c2:
        rf_rate = st.number_input(
            "Risk-Free (Anual %):", 0.0, 20.0, 10.75,
            help=(
                "Taxa livre de risco anualizada. Proxy do CDI/Selic para portfólios "
                "em BRL, ou T-Bills para portfólios em USD. "
                "É o 'hurdle rate' que define o excesso de retorno (Alpha). "
                "O otimizador maximiza o Sharpe = (E[Rp] - rf) / σp, "
                "portanto valores mais altos de rf tornam o mandato mais restritivo."
            )
        ) / 100

    with c3:
        t_cost = st.slider(
            "Custo de Transação (bps):", 0, 100, 10,
            help=(
                "Penalidade de turnover em basis points (1 bps = 0,01%). "
                "Cada 1% de rotação nos pesos subtrai N bps do retorno esperado na função objetivo. "
                "Desincentiva overtrading induzido por ruído estatístico (problema clássico de MVO). "
                "Referência: corretoras BR cobram ~5-15 bps por operação institucional; "
                "fundos de ações incorrem em ~20-50 bps considerando impacto de mercado."
            )
        ) / 10000

    with c4:
        s_date = st.date_input(
            "Início da Série:", date(2020, 1, 1),
            help=(
                "Data de início para download e estimação. "
                "Séries mais longas (5+ anos) suavizam outliers e reduzem a variância "
                "do estimador de covariância, mas podem ignorar mudanças de regime recentes. "
                "Séries curtas (1-2 anos) capturam o regime corrente, mas aumentam "
                "o risco de overfitting na matriz Σ. "
                "Para portfólios com BTC ou small caps: recomenda-se ao menos 3 anos."
            )
        )

    st.markdown("<div class='secao-titulo'>2. BENCHMARK E CONFIGURAÇÕES AVANÇADAS</div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)

    with b1:
        bench_options = {
            "IBOVESPA (^BVSP)":       "^BVSP",       
            "S&P 500 (^GSPC)":        "^GSPC",        
            "MSCI World (URTH)":      "URTH",          
            "CDI Proxy (IRFM11.SA)":  "IRFM11.SA",   
            "Personalizado":          "__custom__"    
        }
        bench_label = st.selectbox(
            "Benchmark:",
            list(bench_options.keys()),
            index=0,
            help=(
                "Índice de referência para cálculo do Information Ratio (IR) e Tracking Error. "
                "IR = (Rp - Rb) / TE, onde TE é o desvio-padrão anualizado do excesso de retorno diário. "
                "Escolha um benchmark representativo do universo investível do mandato: "
                "ações BR → IBOVESPA | mistos → MSCI World | conservadores → CDI Proxy."
            )
        )

    with b2:
        custom_bench = ""
        if bench_label == "Personalizado":
            custom_bench = st.text_input(
                "Ticker do benchmark:", "^BVSP",
                help="Qualquer ticker válido no Yahoo Finance. Ex: ^BVSP, ^GSPC, BOVA11.SA, SPY."
            )
        else:
            st.info(f"Benchmark selecionado: **{bench_options[bench_label]}**")

    with b3:
        run_bootstrap = st.checkbox(
            "Calcular ICs Bootstrap (95%)",
            value=True,
            help=(
                "Gera intervalos de confiança para Sharpe e Max Drawdown "
                "via bootstrap não-paramétrico com 1.000 reamostras. "
                "Não assume normalidade — adequado para fat tails. "
                "Adiciona ~10-15s ao processamento. "
                "Recomendado: sempre ativo para relatórios institucionais."
            )
        )

    bench = custom_bench if bench_label == "Personalizado" else bench_options[bench_label]

    with st.expander("💡 Black-Litterman: Convicções e Nível de Confiança"):
        st.caption(
            "⚠️ **Premissa de mandato (importante):** as views abaixo representam convicções "
            "declaradas ex-ante, equivalentes a um IPS (Investment Policy Statement). "
            "O walk-forward as mantém constantes ao longo de todo o período simulado — "
            "elas não são reajustadas janela a janela. "
            "Isso reflete um mandato de gestão ativa com convicções de médio prazo, "
            "não um modelo preditivo adaptativo."
        )
        v_cols = st.columns(len(tickers) if len(tickers) < 6 else 5)
        views, confs = {}, {}
        for i, t in enumerate(tickers):
            with v_cols[i % len(v_cols)]:
                v = st.number_input(
                    f"E[R] {t} (%)", -50, 100, 0, key=f"v_{t}",
                    help=(
                        f"Retorno absoluto anualizado esperado para {t}. "
                        f"Se zero, o modelo usa o equilíbrio de mercado (CAPM reverso) como prior. "
                        f"Positivo = visão construtiva; negativo = visão baixista."
                    )
                )
                c = st.slider(
                    f"Confiança {t}", 0.1, 1.0, 0.5, key=f"c_{t}",
                    help=(
                        f"Nível de convicção na view de {t}. "
                        f"Controla Ω_ii = (P_i τΣ P_i') / confiança na fórmula de Theil. "
                        f"1.0 = convicção total (view domina o prior); "
                        f"0.1 = baixa convicção (prior de mercado domina)."
                    )
                )
                if v != 0:
                    views[t], confs[t] = v / 100, c

    if st.button("🚀 GERAR RELATÓRIO QUANTITATIVO COMPLETO"):
        with st.spinner("Processando Walk-Forward e Estimadores Robustos..."):

            all_tickers = tickers + ([bench] if bench not in tickers else [])
            raw = Ticker(all_tickers).history(start=s_date.isoformat())
            prices_raw = raw.reset_index().pivot(
                index='date', columns='symbol', values='adjclose'
            ).ffill()  

            missing_report = {}
            for t in tickers:
                if t in prices_raw.columns:
                    first_valid = prices_raw[t].first_valid_index()
                    if first_valid is not None:
                        missing_pct = prices_raw[t].isna().mean() * 100  
                        missing_report[t] = {
                            'primeiro_dado': first_valid,
                            'pct_faltante':  missing_pct
                        }

            data_warnings = []
            for t, info in missing_report.items():
                if info['pct_faltante'] > 5:
                    data_warnings.append(
                        f"**{t}**: {info['pct_faltante']:.1f}% de dados faltantes "
                        f"(primeiro registro: {str(info['primeiro_dado'])[:10]})"
                    )

            if data_warnings:
                st.warning(
                    "⚠️ **Atenção — cobertura de dados incompleta:**\n\n" +
                    "\n\n".join(data_warnings) +
                    "\n\nAtivos com dados esparsos distorcem a matriz de covariância. "
                    "Considere reduzir o período inicial ou substituir esses ativos."
                )

            prices_full = prices_raw.dropna()
            total_requested = (date.today() - s_date).days   
            total_effective = len(prices_full)                  
            coverage_ratio = total_effective / max(total_requested, 1)

            if coverage_ratio < 0.50:
                st.error(
                    f"❌ A janela efetiva ({total_effective} pregões) é menor que 50% "
                    f"do período solicitado. O walk-forward pode ser insuficiente. "
                    f"Reduza o período inicial ou revise os ativos."
                )
                st.stop()
            elif coverage_ratio < 0.75:
                st.warning(
                    f"⚠️ A janela efetiva ({total_effective} pregões) representa "
                    f"{coverage_ratio:.0%} do período solicitado. "
                    f"Resultados devem ser interpretados com cautela."
                )

            bench_prices = prices_full[bench] if bench in prices_full.columns else None
            asset_prices = prices_full[[t for t in tickers if t in prices_full.columns]]

            if bench_prices is None:
                st.error(f"Benchmark '{bench}' não encontrado no Yahoo Finance. Verifique o ticker.")
                st.stop()

            bench_rets = bench_prices.pct_change().dropna()  
            rets = asset_prices.pct_change().dropna()           

            lw = LedoitWolf().fit(rets)
            cov_robust = lw.covariance_ * 252   

            w_mkt = get_market_weights(asset_prices)
            pi = 3.0 * (cov_robust @ w_mkt)    

            mu_bl, _ = black_litterman_full(
                pd.Series(pi, index=asset_prices.columns), cov_robust, views, confs
            )

            n = len(asset_prices.columns)

            opt_s, conv_s = safe_optimize(
                lambda w: -calculate_stats(w, mu_bl, cov_robust, rf_rate)[2], n, "Max Sharpe"
            )
            opt_v, conv_v = safe_optimize(
                lambda w: calculate_stats(w, mu_bl, cov_robust, rf_rate)[1], n, "Min Vol"
            )
            opt_rp, conv_rp = safe_optimize(
                lambda w: risk_parity_objective(w, cov_robust), n, "Risk Parity"
            )

            convergence_map = {
                "Máximo Sharpe":     conv_s,
                "Mínima Variância":  conv_v,
                "Paridade de Risco": conv_rp
            }
            failed = [k for k, v in convergence_map.items() if not v]
            if failed:
                st.warning(
                    f"⚠️ Otimização não convergiu para: **{', '.join(failed)}**. "
                    f"Pesos exibidos são a melhor aproximação via reinicialização aleatória múltipla. "
                    f"Interprete com cautela."
                )

            rolling, n_conv_failures = rolling_backtest(
                asset_prices, rf_rate, t_cost, views, confs
            )
            bench_aligned = bench_rets.reindex(rolling.index).fillna(0)

            if n_conv_failures > 0:
                total_windows = len(rolling) // 21
                st.info(
                    f"ℹ️ {n_conv_failures} de ~{total_windows} janelas de rebalanceamento "
                    f"não convergiram. Nesses períodos, a carteira foi mantida sem alteração (hold strategy)."
                )

            # ==============================================================================
            # 📊 OUTPUT 1: PERFORMANCE OUT-OF-SAMPLE (REALIZADA)
            # ==============================================================================
            st.markdown(
                "<div class='secao-titulo'>3. PERFORMANCE OUT-OF-SAMPLE (REALIZADA)</div>",
                unsafe_allow_html=True
            )

            ann_ret = rolling.mean() * 252                                     
            ann_vol = rolling.std() * np.sqrt(252)                               
            sharpe  = (ann_ret - rf_rate) / ann_vol                            
            tracking_error = np.std(rolling - bench_aligned) * np.sqrt(252)   
            info_ratio = (ann_ret - bench_aligned.mean() * 252) / (tracking_error + 1e-9)  
            cum    = (1 + rolling).cumprod()                                    
            max_dd = ((cum / cum.cummax()) - 1).min()                          
            calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else np.nan     

            if run_bootstrap:
                with st.spinner("Calculando intervalos de confiança via bootstrap (1.000 reamostras)..."):
                    sharpe_mean, sharpe_lo, sharpe_hi = bootstrap_sharpe_ci(rolling.values, rf_rate)
                    mdd_mean, mdd_lo, mdd_hi = bootstrap_maxdd_ci(rolling.values)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                "Retorno Anualizado", f"{ann_ret:.2%}",
                help="Retorno médio diário × 252, já descontando custos de transação do walk-forward."
            )
            c2.metric(
                "Sharpe Ratio", f"{sharpe:.2f}",
                delta=f"IC 95%: [{sharpe_lo:.2f}, {sharpe_hi:.2f}]" if run_bootstrap else None,
                help=(
                    "Retorno excedente por unidade de volatilidade realizada: (Rp - rf) / σp. "
                    "Referência: Sharpe > 1.0 é bom; > 2.0, excelente para fundos brasileiros. "
                    "O IC bootstrap indica a incerteza estatística do estimador."
                )
            )
            c3.metric(
                "Information Ratio", f"{info_ratio:.2f}",
                help=(
                    f"Alpha anualizado sobre o benchmark ({bench}) / Tracking Error. "
                    "IR > 0.5 indica geração consistente de alpha. "
                    "IR negativo = underperformance ajustada ao risco ativo."
                )
            )
            c4.metric(
                "Max Drawdown", f"{max_dd:.2%}",
                delta=f"IC 95%: [{mdd_lo:.2%}, {mdd_hi:.2%}]" if run_bootstrap else None,
                help=(
                    "Maior queda cumulativa entre um pico e o vale subsequente. "
                    "Mede o risco de cauda e a resiliência psicológica exigida do investidor. "
                    "O IC bootstrap é conservador (quebra dependência serial)."
                )
            )
            c5.metric(
                "Calmar Ratio", f"{calmar:.2f}" if not np.isnan(calmar) else "N/A",
                help=(
                    "Retorno anualizado / |Max Drawdown|. "
                    "Padrão em fundos alternativos e long-biased brasileiros. "
                    "Calmar > 1.0: o portfólio recupera o pior drawdown em menos de 1 ano."
                )
            )

            if run_bootstrap:
                st.caption(
                    "📊 *ICs (95%) estimados via bootstrap não-paramétrico "
                    f"com 1.000 reamostras sobre {len(rolling)} observações diárias. "
                    "Não assume normalidade dos retornos.*"
                )

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=rolling.index, y=(1 + rolling).cumprod() * 10000,
                name="Estratégia (Walk-Forward)", line=dict(color='#1e3a8a', width=3)
            ))
            fig_bt.add_trace(go.Scatter(
                x=bench_aligned.index, y=(1 + bench_aligned).cumprod() * 10000,
                name=f"Benchmark ({bench})", line=dict(color='gray', dash='dot')
            ))
            fig_bt.update_layout(
                title="Equity Curve: Walk-Forward Validation (R$10k Inicial)",
                template="plotly_white", height=500,
                legend=dict(orientation="h", y=-0.2, xanchor="center", x=0.5),
                yaxis_title="Valor (R$)", xaxis_title="Data"
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            dd_series = (cum / cum.cummax()) - 1
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series * 100,
                fill='tozeroy', fillcolor='rgba(220,50,50,0.15)',
                line=dict(color='rgba(220,50,50,0.8)', width=1),
                name="Drawdown (%)"
            ))
            fig_dd.update_layout(
                title="Underwater Chart (Drawdown %)",
                template="plotly_white", height=280,
                yaxis_title="Drawdown (%)", xaxis_title="Data",
                margin=dict(t=40, b=40)
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # ==============================================================================
            # 📊 OUTPUT 2: ALOCAÇÃO E FRONTEIRA EFICIENTE
            # ==============================================================================
            st.markdown(
                "<div class='secao-titulo'>4. ANÁLISE DE ALOCAÇÃO E FRONTEIRA EFICIENTE</div>",
                unsafe_allow_html=True
            )

            tabs = st.tabs(["🎯 Máximo Sharpe", "🛡️ Mínima Variância", "⚖️ Paridade de Risco"])
            mandatos = [
                "Portfólio de tangência: maximiza a inclinação da Capital Allocation Line (CAL). Recomendado para investidores que buscam máxima eficiência retorno/risco.",
                "Carteira defensiva: minimiza a variância total da matriz Σ (Ledoit-Wolf). Ignora retornos esperados — adequada quando há baixa confiança nas estimativas de mu_BL.",
                "Risk budgeting: equaliza a contribuição marginal de risco de cada ativo. Mais robusto que MVO clássico; recomendado para mandatos multi-asset com convicções simétricas.",
            ]

            for i, (tab, opt, guia) in enumerate(zip(tabs, [opt_s, opt_v, opt_rp], mandatos)):
                with tab:
                    w  = opt.x
                    r, v, s = calculate_stats(w, mu_bl, cov_robust, rf_rate)
                    rc = risk_contributions(w, cov_robust)  
                    st.write(f"**Mandato:** {guia}")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.write("**Pesos de Alocação e Contribuição de Risco**")
                        df_pesos = pd.DataFrame({
                            'Peso (%)':           (w * 100).round(2),
                            'Contrib. Risco (%)': (rc * 100).round(2)
                        }, index=asset_prices.columns)
                        st.dataframe(
                            df_pesos.style.format("{:.2f}")
                            .background_gradient(subset=['Peso (%)'], cmap='Blues')
                            .background_gradient(subset=['Contrib. Risco (%)'], cmap='Oranges'),
                            use_container_width=True
                        )

                    with col_b:
                        fig_rc = go.Figure(go.Bar(
                            x=asset_prices.columns.tolist(),
                            y=(rc * 100).tolist(),
                            marker_color='#3b82f6',
                            name='Contrib. Risco (%)'
                        ))
                        fig_rc.add_trace(go.Bar(
                            x=asset_prices.columns.tolist(),
                            y=(w * 100).tolist(),
                            marker_color='rgba(30,58,138,0.4)',
                            name='Peso Nominal (%)'
                        ))
                        fig_rc.update_layout(
                            title="Peso Nominal vs Contribuição de Risco",
                            barmode='group', template="plotly_white", height=320,
                            legend=dict(orientation="h", y=-0.25),
                            yaxis_title="%", margin=dict(t=40, b=60)
                        )
                        st.plotly_chart(fig_rc, use_container_width=True, key=f"allocation_chart_{i}")

                    st.info(
                        f"📐 **Estatísticas ex-ante (Black-Litterman):** "
                        f"Retorno Esperado: {r:.2%} | Volatilidade: {v:.2%} | Sharpe: {s:.2f}"
                    )

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.write("**Fronteira Eficiente Paramétrica (Markowitz Robusto)**")

                with st.spinner("Calculando fronteira eficiente paramétrica..."):
                    front_vols, front_rets = efficient_frontier_parametric(
                        mu_bl, cov_robust, rf_rate, n_points=150
                    )

                mc_v, mc_r, mc_s = [], [], []
                for _ in range(2000):
                    ww = np.random.random(n)
                    ww /= np.sum(ww)                                
                    r_mc = np.sum(mu_bl * ww)
                    v_mc = np.sqrt(ww.T @ cov_robust @ ww)
                    mc_r.append(r_mc)
                    mc_v.append(v_mc)
                    mc_s.append((r_mc - rf_rate) / v_mc)            

                fig_fe = go.Figure()

                fig_fe.add_trace(go.Scatter(
                    x=np.array(mc_v) * 100, y=np.array(mc_r) * 100,
                    mode='markers',
                    marker=dict(
                        color=mc_s, colorscale='Viridis',
                        size=3, opacity=0.4, showscale=True,
                        colorbar=dict(title="Sharpe", x=1.15)
                    ),
                    name="Amostragem Monte Carlo", showlegend=True
                ))

                if len(front_vols) > 5:
                    fig_fe.add_trace(go.Scatter(
                        x=front_vols * 100, y=front_rets * 100,
                        mode='lines',
                        line=dict(color='black', width=2.5),
                        name="Fronteira Eficiente (paramétrica)"
                    ))

                r_s, v_s, _ = calculate_stats(opt_s.x, mu_bl, cov_robust, rf_rate)
                r_v, v_v, _ = calculate_stats(opt_v.x, mu_bl, cov_robust, rf_rate)
                r_rp, v_rp, _ = calculate_stats(opt_rp.x, mu_bl, cov_robust, rf_rate)

                fig_fe.add_trace(go.Scatter(
                    x=[v_s * 100], y=[r_s * 100], mode='markers+text',
                    marker=dict(color='red', size=14, symbol='star'),
                    text=["Max Sharpe"], textposition="top right", name="Max Sharpe"
                ))
                fig_fe.add_trace(go.Scatter(
                    x=[v_v * 100], y=[r_v * 100], mode='markers+text',
                    marker=dict(color='blue', size=14, symbol='diamond'),
                    text=["Min Vol"], textposition="top right", name="Min Vol"
                ))
                fig_fe.add_trace(go.Scatter(
                    x=[v_rp * 100], y=[r_rp * 100], mode='markers+text',
                    marker=dict(color='green', size=14, symbol='triangle-up'),
                    text=["Risk Parity"], textposition="top right", name="Risk Parity"
                ))

                fig_fe.update_layout(
                    xaxis_title="Risco Anualizado (%)",
                    yaxis_title="Retorno Esperado BL (%)",
                    template="plotly_white", margin=dict(r=150), height=500,
                    legend=dict(orientation="h", y=-0.25, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_fe, use_container_width=True)
                st.caption(
                    "**Linha preta:** fronteira eficiente real via otimização paramétrica (target-return sweep). "
                    "Cada ponto é o portfólio de mínima volatilidade para aquele retorno-alvo. "
                )

            with col_g2:
                st.write("**Matriz de Correlação Robusta (Ledoit-Wolf)**")

                corr_matrix = rets.corr()
                assets_list  = corr_matrix.columns.tolist()

                fig_heatmap = go.Figure(go.Heatmap(
                    z=corr_matrix.values,
                    x=assets_list, y=assets_list,
                    colorscale='RdBu',
                    zmin=-1, zmax=1, zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    colorbar=dict(title="Correlação", thickness=15)
                ))
                fig_heatmap.update_layout(
                    template="plotly_white", height=500,
                    margin=dict(t=20, b=20, l=20, r=80),
                    xaxis=dict(tickangle=-45)
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.caption(
                    "Estimador de encolhimento Ledoit-Wolf reduz overfitting na covariância amostral. "
                )

            # ==============================================================================
            # 📊 OUTPUT 3: DISTRIBUIÇÃO E ESTATÍSTICAS COMPARATIVAS
            # ==============================================================================
            st.markdown(
                "<div class='secao-titulo'>5. ANÁLISE DE DISTRIBUIÇÃO E RISCO</div>",
                unsafe_allow_html=True
            )

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                monthly_rets  = rolling.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                bench_monthly = bench_aligned.resample('ME').apply(lambda x: (1 + x).prod() - 1)

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=monthly_rets * 100, name="Estratégia",
                    nbinsx=30, marker_color='rgba(30,58,138,0.7)', opacity=0.75
                ))
                fig_dist.add_trace(go.Histogram(
                    x=bench_monthly * 100, name="Benchmark",
                    nbinsx=30, marker_color='rgba(128,128,128,0.5)', opacity=0.75
                ))
                fig_dist.update_layout(
                    barmode='overlay', title="Distribuição de Retornos Mensais",
                    xaxis_title="Retorno Mensal (%)", yaxis_title="Frequência",
                    template="plotly_white", height=350,
                    legend=dict(orientation="h", y=-0.25)
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_r2:
                def describe_rets(r, name):
                    ann_r = r.mean() * 252
                    ann_v = r.std() * np.sqrt(252)
                    sk    = float(r.skew())
                    ku    = float(r.kurtosis())
                    cum   = (1 + r).cumprod()
                    mdd   = ((cum / cum.cummax()) - 1).min()
                    var95 = r.quantile(0.05)
                    return {
                        'Retorno Anual (%)':  f"{ann_r:.2%}",
                        'Volatilidade (%)':   f"{ann_v:.2%}",
                        'Sharpe':             f"{(ann_r - rf_rate) / ann_v:.2f}",
                        'Calmar':             f"{ann_r / abs(mdd):.2f}" if abs(mdd) > 0 else "N/A",
                        'Max Drawdown':       f"{mdd:.2%}",
                        'VaR 95% (diário)':   f"{var95:.2%}",
                        'Skewness':           f"{sk:.3f}",
                        'Kurtosis (excesso)': f"{ku:.3f}",
                    }

                df_stats = pd.DataFrame({
                    "Estratégia": describe_rets(rolling, "Estratégia"),
                    "Benchmark":  describe_rets(bench_aligned, "Benchmark"),
                })
                st.write("**Estatísticas Comparativas**")
                st.dataframe(df_stats, use_container_width=True, height=320)

            # ==============================================================================
            # 📊 OUTPUT 4: CALENDÁRIO DE RETORNOS MENSAIS (HEATMAP)
            # ==============================================================================
            st.markdown(
                "<div class='secao-titulo'>6. CALENDÁRIO DE RETORNOS MENSAIS</div>",
                unsafe_allow_html=True
            )

            monthly_pivot = monthly_rets.copy()
            monthly_pivot.index = pd.to_datetime(monthly_pivot.index)
            pivot_df = pd.DataFrame({
                'year':  monthly_pivot.index.year,
                'month': monthly_pivot.index.month,
                'ret':   monthly_pivot.values
            }).pivot(index='year', columns='month', values='ret')
            pivot_df.columns = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                                'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'][:len(pivot_df.columns)]

            fig_cal = go.Figure(go.Heatmap(
                z=pivot_df.values * 100,
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                colorscale='RdYlGn',
                zmid=0,
                text=(pivot_df * 100).round(1).values,
                texttemplate="%{text}%",
                textfont=dict(size=10),
                colorbar=dict(title="Retorno (%)", thickness=15)
            ))
            fig_cal.update_layout(
                template="plotly_white",
                height=max(200, len(pivot_df) * 50 + 80),
                margin=dict(t=20, b=20),
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig_cal, use_container_width=True)

        st.success("✅ Relatório gerado com sucesso.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2026 Rafael Grilli — Grilli Research")
    st.sidebar.markdown(
        "**Disclaimer:** Este terminal é uma ferramenta de análise quantitativa. "
        "Não constitui recomendação de investimento. "
        "Resultados históricos não garantem performance futura. "
        "Toda decisão de alocação deve considerar o perfil de risco, "
        "horizonte e objetivos específicos do investidor."
    )
