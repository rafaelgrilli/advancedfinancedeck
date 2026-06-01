import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import scipy.optimize as sco
from sklearn.covariance import LedoitWolf

# ==============================================================================
# ⚙️ ESTILO E IDENTIDADE (DENSIDADE ANALÍTICA)
# ==============================================================================
st.set_page_config(page_title="Grilli Analytics | Institutional Terminal", layout="wide")

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

# ==============================================================================
# 🔢 CORE FUNCTIONS
# ==============================================================================

def get_market_weights(prices):
    """
    Proxy de capitalização de mercado via volatilidade inversa.
    Em mercados sem dados de market cap em tempo real, pesos inversamente
    proporcionais à volatilidade histórica aproximam o portfólio de equilíbrio
    usado como prior no modelo Black-Litterman.
    """
    vols = prices.pct_change().std()   # Volatilidade diária histórica por ativo
    inv_vol = 1 / vols                  # Inversão: ativos menos voláteis recebem mais peso
    w = inv_vol / inv_vol.sum()         # Normalização para soma = 1
    return w.values


def calculate_stats(weights, mu, cov_mat, rf):
    """
    Tríade fundamental de métricas de portfólio.
    Retorna (retorno esperado, volatilidade, Sharpe) anualizados.
    - mu: vetor de retornos esperados Black-Litterman (já anualizados)
    - cov_mat: matriz de covariância anualizada (Ledoit-Wolf)
    - rf: taxa livre de risco anualizada (hurdle rate do mandato)
    """
    p_ret = np.sum(mu * weights)                              # E[Rp] = w' * mu
    p_vol = np.sqrt(weights.T @ cov_mat @ weights)            # σp = sqrt(w' * Σ * w)
    p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0      # Sharpe = (E[Rp] - rf) / σp
    return p_ret, p_vol, p_sharpe


def risk_parity_objective(weights, cov_mat):
    """
    Função objetivo para Risk Parity (Maillard, Roncalli & Teiletche, 2010).
    Minimiza a soma dos quadrados dos desvios entre a contribuição marginal de risco
    de cada ativo (RC_i = w_i * ∂σp/∂w_i) e o alvo de equalização (σp / N).
    O mínimo global ocorre quando todos os ativos contribuem igualmente
    para a volatilidade total do portfólio.
    """
    p_vol = np.sqrt(weights.T @ cov_mat @ weights)   # Volatilidade total do portfólio
    marginal_risk = (cov_mat @ weights) / p_vol       # Risco marginal: ∂σp/∂w = Σw / σp
    risk_contribution = weights * marginal_risk        # RC_i = w_i * (∂σp/∂w_i)
    target = p_vol / len(weights)                      # Alvo de igualdade: σp / N
    return np.sum(np.square(risk_contribution - target))


def risk_contributions(weights, cov_mat):
    """
    Decompõe a volatilidade total em contribuições percentuais por ativo.
    Fundamental para auditoria do portfólio de Paridade de Risco e para
    identificar concentração implícita de risco em carteiras aparentemente diversificadas.
    Retorna vetor de contribuições normalizadas (soma = 1).
    """
    p_vol = np.sqrt(weights.T @ cov_mat @ weights)   # Volatilidade total
    marginal_risk = (cov_mat @ weights) / p_vol       # Risco marginal por ativo
    rc = weights * marginal_risk                       # Contribuição absoluta de risco
    return rc / rc.sum()                               # Normalização para porcentagem


def black_litterman_full(mu_prior, cov, views_dict, confidences=None, tau=0.05):
    """
    Implementação completa do modelo Black-Litterman (He & Litterman, 1999).

    O modelo combina o equilíbrio de mercado (prior) com as convicções do gestor
    (views) usando a fórmula de Theil para gerar retornos esperados posteriores.

    Parâmetros:
    - mu_prior   : vetor pi (retornos implícitos de equilíbrio = λ * Σ * w_mkt)
    - cov        : matriz de covariância Σ (Ledoit-Wolf, anualizada)
    - views_dict : dict {ticker: retorno_esperado_anualizado}
    - confidences: dict {ticker: confiança 0-1} — escala a incerteza Omega da view
    - tau        : escalar de incerteza do prior (tipicamente 0.01–0.10).
                    Valores menores tornam o prior mais rígido (menos influenciado pelas views).

    Fórmula posterior (Theil):
    μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ * [(τΣ)⁻¹π + P'Ω⁻¹Q]

    Onde:
    - P: matriz de seleção (picking matrix) — quais ativos cada view afeta
    - Q: vetor de retornos esperados das views
    - Ω: matriz diagonal de incerteza — Ω_ii = (P_i τΣ P_i') / confiança
          Confiança alta → Ω pequeno → view domina o prior
          Confiança baixa → Ω grande → prior domina a view
    """
    n = len(mu_prior)
    if not views_dict:
        return mu_prior, cov  # Sem views: retorna prior de mercado puro (CAPM reverso)

    # Construção de P (picking matrix) e Q (vetor de views)
    P = np.zeros((len(views_dict), n))
    Q = np.zeros(len(views_dict))
    assets = list(mu_prior.index)
    omega_diag = []

    for i, (asset, view_val) in enumerate(views_dict.items()):
        if asset in assets:
            idx = assets.index(asset)
            P[i, idx] = 1        # View absoluta: afeta apenas o ativo especificado
            Q[i] = view_val     # Retorno anualizado da view
            conf = confidences.get(asset, 0.5) if confidences else 0.5
            omega_diag.append((P[i] @ (tau * cov) @ P[i].T) / conf)

    omega = np.diag(omega_diag)

    # Inversas necessárias para a fórmula de Theil
    inv_prior = np.linalg.inv(tau * cov)
    inv_omega = np.linalg.inv(omega)

    # Fórmula posterior de Black-Litterman
    term1 = np.linalg.inv(inv_prior + P.T @ inv_omega @ P)
    mu_bl = term1 @ (inv_prior @ mu_prior + P.T @ inv_omega @ Q)

    return pd.Series(mu_bl, index=assets), cov


def rolling_backtest(prices, rf, t_cost, views, confs, window=252, rebalance=21):
    """
    Walk-Forward Validation com rebalanceamento mensal e penalidade de turnover.

    Metodologia:
    - Janela de treinamento (window=252): estima Σ via Ledoit-Wolf nos últimos 252 pregões
    - Janela de teste (rebalance=21): aplica os pesos nos próximos ~21 pregões (out-of-sample)
    - Custo de transação: penaliza o turnover em cada rebalanceamento, desincentivando
      rotação excessiva induzida por ruído estatístico (problema clássico de MVO)

    Nota sobre as views BL:
    As views são tratadas como mandato ex-ante (equivalente a um IPS — Investment
    Policy Statement). Elas são mantidas constantes ao longo de toda a simulação,
    refletindo convicções de médio prazo declaradas antes do período, não um modelo
    preditivo adaptativo janela-a-janela.

    FIX #6: Em caso de não-convergência do SLSQP, mantém a carteira anterior
    sem alteração (hold strategy), em vez de aplicar pesos inválidos silenciosamente.
    """
    rets = prices.pct_change().dropna()
    n_assets = len(prices.columns)
    w_prev = np.array([1 / n_assets] * n_assets)  # Inicialização: equally-weighted
    port_rets, dates = [], []
    convergence_failures = 0                        # Contador de falhas para diagnóstico

    for i in range(window, len(rets) - rebalance, rebalance):
        train = rets.iloc[i - window:i]    # Janela de treinamento: 252 pregões
        test  = rets.iloc[i:i + rebalance] # Janela de teste: ~21 pregões (out-of-sample)

        # Estimação robusta da covariância na janela de treinamento
        lw = LedoitWolf().fit(train)
        cov = lw.covariance_ * 252          # Anualização: × 252 pregões por ano

        # Prior de equilíbrio: retornos implícitos via CAPM reverso
        # λ = 3.0 é o coeficiente de aversão ao risco (calibrado para mercados emergentes)
        w_mkt = get_market_weights(train)
        pi = 3.0 * (cov @ w_mkt)           # π = λ * Σ * w_mkt

        # Atualização bayesiana: combina prior com views do gestor
        mu_bl, cov_bl = black_litterman_full(
            pd.Series(pi, index=prices.columns), cov, views, confs
        )

        # Função objetivo: maximiza Sharpe penalizado pelo custo de turnover
        def obj(w):
            r = np.sum(mu_bl * w)                 # Retorno esperado do portfólio
            v = np.sqrt(w.T @ cov_bl @ w)          # Volatilidade esperada
            turnover = np.sum(np.abs(w - w_prev))  # L1-norm da variação de pesos
            cost = t_cost * turnover               # Penalidade: bps × turnover
            return -(r - cost - rf) / v            # Minimiza o negativo do Sharpe líquido

        res = sco.minimize(
            obj, w_prev,
            bounds=tuple((0, 0.5) for _ in range(n_assets)),  # Long-only, max 50% por ativo
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Fully invested
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if res.success:
            w_prev = res.x           # Atualiza pesos apenas se a otimização convergiu
        else:
            convergence_failures += 1  # Registra falha; mantém w_prev (hold strategy)

        # Aplica os pesos na janela de teste (realização out-of-sample)
        r_test = test.dot(w_prev)
        port_rets.extend(r_test.tolist())
        dates.extend(test.index.tolist())

    # CORREÇÃO CIRÚRGICA: Converter índice para Datetime para evitar TypeError no resample
    return pd.Series(port_rets, index=pd.to_datetime(dates)), convergence_failures


def efficient_frontier_parametric(mu, cov_mat, rf, n_points=200):
    """
    FIX #4: Fronteira eficiente REAL via otimização paramétrica (target-return sweep).

    Problema de Markowitz original — para cada retorno-alvo μ* ∈ [μ_min, μ_max]:
        min   w' Σ w          (minimiza variância)
        s.t.  w' μ = μ* (atinge o retorno-alvo)
              Σw_i = 1          (fully invested)
              w_i ≥ 0           (long-only)
              w_i ≤ 0.5        (concentração máxima)

    Vantagem sobre Monte Carlo: a fronteira resultante é matematicamente ótima —
    não é uma aproximação por amostragem. Cada ponto é o portfólio de mínima
    volatilidade para aquele nível de retorno.

    Usa warm start (w0 = solução anterior) para acelerar convergência ao longo
    da fronteira, explorando a continuidade dos pesos ótimos.
    """
    n = len(mu)
    bnds = tuple((0, 0.5) for _ in range(n))
    cons_sum = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Sweep de retorno-alvo: de 80% do mínimo a 110% do máximo dos retornos individuais
    r_min = mu.min() * 0.8
    r_max = mu.max() * 1.1
    targets = np.linspace(r_min, r_max, n_points)

    front_vols, front_rets = [], []
    w0 = np.array([1 / n] * n)  # Chute inicial: equally-weighted

    for target in targets:
        cons = [
            cons_sum,
            {'type': 'eq', 'fun': lambda x, t=target: np.sum(mu * x) - t}  # Restrição de retorno-alvo
        ]
        res = sco.minimize(
            lambda w: np.sqrt(w.T @ cov_mat @ w),  # Minimiza volatilidade
            w0,
            bounds=bnds,
            constraints=cons,
            method='SLSQP',
            options={'ftol': 1e-10, 'maxiter': 1000}
        )
        if res.success:
            front_vols.append(res.fun)
            front_rets.append(target)
            w0 = res.x  # Warm start: usa solução anterior como chute inicial

    return np.array(front_vols), np.array(front_rets)


def bootstrap_sharpe_ci(returns, rf, n_boot=1000, ci=0.95):
    """
    FIX #5: Intervalo de confiança para o Sharpe Ratio via bootstrap não-paramétrico.

    O Sharpe histórico é um estimador ruidoso, especialmente com séries curtas.
    O bootstrap reamostrada os retornos diários com reposição N vezes, recalcula
    o Sharpe em cada amostra e usa os percentis da distribuição empírica como IC.
    Não assume normalidade — adequado para ativos com fat tails (cripto, small caps).

    Retorna: (sharpe_médio_bootstrap, limite_inferior_IC, limite_superior_IC)
    """
    sharpes = []
    n = len(returns)
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=n, replace=True)  # Reamostragem com reposição
        ann_r = sample.mean() * 252
        ann_v = sample.std() * np.sqrt(252)
        if ann_v > 0:
            sharpes.append((ann_r - rf) / ann_v)

    alpha = (1 - ci) / 2  # 2.5% em cada cauda para IC de 95%
    return (
        np.mean(sharpes),
        np.percentile(sharpes, alpha * 100),         # Limite inferior (percentil 2.5%)
        np.percentile(sharpes, (1 - alpha) * 100)    # Limite superior (percentil 97.5%)
    )


def bootstrap_maxdd_ci(returns, n_boot=1000, ci=0.95):
    """
    Intervalo de confiança para Max Drawdown via bootstrap.

    Nota: a reamostragem quebra a dependência serial dos retornos, subestimando
    drawdowns em períodos de tendência. O IC deve ser interpretado como limite
    inferior conservador do risco de cauda sequencial real.

    Retorna: (mdd_médio, limite_inferior, limite_superior)
    """
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
    """
    FIX #6: Wrapper robusto de otimização com fallback por reinicialização múltipla.

    Problema clássico de MVO: o SLSQP pode convergir para mínimos locais ou falhar
    completamente dependendo do chute inicial, especialmente com ativos de alta
    correlação ou covariância mal condicionada.

    Estratégia:
    1. Tentativa 1: SLSQP a partir de equally-weighted (w0 = 1/N)
    2. Se falhar: 10 reinicializações via Dirichlet(1,...,1) — amostra uniforme no simplex
        Aceita a melhor solução encontrada (convergida ou menor valor de função objetivo).

    Retorna: (resultado_scipy, booleano_convergiu)
    """
    w0 = np.array([1 / n_assets] * n_assets)               # Chute inicial: equally-weighted
    bnds = tuple((0, 0.5) for _ in range(n_assets))         # Long-only, max 50% por ativo
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Fully invested

    res = sco.minimize(
        obj_fn, w0, bounds=bnds, constraints=cons,
        method='SLSQP', options={'ftol': 1e-9, 'maxiter': 1000}
    )

    if not res.success:
        best = res
        for _ in range(10):
            # Dirichlet(α=1): amostra uniforme no simplex — garante diversidade de chutes
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
# 🎛️ PAINEL DE CONFIGURAÇÃO (CENTRALIZADO)
# ==============================================================================

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

# FIX #1: Benchmark selecionável pelo usuário
# O benchmark original era determinado automaticamente (.SA → IBOVESPA), o que gerava
# Information Ratio incorreto para portfólios mistos BRL/USD.
st.markdown("<div class='secao-titulo'>2. BENCHMARK E CONFIGURAÇÕES AVANÇADAS</div>", unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)

with b1:
    bench_options = {
        "IBOVESPA (^BVSP)":       "^BVSP",       # Referência para ações brasileiras
        "S&P 500 (^GSPC)":        "^GSPC",        # Referência para portfólios USD
        "MSCI World (URTH)":      "URTH",          # Referência global desenvolvida
        "CDI Proxy (IRFM11.SA)":  "IRFM11.SA",   # ETF renda fixa BR: proxy do CDI
        "Personalizado":          "__custom__"    # Qualquer ticker do Yahoo Finance
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

# ==============================================================================
# 🚀 EXECUÇÃO PRINCIPAL
# ==============================================================================

if st.button("🚀 GERAR RELATÓRIO QUANTITATIVO COMPLETO"):
    with st.spinner("Processando Walk-Forward e Estimadores Robustos..."):

        # --- Download e validação de dados ---
        all_tickers = tickers + ([bench] if bench not in tickers else [])
        raw = Ticker(all_tickers).history(start=s_date.isoformat())
        prices_raw = raw.reset_index().pivot(
            index='date', columns='symbol', values='adjclose'
        ).ffill()  # Forward-fill: propaga último preço válido (trata feriados nacionais)

        # FIX #3: Diagnóstico de cobertura de dados por ativo
        missing_report = {}
        for t in tickers:
            if t in prices_raw.columns:
                first_valid = prices_raw[t].first_valid_index()
                if first_valid is not None:
                    missing_pct = prices_raw[t].isna().mean() * 100  # % de NaN antes do ffill
                    missing_report[t] = {
                        'primeiro_dado': first_valid,
                        'pct_faltante':  missing_pct
                    }

        # Emite alertas para ativos com >5% de dados originalmente faltantes
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

        # Verifica cobertura efetiva após remoção de linhas com NaN
        prices_full = prices_raw.dropna()
        total_requested = (date.today() - s_date).days   # Dias totais solicitados
        total_effective = len(prices_full)                  # Pregões efetivos disponíveis
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

        # Separação de benchmark e ativos do portfólio
        bench_prices = prices_full[bench] if bench in prices_full.columns else None
        asset_prices = prices_full[[t for t in tickers if t in prices_full.columns]]

        if bench_prices is None:
            st.error(f"Benchmark '{bench}' não encontrado no Yahoo Finance. Verifique o ticker.")
            st.stop()

        bench_rets = bench_prices.pct_change().dropna()  # Retornos diários do benchmark
        rets = asset_prices.pct_change().dropna()           # Retornos diários dos ativos

        # --- Estimadores robustos (estáticos, para análise ex-ante) ---
        lw = LedoitWolf().fit(rets)
        cov_robust = lw.covariance_ * 252   # Σ anualizada via encolhimento de Ledoit-Wolf

        # Prior de equilíbrio de mercado (retornos implícitos via CAPM reverso)
        w_mkt = get_market_weights(asset_prices)
        pi = 3.0 * (cov_robust @ w_mkt)    # π = λ * Σ * w_mkt (λ=3.0: aversão ao risco)

        # Retornos esperados Black-Litterman (posterior bayesiano)
        mu_bl, _ = black_litterman_full(
            pd.Series(pi, index=asset_prices.columns), cov_robust, views, confs
        )

        # --- Otimizações estáticas com verificação de convergência (FIX #6) ---
        n = len(asset_prices.columns)

        # Portfólio de Máximo Sharpe (tangência na CAL)
        opt_s, conv_s = safe_optimize(
            lambda w: -calculate_stats(w, mu_bl, cov_robust, rf_rate)[2], n, "Max Sharpe"
        )
        # Portfólio de Mínima Variância (defensivo, ignora retornos esperados)
        opt_v, conv_v = safe_optimize(
            lambda w: calculate_stats(w, mu_bl, cov_robust, rf_rate)[1], n, "Min Vol"
        )
        # Portfólio de Paridade de Risco (equaliza contribuições marginais de risco)
        opt_rp, conv_rp = safe_optimize(
            lambda w: risk_parity_objective(w, cov_robust), n, "Risk Parity"
        )

        # Relatório de convergência: alerta se alguma otimização falhou mesmo com fallback
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

        # --- Walk-Forward Backtest (out-of-sample) ---
        rolling, n_conv_failures = rolling_backtest(
            asset_prices, rf_rate, t_cost, views, confs
        )
        # Alinha benchmark ao índice da série out-of-sample; preenche lacunas com zero
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

        # Métricas de performance anualizadas
        ann_ret = rolling.mean() * 252                                     # Retorno médio anualizado
        ann_vol = rolling.std() * np.sqrt(252)                               # Volatilidade anualizada
        sharpe  = (ann_ret - rf_rate) / ann_vol                            # Sharpe ratio realizado
        tracking_error = np.std(rolling - bench_aligned) * np.sqrt(252)   # TE anualizado
        info_ratio = (ann_ret - bench_aligned.mean() * 252) / (tracking_error + 1e-9)  # IR (evita div/0)
        cum    = (1 + rolling).cumprod()                                    # Equity curve acumulada
        max_dd = ((cum / cum.cummax()) - 1).min()                          # Max Drawdown (pico-a-vale)
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else np.nan     # Calmar = retorno / |MDD|

        # FIX #5: Intervalos de confiança bootstrap não-paramétrico (95%)
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

        # Equity Curve: estratégia vs benchmark
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

        # Underwater Chart: visualização contínua do drawdown
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

        # CORREÇÃO UNICA E EXCLUSIVA AQUI: Adição de 'enumerate' e 'key' no plotly_chart
        for i, (tab, opt, guia) in enumerate(zip(tabs, [opt_s, opt_v, opt_rp], mandatos)):
            with tab:
                w  = opt.x
                r, v, s = calculate_stats(w, mu_bl, cov_robust, rf_rate)
                rc = risk_contributions(w, cov_robust)  # Contribuição % de risco por ativo
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
                    # Adição do parâmetro 'key' para evitar IDs duplicados
                    st.plotly_chart(fig_rc, use_container_width=True, key=f"allocation_chart_{i}")

                st.info(
                    f"📐 **Estatísticas ex-ante (Black-Litterman):** "
                    f"Retorno Esperado: {r:.2%} | Volatilidade: {v:.2%} | Sharpe: {s:.2f}"
                )

        # FIX #4: Fronteira eficiente paramétrica real (target-return sweep)
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.write("**Fronteira Eficiente Paramétrica (Markowitz Robusto)**")

            with st.spinner("Calculando fronteira eficiente paramétrica..."):
                front_vols, front_rets = efficient_frontier_parametric(
                    mu_bl, cov_robust, rf_rate, n_points=150
                )

            # Nuvem Monte Carlo como pano de fundo ilustrativo
            mc_v, mc_r, mc_s = [], [], []
            for _ in range(2000):
                ww = np.random.random(n)
                ww /= np.sum(ww)                                # Normalização
                r_mc = np.sum(mu_bl * ww)
                v_mc = np.sqrt(ww.T @ cov_robust @ ww)
                mc_r.append(r_mc)
                mc_v.append(v_mc)
                mc_s.append((r_mc - rf_rate) / v_mc)            # Sharpe

            fig_fe = go.Figure()

            # Fundo: amostragem Monte Carlo
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

            # Fronteira eficiente real
            if len(front_vols) > 5:
                fig_fe.add_trace(go.Scatter(
                    x=front_vols * 100, y=front_rets * 100,
                    mode='lines',
                    line=dict(color='black', width=2.5),
                    name="Fronteira Eficiente (paramétrica)"
                ))

            # Marcadores dos portfólios ótimos
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
            # Retornos mensais compostos
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
                """Computa tabela de estatísticas descritivas para série de retornos diários."""
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

# ==============================================================================
# 🗂️ SIDEBAR: CRÉDITOS E DISCLAIMER
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Rafael Grilli — Grilli Research")
st.sidebar.markdown(
    "**Disclaimer:** Este terminal é uma ferramenta de análise quantitativa. "
    "Não constitui recomendação de investimento. "
    "Resultados históricos não garantem performance futura. "
    "Toda decisão de alocação deve considerar o perfil de risco, "
    "horizonte e objetivos específicos do investidor."
)
