import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ---------------------------
# Config
# ---------------------------
TRADES_FILE = "data_trades.csv"
TARGETS_FILE = "data_targets.csv"
CASH_FILE = "data_cash.csv"

# CoinGecko IDs (confirmed: NOCK = nockchain; TAO = bittensor)
COINGECKO_ID_BY_PROJECT = {
    "NOCK": "nockchain",
    "TAO": "bittensor",
}

# Optional fallback if a price is missing
FALLBACK_PRICE_BY_PROJECT: Dict[str, float] = {}

DEFAULT_VS_CURRENCY = "usd"


# ---------------------------
# Styles (premium)
# ---------------------------
PREMIUM_CSS = """
<style>
h1, h2, h3 { letter-spacing: -0.02em; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* Metric cards */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 14px 14px 10px 14px;
}
div[data-testid="stMetric"] > div { gap: 6px; }

/* Dataframe radius */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.06);
}

/* Section separators */
.hr {
  height: 1px;
  background: rgba(255,255,255,0.08);
  margin: 18px 0 18px 0;
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
}
.muted { opacity: 0.75; }
.small { font-size: 12px; opacity: 0.8; }
</style>
"""


# ---------------------------
# Helpers
# ---------------------------
@dataclass
class PriceResult:
    prices: Dict[str, float]
    source: str
    as_of_epoch: int


@st.cache_data(ttl=60, show_spinner=False)
def fetch_coingecko_prices(ids: List[str], vs_currency: str = DEFAULT_VS_CURRENCY) -> PriceResult:
    if not ids:
        return PriceResult(prices={}, source="coingecko", as_of_epoch=int(time.time()))

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": vs_currency}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        prices: Dict[str, float] = {}
        for _id in ids:
            if _id in data and vs_currency in data[_id]:
                prices[_id] = float(data[_id][vs_currency])
        return PriceResult(prices=prices, source="coingecko", as_of_epoch=int(time.time()))
    except Exception:
        return PriceResult(prices={}, source="coingecko_error", as_of_epoch=int(time.time()))


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["project"] = df["project"].astype(str).str.upper().str.strip()
    for col in ["amount_invested", "buy_price", "tokens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["project", "amount_invested", "buy_price", "tokens"])
    return df


def load_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["project"] = df["project"].astype(str).str.upper().str.strip()
    df["stage"] = df["stage"].astype(str).str.strip()
    df["target_price"] = pd.to_numeric(df["target_price"], errors="coerce")
    df["sell_pct"] = pd.to_numeric(df["sell_pct"], errors="coerce")
    df["note"] = df.get("note", "").astype(str)
    df = df.dropna(subset=["project", "stage", "target_price", "sell_pct"])
    return df


def load_cash(path: str) -> pd.DataFrame:
    """
    data_cash.csv:
    asset,amount
    USDT,4543
    """
    try:
        df = pd.read_csv(path)
        df["asset"] = df["asset"].astype(str).str.upper().str.strip()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["asset", "amount"])
        return df
    except Exception:
        return pd.DataFrame(columns=["asset", "amount"])


def consolidate_positions(trades: pd.DataFrame) -> pd.DataFrame:
    g = trades.groupby("project", as_index=False).agg(
        tokens_total=("tokens", "sum"),
        invested_total=("amount_invested", "sum"),
    )
    g["avg_entry"] = np.where(g["tokens_total"] > 0, g["invested_total"] / g["tokens_total"], np.nan)
    return g


def attach_live_prices(pos: pd.DataFrame, vs_currency: str) -> Tuple[pd.DataFrame, str]:
    ids: List[str] = []
    proj_to_id: Dict[str, str] = {}

    for p in pos["project"].tolist():
        _id = COINGECKO_ID_BY_PROJECT.get(p)
        if _id:
            ids.append(_id)
            proj_to_id[p] = _id

    pr = fetch_coingecko_prices(ids=ids, vs_currency=vs_currency)

    live_price: List[Optional[float]] = []
    for p in pos["project"].tolist():
        _id = proj_to_id.get(p)
        price_val: Optional[float] = None
        if _id and _id in pr.prices:
            price_val = pr.prices[_id]
        elif p in FALLBACK_PRICE_BY_PROJECT:
            price_val = FALLBACK_PRICE_BY_PROJECT[p]
        live_price.append(price_val)

    out = pos.copy()
    out["price_live"] = live_price
    out["value_live"] = out["tokens_total"] * out["price_live"]
    out["pnl_$"] = out["value_live"] - out["invested_total"]
    out["pnl_%"] = np.where(out["invested_total"] > 0, (out["pnl_$"] / out["invested_total"]) * 100, np.nan)
    return out, pr.source


def money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.2f}"


def price(x: Optional[float]) -> str:
    """Avoid misleading rounding. Show exact feel for sub-$1 assets."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) < 1:
        return f"${x:,.4f}"
    return f"${x:,.2f}"


def qty_tokens(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.2f}%"


def is_number(x: Optional[float]) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def progress(current: Optional[float], target: float) -> float:
    if not is_number(current) or target <= 0:
        return 0.0
    return float(np.clip(float(current) / target, 0.0, 1.0))


def yesno(flag: bool) -> str:
    return "Oui" if flag else "Non"


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="BW Crypto Dashboard", page_icon="📈", layout="wide")
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

st.title("📈 BW Crypto Dashboard")

with st.sidebar:
    st.header("⚙️ Paramètres")
    vs_currency = st.selectbox("Devise", options=["usd", "eur"], index=0, format_func=lambda x: x.upper())
    auto_refresh = st.toggle("Auto-refresh (60s)", value=True)
    refresh = st.button("🔄 Rafraîchir maintenant")
    st.divider()
    show_trades = st.toggle("Voir le détail des achats (DCA)", value=True)
    st.caption("Modifie data_trades.csv / data_targets.csv / data_cash.csv pour mettre à jour.")

# Real auto refresh (no manual page reload)
if auto_refresh:
    try:
        st.autorefresh(interval=60_000, key="auto_refresh_60s")
    except Exception:
        # fallback for older streamlit versions (shouldn't happen on cloud)
        pass

if refresh:
    st.cache_data.clear()
    st.rerun()

# Load data
trades = load_trades(TRADES_FILE)
targets = load_targets(TARGETS_FILE)
cash_df = load_cash(CASH_FILE)

# Compute positions
positions = consolidate_positions(trades)
positions_live, price_source = attach_live_prices(positions, vs_currency)

# Add Cash line (USDT)
cash_usdt = float(cash_df.loc[cash_df["asset"] == "USDT", "amount"].sum()) if not cash_df.empty else 0.0
cash_row = pd.DataFrame([{
    "project": "USDT (cash)",
    "tokens_total": cash_usdt,     # interpret as dollars for cash line
    "invested_total": 0.0,
    "avg_entry": np.nan,
    "price_live": 1.0,            # 1 USDT ~ 1$
    "value_live": cash_usdt,
    "pnl_$": 0.0,
    "pnl_%": np.nan
}])

# For table and charts where needed, we use a combined view
positions_all = pd.concat([positions_live, cash_row], ignore_index=True)

# KPI logic
invested_total = float(positions_live["invested_total"].sum())  # ONLY invested positions
value_positions_live = float(positions_live["value_live"].sum(skipna=True))
value_total_live = value_positions_live + cash_usdt             # positions + cash
pnl_positions = value_positions_live - invested_total           # pnl only on invested positions
pnl_positions_pct = (pnl_positions / invested_total * 100) if invested_total > 0 else np.nan

# KPI cards (5 columns for clarity)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Investi", money(invested_total))
k2.metric("Cash (USDT)", money(cash_usdt))
k3.metric("Valeur totale (live)", money(value_total_live))
k4.metric("PnL (positions)", money(pnl_positions))
k5.metric("PnL % (positions)", pct(pnl_positions_pct))

st.markdown(
    f'<span class="badge">Source prix: {price_source} • cache 60s</span> '
    f'<span class="muted">Si un prix est absent, il s’affichera en —</span>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("📌 Positions consolidées")

    df_show = positions_all.copy()
    df_show["Tokens"] = df_show["tokens_total"].map(qty_tokens)

    # PRU: show only for invested positions
    df_show["PRU (DCA)"] = df_show["avg_entry"].map(price)
    df_show.loc[df_show["project"] == "USDT (cash)", "PRU (DCA)"] = "—"

    df_show["Prix live"] = df_show["price_live"].map(price)
    df_show["Investi"] = df_show["invested_total"].map(money)
    df_show.loc[df_show["project"] == "USDT (cash)", "Investi"] = "—"

    df_show["Valeur"] = df_show["value_live"].map(money)

    # PnL: show only for invested positions
    df_show["PnL"] = df_show["pnl_$"].map(money)
    df_show["PnL %"] = df_show["pnl_%"].map(pct)
    df_show.loc[df_show["project"] == "USDT (cash)", ["PnL", "PnL %"]] = ["—", "—"]

    cols = ["project", "Tokens", "PRU (DCA)", "Prix live", "Investi", "Valeur", "PnL", "PnL %"]
    df_table = df_show[cols].rename(columns={"project": "Token"})
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("🎯 Objectifs (TP) — logique séquentielle")

    cur_price_by_proj = dict(zip(positions_live["project"], positions_live["price_live"]))
    inv_by_proj = dict(zip(positions_live["project"], positions_live["invested_total"]))
    tok_by_proj = dict(zip(positions_live["project"], positions_live["tokens_total"]))

    for proj in positions_live["project"].tolist():
        st.markdown(f"### {proj}")

        cur = cur_price_by_proj.get(proj)
        invested_proj = float(inv_by_proj.get(proj, 0.0))
        tokens_total_proj = float(tok_by_proj.get(proj, 0.0))

        t = targets[targets["project"] == proj].copy().sort_values("stage")
        if t.empty:
            st.info("Pas d'objectifs configurés.")
            continue

        remaining = tokens_total_proj
        cumulative_cash = 0.0

        for _, row in t.iterrows():
            stage = str(row["stage"])
            tgt = float(row["target_price"])
            sell_pct = float(row["sell_pct"])
            note = str(row.get("note", "")).strip()

            sold_tokens = remaining * sell_pct
            remaining_after = remaining - sold_tokens

            cash_if_hit = sold_tokens * tgt
            cumulative_cash += cash_if_hit

            st.write(f"**{stage}** • cible: **{price(tgt)}** • vente: **{int(sell_pct*100)}% du restant**")
            st.progress(progress(cur, tgt))
            if note:
                st.caption(note)

            # Clean layout: only metrics (no long text that shifts card heights)
            c1, c2, c3 = st.columns([1.05, 1.05, 1.2])

            with c1:
                st.metric("Tokens vendus (étape)", qty_tokens(sold_tokens))
                st.metric("Tokens restants", qty_tokens(remaining_after))

            with c2:
                st.metric("Cash si cible atteinte (étape)", money(cash_if_hit))
                st.metric("Valeur du restant (à la cible)", money(remaining_after * tgt))

            with c3:
                hit_live = is_number(cur) and float(cur) >= tgt
                st.metric("Target atteinte (prix live)", "Oui" if hit_live else "Non")

                # Break-even price to recover initial using THIS step's sold tokens
                if sold_tokens > 0 and invested_proj > 0:
                    be_price = invested_proj / sold_tokens
                    st.metric("Prix pour récupérer la mise (via étape)", price(be_price))
                    rec_at_target = tgt >= be_price
                    st.metric("À la cible, étape récupère la mise", "Oui" if rec_at_target else "Non")
                else:
                    st.metric("Prix pour récupérer la mise (via étape)", "—")
                    st.metric("À la cible, étape récupère la mise", "—")

                st.metric("Cash cumulé (si étapes atteintes)", money(cumulative_cash))

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
            remaining = remaining_after

        net_profit = cumulative_cash - invested_proj
        st.markdown(
            f"<span class='badge'>Total cash si toutes les étapes sont exécutées aux cibles : {money(cumulative_cash)}</span>",
            unsafe_allow_html=True,
        )
        st.write(f"**Bénéfice net (cash total - mise initiale)** : {money(net_profit)}")
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

with right:
    st.subheader("📊 Répartition du portefeuille")

    pie_df = positions_all.dropna(subset=["value_live"]).copy()
    if pie_df.empty:
        st.warning("Aucune valeur live disponible pour afficher la répartition (prix manquants).")
    else:
        fig = px.pie(pie_df, names="project", values="value_live", hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("📉 PnL par token (positions)")

    # Exclude cash from PnL chart
    bar_df = positions_live.dropna(subset=["pnl_$"]).copy()
    if not bar_df.empty:
        fig2 = px.bar(bar_df, x="project", y="pnl_$")
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("PnL indisponible (prix manquants).")

    if show_trades:
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.subheader("🧾 Détail des achats (DCA)")

        td = trades.copy().sort_values("date", ascending=False)
        td["Date"] = td["date"].dt.strftime("%Y-%m-%d")
        td["Investi"] = td["amount_invested"].map(money)
        td["Prix achat"] = td["buy_price"].map(price)
        td["Tokens"] = td["tokens"].map(qty_tokens)

        st.dataframe(
            td[["Date", "project", "Investi", "Prix achat", "Tokens"]].rename(columns={"project": "Token"}),
            use_container_width=True,
            hide_index=True,
        )

st.caption("🛠️ Pour ajouter d’autres tokens : ajoute des lignes dans data_trades.csv et renseigne leur ID CoinGecko dans COINGECKO_ID_BY_PROJECT.")
