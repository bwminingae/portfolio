\
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
DEFAULT_VS_CURRENCY = "usd"  # user: en $
TRADES_FILE = "data_trades.csv"
TARGETS_FILE = "data_targets.csv"

# CoinGecko IDs (confirmed: NOCK = nockchain; TAO is listed as 'bittensor')
COINGECKO_ID_BY_PROJECT = {
    "NOCK": "nockchain",
    "TAO": "bittensor",
}

# If a token isn't listed or API fails, fallback to last known price (optional)
FALLBACK_PRICE_BY_PROJECT: Dict[str, float] = {
    # "NOCK": 0.01,
    # "TAO": 172.0,
}


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
    """
    Fetch prices from CoinGecko simple/price endpoint.
    Cached for 60 seconds (ttl).
    """
    if not ids:
        return PriceResult(prices={}, source="coingecko", as_of_epoch=int(time.time()))

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": vs_currency}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        prices = {}
        for _id in ids:
            if _id in data and vs_currency in data[_id]:
                prices[_id] = float(data[_id][vs_currency])
        return PriceResult(prices=prices, source="coingecko", as_of_epoch=int(time.time()))
    except Exception:
        # Fall back to empty; caller can use fallback mapping
        return PriceResult(prices={}, source="coingecko_error", as_of_epoch=int(time.time()))


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    # normalize project symbol
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


def consolidate_positions(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate trades per project.
    - tokens_total: sum tokens
    - invested_total: sum invested
    - avg_entry: invested_total / tokens_total (weighted)
    """
    g = trades.groupby("project", as_index=False).agg(
        tokens_total=("tokens", "sum"),
        invested_total=("amount_invested", "sum"),
    )
    g["avg_entry"] = np.where(g["tokens_total"] > 0, g["invested_total"] / g["tokens_total"], np.nan)
    return g


def attach_live_prices(pos: pd.DataFrame, vs_currency: str) -> Tuple[pd.DataFrame, str]:
    ids = []
    proj_to_id = {}
    for p in pos["project"].tolist():
        _id = COINGECKO_ID_BY_PROJECT.get(p)
        if _id:
            ids.append(_id)
            proj_to_id[p] = _id

    pr = fetch_coingecko_prices(ids=ids, vs_currency=vs_currency)
    live_price = []
    for p in pos["project"].tolist():
        _id = proj_to_id.get(p)
        price = None
        if _id and _id in pr.prices:
            price = pr.prices[_id]
        elif p in FALLBACK_PRICE_BY_PROJECT:
            price = FALLBACK_PRICE_BY_PROJECT[p]
        live_price.append(price)

    out = pos.copy()
    out["price_live"] = live_price
    out["value_live"] = out["tokens_total"] * out["price_live"]
    out["pnl_$"] = out["value_live"] - out["invested_total"]
    out["pnl_%"] = np.where(out["invested_total"] > 0, (out["pnl_$"] / out["invested_total"]) * 100, np.nan)
    source = pr.source
    return out, source


def money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.2f}"


def qty(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.2f}%"


def progress(current: Optional[float], target: float) -> float:
    if current is None or (isinstance(current, float) and np.isnan(current)) or target <= 0:
        return 0.0
    return float(np.clip(current / target, 0.0, 1.0))


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="BW Crypto Dashboard", page_icon="📈", layout="wide")

st.title("📈 BW Crypto Dashboard")
st.caption("Prix live via CoinGecko • Portefeuille consolidé (sans mention des wallets)")

with st.sidebar:
    st.header("⚙️ Paramètres")
    vs_currency = st.selectbox("Devise", options=["usd", "eur"], index=0, format_func=lambda x: x.upper())
    refresh = st.button("🔄 Rafraîchir maintenant")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=True)
    st.divider()
    st.subheader("Affichage")
    show_trades = st.toggle("Voir le détail des achats (DCA)", value=True)
    st.caption("Astuce: modifie les fichiers data_trades.csv / data_targets.csv pour mettre à jour.")

if auto_refresh and not refresh:
    # rerun roughly every 60s (cache ttl aligns)
    st.query_params["_ts"] = str(int(time.time() // 60))

# Load data
trades = load_trades(TRADES_FILE)
targets = load_targets(TARGETS_FILE)

positions = consolidate_positions(trades)
positions_live, price_source = attach_live_prices(positions, vs_currency)

# KPI
total_invested = float(positions_live["invested_total"].sum())
total_value = float(positions_live["value_live"].sum(skipna=True))
total_pnl = total_value - total_invested
total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Investi", money(total_invested))
k2.metric("Valeur (live)", money(total_value))
k3.metric("PnL ($)", money(total_pnl))
k4.metric("PnL (%)", pct(total_pnl_pct))
st.caption(f"Source prix: **{price_source}** (cache 60s). Si un prix est absent, il s’affichera en '—'.")

st.divider()

# Layout main
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("📌 Positions consolidées")
    df_show = positions_live.copy()
    df_show["Investi"] = df_show["invested_total"].map(money)
    df_show["Tokens"] = df_show["tokens_total"].map(qty)
    df_show["PRU (DCA)"] = df_show["avg_entry"].map(lambda x: money(x).replace("$", "$") if x==x else "—")
    df_show["Prix live"] = df_show["price_live"].map(money)
    df_show["Valeur"] = df_show["value_live"].map(money)
    df_show["PnL"] = df_show["pnl_$"].map(money)
    df_show["PnL %"] = df_show["pnl_%"].map(pct)

    cols = ["project", "Tokens", "PRU (DCA)", "Prix live", "Investi", "Valeur", "PnL", "PnL %"]
    df_table = df_show[cols].rename(columns={"project": "Token"})
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.subheader("🎯 Objectifs (TP)")
    # Join targets with current price
    cur_price_by_proj = dict(zip(positions_live["project"], positions_live["price_live"]))
    inv_by_proj = dict(zip(positions_live["project"], positions_live["invested_total"]))
    tok_by_proj = dict(zip(positions_live["project"], positions_live["tokens_total"]))

    for proj in positions_live["project"].tolist():
        st.markdown(f"### {proj}")
        cur = cur_price_by_proj.get(proj)

        t = targets[targets["project"] == proj].copy()
        if t.empty:
            st.info("Pas d'objectifs configurés.")
            continue

        # display cards per stage
        for _, row in t.sort_values("stage").iterrows():
            tgt = float(row["target_price"])
            sell_pct = float(row["sell_pct"])
            note = str(row.get("note", ""))

            # scenario: value if target hits
            tokens_total = float(tok_by_proj.get(proj, 0))
            invested_total = float(inv_by_proj.get(proj, 0))
            value_at_target = tokens_total * tgt
            # amount sold at stage
            sold_value = value_at_target * sell_pct
            sold_tokens = tokens_total * sell_pct

            colA, colB, colC = st.columns([1.2, 1.0, 1.0])
            with colA:
                st.write(f"**{row['stage']}** • cible: **{money(tgt)}** • vente: **{int(sell_pct*100)}%**")
                st.progress(progress(cur, tgt))
                st.caption(note if note else "")
            with colB:
                st.write("À la cible")
                st.metric("Valeur bag", money(value_at_target))
                st.metric("Valeur vendue", money(sold_value))
            with colC:
                st.write("Quantités")
                st.metric("Tokens vendus", qty(sold_tokens))
                st.metric("Tokens restants", qty(tokens_total - sold_tokens))

        st.divider()

with right:
    st.subheader("🥧 Répartition (camembert)")
    pie_df = positions_live.dropna(subset=["value_live"]).copy()
    if pie_df.empty:
        st.warning("Aucune valeur live disponible pour afficher le camembert (prix manquants).")
    else:
        fig = px.pie(pie_df, names="project", values="value_live", hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 PnL par token")
    bar_df = positions_live.dropna(subset=["pnl_$"]).copy()
    if not bar_df.empty:
        fig2 = px.bar(bar_df, x="project", y="pnl_$")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("PnL indisponible (prix manquants).")

    if show_trades:
        st.subheader("🧾 Détail des achats (DCA)")
        t = trades.copy().sort_values("date", ascending=False)
        t["Date"] = t["date"].dt.strftime("%Y-%m-%d")
        t["Investi"] = t["amount_invested"].map(money)
        t["Prix achat"] = t["buy_price"].map(money)
        t["Tokens"] = t["tokens"].map(qty)
        st.dataframe(t[["Date","project","Investi","Prix achat","Tokens"]].rename(columns={"project":"Token"}),
                     use_container_width=True, hide_index=True)

st.caption("🛠️ Pour ajouter d’autres tokens, ajoute des lignes dans data_trades.csv et renseigne leur ID CoinGecko dans COINGECKO_ID_BY_PROJECT.")
