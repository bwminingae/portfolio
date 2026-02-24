import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ---------------------------
# Config
# ---------------------------
TRADES_FILE = "data_trades.csv"
TARGETS_FILE = "data_targets.csv"
CASH_FILE = "data_cash.csv"

DEFAULT_VS_CURRENCY = "usd"

COINGECKO_ID_BY_PROJECT = {
    "NOCK": "nockchain",
    "TAO": "bittensor",
}

FALLBACK_PRICE_BY_PROJECT: Dict[str, float] = {}  # optional


# ---------------------------
# Styles (premium)
# ---------------------------
PREMIUM_CSS = """
<style>
h1, h2, h3 { letter-spacing: -0.02em; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 14px 14px 10px 14px;
}
div[data-testid="stMetric"] > div { gap: 6px; }

div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.06);
}

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
</style>
"""


# ---------------------------
# Helpers
# ---------------------------
def is_number(x: Optional[float]) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def money(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    return f"${float(x):,.2f}"


def price(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    x = float(x)
    if abs(x) < 1:
        return f"${x:,.4f}"
    return f"${x:,.2f}"


def qty_tokens(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def pct(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    return f"{float(x):,.2f}%"


def progress(current: Optional[float], target: float) -> float:
    if not is_number(current) or target <= 0:
        return 0.0
    return float(np.clip(float(current) / target, 0.0, 1.0))


@st.cache_data(ttl=60, show_spinner=False)
def fetch_coingecko_prices(ids: List[str], vs_currency: str) -> Tuple[Dict[str, float], str, int]:
    """
    Return only pickle-serializable types for st.cache_data.
    Returns: (prices_by_id, source, as_of_epoch)
    """
    if not ids:
        return {}, "coingecko", int(time.time())

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": vs_currency}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        out: Dict[str, float] = {}
        for _id in ids:
            if _id in data and vs_currency in data[_id]:
                out[_id] = float(data[_id][vs_currency])
        return out, "coingecko", int(time.time())
    except Exception:
        return {}, "coingecko_error", int(time.time())


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["project"] = df["project"].astype(str).str.upper().str.strip()
    for col in ["amount_invested", "buy_price", "tokens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["project", "amount_invested", "buy_price", "tokens"])
    return df


def load_targets(path: str) -> pd.DataFrame:
    """
    New format:
    project,stage,multiple,sell_pct,note

    Backward compatibility:
    if a file still contains 'target_price', we keep it, but prefer 'multiple' when present.
    """
    df = pd.read_csv(path)
    df["project"] = df["project"].astype(str).str.upper().str.strip()
    df["stage"] = df["stage"].astype(str).str.strip()
    df["sell_pct"] = pd.to_numeric(df["sell_pct"], errors="coerce")
    df["note"] = df.get("note", "").astype(str)

    if "multiple" in df.columns:
        df["multiple"] = pd.to_numeric(df["multiple"], errors="coerce")
    else:
        df["multiple"] = np.nan

    if "target_price" in df.columns:
        df["target_price"] = pd.to_numeric(df["target_price"], errors="coerce")
    else:
        df["target_price"] = np.nan

    df = df.dropna(subset=["project", "stage", "sell_pct"])
    return df


def load_cash(path: str) -> pd.DataFrame:
    """
    data_cash.csv:
    asset,amount
    USDC,4543
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

    prices_by_id, source, _ = fetch_coingecko_prices(ids=ids, vs_currency=vs_currency)

    live_prices: List[Optional[float]] = []
    for p in pos["project"].tolist():
        _id = proj_to_id.get(p)
        val: Optional[float] = None
        if _id and _id in prices_by_id:
            val = prices_by_id[_id]
        elif p in FALLBACK_PRICE_BY_PROJECT:
            val = FALLBACK_PRICE_BY_PROJECT[p]
        live_prices.append(val)

    out = pos.copy()
    out["price_live"] = live_prices
    out["value_live"] = out["tokens_total"] * out["price_live"]
    out["pnl_$"] = out["value_live"] - out["invested_total"]
    out["pnl_%"] = np.where(out["invested_total"] > 0, (out["pnl_$"] / out["invested_total"]) * 100, np.nan)
    return out, source


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
    manual_refresh = st.button("🔄 Rafraîchir maintenant")
    st.divider()
    show_trades = st.toggle("Voir le détail des achats (DCA)", value=True)
    st.caption("Modifie data_trades.csv / data_targets.csv / data_cash.csv pour mettre à jour.")

if auto_refresh:
    st_autorefresh(interval=60_000, key="autorefresh_60s")

if manual_refresh:
    st.cache_data.clear()
    st.rerun()

# Load data
trades = load_trades(TRADES_FILE)
targets = load_targets(TARGETS_FILE)
cash_df = load_cash(CASH_FILE)

# Compute positions
positions = consolidate_positions(trades)
positions_live, price_source = attach_live_prices(positions, vs_currency)

# Cash: accept USDC/USDT/DAI and sum (stablecoins)
stable_assets = {"USDC", "USDT", "DAI"}
cash_total = 0.0
cash_breakdown = []
if not cash_df.empty:
    for asset in cash_df["asset"].unique():
        amt = float(cash_df.loc[cash_df["asset"] == asset, "amount"].sum())
        if asset in stable_assets:
            cash_total += amt
        cash_breakdown.append((asset, amt))

# KPI logic
invested_total = float(positions_live["invested_total"].sum())  # invested only
value_positions_live = float(np.nansum(positions_live["value_live"].to_numpy()))  # positions only
value_total_live = value_positions_live + cash_total
pnl_positions = value_positions_live - invested_total
pnl_positions_pct = (pnl_positions / invested_total * 100) if invested_total > 0 else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Investi", money(invested_total))
k2.metric("Cash (stables)", money(cash_total))
k3.metric("Valeur totale (live)", money(value_total_live))
k4.metric("PnL (positions)", money(pnl_positions))
k5.metric("PnL % (positions)", pct(pnl_positions_pct))

st.markdown(
    f'<span class="badge">Source prix: {price_source} • cache 60s</span> '
    f'<span class="muted">Si un prix est absent, il s’affichera en —</span>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# For table + pie: represent cash as one line (USDC cash)
cash_label = "USDC (cash)"  # your case
cash_row = pd.DataFrame([{
    "project": cash_label,
    "tokens_total": cash_total,   # dollars
    "invested_total": 0.0,
    "avg_entry": np.nan,
    "price_live": 1.0,
    "value_live": cash_total,
    "pnl_$": np.nan,
    "pnl_%": np.nan,
}])

positions_all = pd.concat([positions_live, cash_row], ignore_index=True)

# Map PRU per project for auto targets
pru_by_proj = dict(zip(positions_live["project"], positions_live["avg_entry"]))

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("📌 Positions consolidées")

    df_show = positions_all.copy()

    df_show["Montant / Tokens"] = df_show.apply(
        lambda r: money(r["tokens_total"]) if r["project"] == cash_label else qty_tokens(r["tokens_total"]),
        axis=1,
    )
    df_show["PRU (DCA)"] = df_show["avg_entry"].map(price)
    df_show.loc[df_show["project"] == cash_label, "PRU (DCA)"] = "—"

    df_show["Prix live"] = df_show["price_live"].map(price)

    df_show["Investi"] = df_show["invested_total"].map(money)
    df_show.loc[df_show["project"] == cash_label, "Investi"] = "—"

    df_show["Valeur"] = df_show["value_live"].map(money)

    df_show["PnL"] = df_show["pnl_$"].map(money)
    df_show["PnL %"] = df_show["pnl_%"].map(pct)
    df_show.loc[df_show["project"] == cash_label, ["PnL", "PnL %"]] = ["—", "—"]

    cols = ["project", "Montant / Tokens", "PRU (DCA)", "Prix live", "Investi", "Valeur", "PnL", "PnL %"]
    st.dataframe(df_show[cols].rename(columns={"project": "Token"}), width="stretch", hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("🎯 Objectifs (TP) — targets auto via PRU (DCA)")

    cur_price_by_proj = dict(zip(positions_live["project"], positions_live["price_live"]))
    inv_by_proj = dict(zip(positions_live["project"], positions_live["invested_total"]))
    tok_by_proj = dict(zip(positions_live["project"], positions_live["tokens_total"]))

    for proj in positions_live["project"].tolist():
        st.markdown(f"### {proj}")

        cur = cur_price_by_proj.get(proj)
        invested_proj = float(inv_by_proj.get(proj, 0.0))
        tokens_total_proj = float(tok_by_proj.get(proj, 0.0))
        pru = pru_by_proj.get(proj)

        t = targets[targets["project"] == proj].copy().sort_values("stage")
        if t.empty:
            st.info("Pas d'objectifs configurés.")
            continue

        remaining = tokens_total_proj
        cumulative_cash = 0.0

        for _, row in t.iterrows():
            stage = str(row["stage"])
            sell_pct = float(row["sell_pct"])
            note = str(row.get("note", "")).strip()

            # Auto target price from PRU * multiple (preferred)
            multiple = row.get("multiple", np.nan)
            target_price = row.get("target_price", np.nan)

            tgt: Optional[float] = None
            if is_number(multiple) and is_number(pru):
                tgt = float(pru) * float(multiple)
            elif is_number(target_price):
                tgt = float(target_price)

            if not is_number(tgt):
                st.warning("Impossible de calculer la cible (PRU manquant).")
                continue

            sold_tokens = remaining * sell_pct
            remaining_after = remaining - sold_tokens

            cash_if_hit = sold_tokens * float(tgt)
            cumulative_cash += cash_if_hit

            mult_label = f"x{int(multiple)}" if is_number(multiple) else ""
            st.write(f"**{stage}** {mult_label} • cible: **{price(tgt)}** • vente: **{int(sell_pct*100)}% du restant**")
            st.progress(progress(cur, float(tgt)))
            if note:
                st.caption(note)

            c1, c2, c3 = st.columns([1.05, 1.05, 1.25])

            with c1:
                st.metric("Tokens vendus (étape)", qty_tokens(sold_tokens))
                st.metric("Tokens restants", qty_tokens(remaining_after))

            with c2:
                st.metric("Cash si cible atteinte (étape)", money(cash_if_hit))
                st.metric("Valeur du restant (à la cible)", money(remaining_after * float(tgt)))

            with c3:
                hit_live = is_number(cur) and float(cur) >= float(tgt)
                st.metric("Target atteinte (prix live)", "Oui" if hit_live else "Non")

                if sold_tokens > 0 and invested_proj > 0:
                    be_price = invested_proj / sold_tokens
                    st.metric("Prix récup. mise (via étape)", price(be_price))
                    rec_at_target = float(tgt) >= be_price
                    st.metric("À la cible, récup. mise ?", "Oui" if rec_at_target else "Non")
                else:
                    st.metric("Prix récup. mise (via étape)", "—")
                    st.metric("À la cible, récup. mise ?", "—")

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

    # If only cash is visible because prices are missing, warn
    if pie_df.empty or (len(pie_df) == 1 and pie_df.iloc[0]["project"] == cash_label and len(positions_live) > 0):
        st.warning("Prix manquants pour certaines positions : la répartition peut être incomplète.")
    else:
        fig = px.pie(pie_df, names="project", values="value_live", hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("📉 PnL par token (positions)")

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

st.caption("🛠️ Targets auto: x2/x4 se recalculent automatiquement avec le PRU (DCA) quand tu ajoutes un buy.")
