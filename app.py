import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ---------------------------
# Files
# ---------------------------
TRANSACTIONS_FILE = "data_transactions.csv"
CASH_FILE = "data_cash.csv"

COINGECKO_ID_BY_PROJECT = {
    "TAO": "bittensor",
    "NOCK": "nockchain",
}

BINANCE_SYMBOL_BY_PROJECT = {
    "TAO": "TAOUSDT",
}

DEXSCREENER_PAIR_BY_PROJECT = {
    "NOCK": {
        "chain": "base",
        "pair": "0x85f1aa3a70fedd1c52705c15baed143e675cd626",
    }
}

# ---------------------------
# Helpers
# ---------------------------
def is_number(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))

def money(x):
    if not is_number(x):
        return "—"
    return f"${float(x):,.2f}"

def price(x):
    if not is_number(x):
        return "—"
    x = float(x)
    if abs(x) < 0.1:
        return f"${x:,.6f}"
    if abs(x) < 1:
        return f"${x:,.4f}"
    return f"${x:,.2f}"

def qty_tokens(x):
    if not is_number(x):
        return "—"
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.4f}"

def pct(x):
    if not is_number(x):
        return "—"
    return f"{float(x):,.2f}%"

def tx_badge_html(tx_type: str):
    if tx_type == "BUY":
        return '<span style="color:#22c55e;font-weight:700;">BUY</span>'
    if tx_type == "SELL":
        return '<span style="color:#ef4444;font-weight:700;">SELL</span>'
    return tx_type


# ---------------------------
# Loaders
# ---------------------------
def load_transactions(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["project"] = df["project"].str.upper().str.strip()
    df["type"] = df["type"].str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"])
    df["unit_price_usd"] = pd.to_numeric(df["unit_price_usd"])
    df["fees_usd"] = pd.to_numeric(df["fees_usd"], errors="coerce").fillna(0)
    df["note"] = df.get("note", "").astype(str)
    return df

def load_cash(path):
    try:
        df = pd.read_csv(path)
        df["asset"] = df["asset"].str.upper().str.strip()
        df["amount"] = pd.to_numeric(df["amount"])
        return df
    except:
        return pd.DataFrame(columns=["asset", "amount"])


# ---------------------------
# Core logic
# ---------------------------
def build_portfolio_and_sales(df):
    positions = []
    sales = []
    warnings = []

    for project, g in df.groupby("project"):
        g = g.sort_values("date")

        qty = 0.0
        cost = 0.0
        realized = 0.0

        for _, r in g.iterrows():
            q = r["quantity"]
            px = r["unit_price_usd"]
            fees = r["fees_usd"]

            if r["type"] == "BUY":
                qty += q
                cost += q * px + fees

            elif r["type"] == "SELL":
                if q > qty:
                    warnings.append(f"{project}: oversell détecté")

                avg = cost / qty if qty > 0 else 0
                cost_sold = min(q, qty) * avg

                proceeds = q * px - fees
                pnl = proceeds - cost_sold

                realized += pnl

                qty -= min(q, qty)
                cost -= cost_sold

                sales.append({
                    "date": r["date"],
                    "project": project,
                    "type": "SELL",
                    "quantity": q,
                    "sell_price": px,
                    "net_proceeds": proceeds,
                    "cost_basis_sold": cost_sold,
                    "realized_pnl": pnl,
                    "note": r["note"]
                })

        positions.append({
            "project": project,
            "qty_current": qty,
            "cost_basis_remaining": cost,
            "avg_cost_current": cost / qty if qty > 0 else np.nan,
            "realized_pnl": realized
        })

    return pd.DataFrame(positions), pd.DataFrame(sales), warnings


# ---------------------------
# Prices
# ---------------------------
def fetch_price(project):
    try:
        if project == "TAO":
            r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "TAOUSDT"})
            return float(r.json()["price"])
        if project == "NOCK":
            url = "https://api.dexscreener.com/latest/dex/pairs/base/0x85f1aa3a70fedd1c52705c15baed143e675cd626"
            r = requests.get(url)
            return float(r.json()["pairs"][0]["priceUsd"])
    except:
        return None


# ---------------------------
# App
# ---------------------------
st.set_page_config(layout="wide")
st.title("📈 Dashboard BW")

transactions = load_transactions(TRANSACTIONS_FILE)
cash_df = load_cash(CASH_FILE)

positions_raw, sales_df, warnings = build_portfolio_and_sales(transactions)

for w in warnings:
    st.warning(w)

# Positions ouvertes
positions = positions_raw[positions_raw["qty_current"] > 0].copy()

# Prix live
prices = []
for p in positions["project"]:
    prices.append(fetch_price(p))

positions["price_live"] = prices
positions["value_live"] = positions["qty_current"] * positions["price_live"]
positions["pnl_unrealized"] = positions["value_live"] - positions["cost_basis_remaining"]

# Cash
cash_total = cash_df["amount"].sum() if not cash_df.empty else 0

# KPIs
cost_open = positions["cost_basis_remaining"].sum()
value_open = positions["value_live"].sum()
pnl_latent = value_open - cost_open
pnl_realized = sales_df["realized_pnl"].sum() if not sales_df.empty else 0
pnl_total = pnl_realized + pnl_latent

# ---------------------------
# HEADER METRICS (UPDATED)
# ---------------------------
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Coût positions ouvertes", money(cost_open))
k2.metric("Cash dispo", money(cash_total))
k3.metric("PnL réalisé", money(pnl_realized))
k4.metric("PnL latent", money(pnl_latent))
k5.metric("PnL total", money(pnl_total))


# ---------------------------
# Positions
# ---------------------------
st.subheader("📊 Positions")

if not positions.empty:
    df_show = positions.copy()

    df_show["Tokens"] = df_show["qty_current"].map(qty_tokens)
    df_show["PRU restant"] = df_show["avg_cost_current"].map(price)
    df_show["Prix live"] = df_show["price_live"].map(price)
    df_show["Valeur"] = df_show["value_live"].map(money)
    df_show["PnL latent"] = df_show["pnl_unrealized"].map(money)
    df_show["PnL réalisé cumulé"] = df_show["realized_pnl"].map(money)

    st.dataframe(
        df_show[["project","Tokens","PRU restant","Prix live","Valeur","PnL latent","PnL réalisé cumulé"]]
        .rename(columns={"project":"Token"}),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------
# Cash display (simple)
# ---------------------------
if not cash_df.empty:
    st.markdown("### 💵 Cash")
    st.dataframe(cash_df, use_container_width=True, hide_index=True)


# ---------------------------
# Sales
# ---------------------------
st.subheader("✅ Ventes réalisées")

if not sales_df.empty:
    s = sales_df.copy()
    s["Date"] = s["date"].dt.strftime("%Y-%m-%d")
    s["Type"] = s["type"].map(tx_badge_html)
    s["Quantité"] = s["quantity"].map(qty_tokens)
    s["Prix"] = s["sell_price"].map(price)
    s["PnL"] = s["realized_pnl"].map(money)

    st.markdown(
        s[["Date","project","Type","Quantité","Prix","PnL","note"]]
        .rename(columns={"project":"Token","note":"Note"})
        .to_html(escape=False,index=False),
        unsafe_allow_html=True
    )
