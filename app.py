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

DEFAULT_VS_CURRENCY = "usd"

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
    },
    "FAI": {
        "chain": "base",
        "pair": "0x5447f7fe76894d98753a0a6d69b9cb840037c13d",
    },
    "OCT": {
        "chain": "ethereum",
        "pair": "0x5eb459d3fc44f3f412ef43f93fa1e44ecb4ca9cb62a16bcbd94b5d0b834ff854",
    },
}

FALLBACK_PRICE_BY_PROJECT: Dict[str, float] = {}


# ---------------------------
# Styles
# ---------------------------
PREMIUM_CSS = """
<style>
h1, h2, h3 { letter-spacing: -0.02em; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

button[title*="Copy link"], button[aria-label*="Copy link"] {
  display: none !important;
}

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

.muted { opacity: 0.75; }

a.stMarkdownAnchor,
a[data-testid="stMarkdownAnchor"],
.stMarkdown a[href^="#"],
h1 a[href^="#"], h2 a[href^="#"], h3 a[href^="#"] {
  display: none !important;
}

/* HTML tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  overflow: hidden;
}
thead tr {
  background: rgba(255,255,255,0.04);
}
thead th {
  text-align: left !important;
  font-weight: 700;
}
tbody td {
  text-align: left !important;
}
tbody tr:hover {
  background: rgba(255,255,255,0.02);
}
</style>
"""


# ---------------------------
# Helpers
# ---------------------------
def is_number(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def money(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    return f"${float(x):,.2f}"


def money_rounded(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    return f"${int(round(float(x))):,}"


def price(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    x = float(x)
    if abs(x) < 0.1:
        return f"${x:,.6f}"
    if abs(x) < 1:
        return f"${x:,.4f}"
    return f"${x:,.2f}"


def qty_tokens(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.4f}"


def pct(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    return f"{float(x):,.2f}%"


def tx_badge_html(tx_type: str) -> str:
    tx_type = str(tx_type).upper().strip()
    if tx_type == "BUY":
        return '<span style="color:#22c55e;font-weight:700;">BUY</span>'
    if tx_type == "SELL":
        return '<span style="color:#ef4444;font-weight:700;">SELL</span>'
    return tx_type


def pnl_html(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    value = float(x)
    color = "#22c55e" if value > 0 else "#ef4444" if value < 0 else "#e5e7eb"
    return f'<span style="color:{color};font-weight:700;">{money(value)}</span>'


def pnl_color_html(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    value = float(x)
    color = "#22c55e" if value > 0 else "#ef4444" if value < 0 else "#e5e7eb"
    return f'<span style="color:{color};font-weight:600;">{money(value)}</span>'


def pct_color_html(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    value = float(x)
    color = "#22c55e" if value > 0 else "#ef4444" if value < 0 else "#e5e7eb"
    return f'<span style="color:{color};font-weight:600;">{value:,.2f}%</span>'


def make_html_table(df: pd.DataFrame) -> str:
    return df.to_html(escape=False, index=False)


# ---------------------------
# Price fetchers
# ---------------------------
@st.cache_data(ttl=20, show_spinner=False)
def fetch_binance_price(symbol: str) -> Optional[float]:
    base_urls = [
        "https://api.binance.com",
        "https://data-api.binance.vision",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DashboardBW/1.0)",
        "Accept": "application/json",
    }

    for base in base_urls:
        url = f"{base}/api/v3/ticker/price"
        try:
            r = requests.get(url, params={"symbol": symbol}, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            p = data.get("price")
            if p is not None:
                return float(p)
        except Exception:
            continue
    return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_coingecko_prices(ids: List[str], vs_currency: str) -> Tuple[Dict[str, float], str, int]:
    if not ids:
        return {}, "coingecko", int(time.time())

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": vs_currency}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        out: Dict[str, float] = {}
        for _id in ids:
            if _id in data and vs_currency in data[_id]:
                out[_id] = float(data[_id][vs_currency])
        return out, "coingecko", int(time.time())
    except Exception:
        return {}, "coingecko_error", int(time.time())


@st.cache_data(ttl=30, show_spinner=False)
def fetch_dexscreener_pair_price_usd(chain: str, pair: str) -> Optional[float]:
    url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair}"
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        pairs = data.get("pairs") or []
        if not pairs:
            return None
        px_ = pairs[0].get("priceUsd")
        return float(px_) if px_ is not None else None
    except Exception:
        return None


def attach_live_prices(pos: pd.DataFrame, vs_currency: str) -> Tuple[pd.DataFrame, str]:
    vs = vs_currency.lower()

    ids: List[str] = []
    proj_to_id: Dict[str, str] = {}

    for p in pos["project"].tolist():
        if p in DEXSCREENER_PAIR_BY_PROJECT:
            continue
        if p in BINANCE_SYMBOL_BY_PROJECT and vs == "usd":
            continue
        _id = COINGECKO_ID_BY_PROJECT.get(p)
        if _id:
            ids.append(_id)
            proj_to_id[p] = _id

    prices_by_id, _, _ = fetch_coingecko_prices(ids=ids, vs_currency=vs_currency)

    if "last_prices" not in st.session_state:
        st.session_state["last_prices"] = {}

    live_prices: List[Optional[float]] = []
    for p in pos["project"].tolist():
        val: Optional[float] = None

        if p in DEXSCREENER_PAIR_BY_PROJECT and vs == "usd":
            cfg = DEXSCREENER_PAIR_BY_PROJECT[p]
            val = fetch_dexscreener_pair_price_usd(cfg["chain"], cfg["pair"])

        if val is None and p in BINANCE_SYMBOL_BY_PROJECT and vs == "usd":
            val = fetch_binance_price(BINANCE_SYMBOL_BY_PROJECT[p])

        if val is None:
            _id = proj_to_id.get(p)
            if _id and _id in prices_by_id:
                val = prices_by_id[_id]

        if val is None and p in FALLBACK_PRICE_BY_PROJECT:
            val = FALLBACK_PRICE_BY_PROJECT[p]

        if val is None and p in st.session_state["last_prices"]:
            val = st.session_state["last_prices"][p]

        if val is not None:
            st.session_state["last_prices"][p] = float(val)

        live_prices.append(val)

    out = pos.copy()
    out["price_live"] = live_prices
    out["value_live"] = out["qty_current"] * out["price_live"]
    out["pnl_unrealized_$"] = out["value_live"] - out["cost_basis_remaining"]
    out["pnl_unrealized_%"] = np.where(
        out["cost_basis_remaining"] > 0,
        (out["pnl_unrealized_$"] / out["cost_basis_remaining"]) * 100,
        np.nan,
    )
    return out, "live"


# ---------------------------
# Data loaders
# ---------------------------
def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["project"] = df["project"].astype(str).str.upper().str.strip()
    df["type"] = df["type"].astype(str).str.upper().str.strip()

    for col in ["quantity", "unit_price_usd", "fees_usd"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "note" not in df.columns:
        df["note"] = ""
    df["note"] = df["note"].fillna("").astype(str)

    df = df.dropna(subset=["date", "project", "type", "quantity", "unit_price_usd"])
    df = df[df["type"].isin(["BUY", "SELL"])].copy()
    return df


def load_cash(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df["asset"] = df["asset"].astype(str).str.upper().str.strip()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["asset", "amount"])
        return df
    except Exception:
        return pd.DataFrame(columns=["asset", "amount"])


# ---------------------------
# Core accounting logic
# Weighted average cost basis
# ---------------------------
def build_portfolio_and_sales(transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if transactions.empty:
        empty_positions = pd.DataFrame(columns=[
            "project",
            "qty_bought",
            "qty_sold",
            "qty_current",
            "buy_cost_gross",
            "sell_proceeds_gross",
            "fees_total",
            "avg_entry_all_buys",
            "avg_cost_current",
            "cost_basis_remaining",
            "realized_pnl",
            "last_tx_date",
        ])
        empty_sales = pd.DataFrame(columns=[
            "date",
            "project",
            "type",
            "quantity",
            "sell_price",
            "gross_proceeds",
            "fees_usd",
            "net_proceeds",
            "cost_basis_sold",
            "realized_pnl",
            "note",
        ])
        return empty_positions, empty_sales, []

    positions_rows = []
    sales_rows = []
    warnings_list: List[str] = []

    for project, grp in transactions.groupby("project", sort=True):
        grp = grp.sort_values("date").reset_index(drop=True)

        qty_held = 0.0
        cost_basis_held = 0.0

        qty_bought = 0.0
        qty_sold = 0.0
        buy_cost_gross = 0.0
        sell_proceeds_gross = 0.0
        fees_total = 0.0
        realized_pnl_total = 0.0

        for _, row in grp.iterrows():
            tx_type = row["type"]
            qty = float(row["quantity"])
            px = float(row["unit_price_usd"])
            fees = float(row["fees_usd"]) if is_number(row["fees_usd"]) else 0.0
            note = row.get("note", "")

            if qty <= 0:
                continue

            if tx_type == "BUY":
                gross = qty * px
                total_cost = gross + fees

                qty_bought += qty
                buy_cost_gross += total_cost
                fees_total += fees

                qty_held += qty
                cost_basis_held += total_cost

            elif tx_type == "SELL":
                available_before_sell = qty_held

                if qty > available_before_sell + 1e-12:
                    warnings_list.append(
                        f"{project}: vente de {qty_tokens(qty)} alors que seulement {qty_tokens(available_before_sell)} étaient disponibles à cette date."
                    )

                avg_cost_before = (cost_basis_held / qty_held) if qty_held > 0 else 0.0
                qty_to_sell = min(qty, qty_held) if qty_held > 0 else 0.0
                cost_basis_sold = qty_to_sell * avg_cost_before

                gross_proceeds = qty * px
                net_proceeds = gross_proceeds - fees
                realized_pnl = net_proceeds - cost_basis_sold

                qty_sold += qty
                sell_proceeds_gross += gross_proceeds
                fees_total += fees
                realized_pnl_total += realized_pnl

                qty_held = qty_held - qty_to_sell
                cost_basis_held = cost_basis_held - cost_basis_sold

                if abs(qty_held) < 1e-12:
                    qty_held = 0.0
                if abs(cost_basis_held) < 1e-12:
                    cost_basis_held = 0.0

                sales_rows.append({
                    "date": row["date"],
                    "project": project,
                    "type": "SELL",
                    "quantity": qty,
                    "sell_price": px,
                    "gross_proceeds": gross_proceeds,
                    "fees_usd": fees,
                    "net_proceeds": net_proceeds,
                    "cost_basis_sold": cost_basis_sold,
                    "realized_pnl": realized_pnl,
                    "note": note,
                })

        avg_entry_all_buys = (buy_cost_gross / qty_bought) if qty_bought > 0 else np.nan
        avg_cost_current = (cost_basis_held / qty_held) if qty_held > 0 else np.nan

        positions_rows.append({
            "project": project,
            "qty_bought": qty_bought,
            "qty_sold": qty_sold,
            "qty_current": qty_held,
            "buy_cost_gross": buy_cost_gross,
            "sell_proceeds_gross": sell_proceeds_gross,
            "fees_total": fees_total,
            "avg_entry_all_buys": avg_entry_all_buys,
            "avg_cost_current": avg_cost_current,
            "cost_basis_remaining": cost_basis_held,
            "realized_pnl": realized_pnl_total,
            "last_tx_date": grp["date"].max(),
        })

    positions = pd.DataFrame(positions_rows)
    sales = pd.DataFrame(sales_rows)

    if not sales.empty:
        sales = sales.sort_values("date", ascending=False).reset_index(drop=True)

    positions = positions.sort_values("project").reset_index(drop=True)
    return positions, sales, warnings_list


# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Dashboard BW", page_icon="📈", layout="wide")
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

st.title("📈 Dashboard BW")

with st.sidebar:
    st.header("⚙️ Paramètres")
    vs_currency = st.selectbox("Devise", options=["usd", "eur"], index=0, format_func=lambda x: x.upper())
    if vs_currency.lower() == "eur":
        st.info("NOCK/TAO/FAI/OCT sont pricés en USD en priorité. En EUR, certains prix peuvent être indisponibles.")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=True)
    manual_refresh = st.button("🔄 Rafraîchir maintenant")
    st.divider()
    show_transactions = st.toggle("Voir le journal complet", value=True)
    st.caption("Modifie data_transactions.csv et data_cash.csv")

if auto_refresh:
    st_autorefresh(interval=60_000, key="autorefresh_60s")

if manual_refresh:
    st.cache_data.clear()
    st.rerun()

transactions = load_transactions(TRANSACTIONS_FILE)
cash_df = load_cash(CASH_FILE)

positions_raw, sales_df, data_warnings = build_portfolio_and_sales(transactions)

for msg in data_warnings:
    st.warning(msg)

positions_open = positions_raw[positions_raw["qty_current"] > 1e-12].copy()
positions_live, _ = attach_live_prices(positions_open, vs_currency) if not positions_open.empty else (positions_open.copy(), "live")

cash_assets = {"USDC", "USDT", "DAI", "RAKBANK"}

cash_total = 0.0
cash_rows = []

if not cash_df.empty:
    for _, row in cash_df.iterrows():
        asset = str(row["asset"]).upper().strip()
        amount = float(row["amount"])

        if asset in cash_assets:
            cash_total += amount
            cash_rows.append({
                "project": asset,
                "qty_current": amount,
                "avg_cost_current": np.nan,
                "buy_cost_gross": np.nan,
                "price_live": 1.0,
                "cost_basis_remaining": 0.0,
                "value_live": amount,
                "pnl_unrealized_$": np.nan,
                "pnl_unrealized_%": np.nan,
                "realized_pnl": np.nan,
                "profit_total_$": np.nan,
                "profit_total_%": np.nan,
            })

cash_positions_df = pd.DataFrame(cash_rows)

# Profit réel des positions ouvertes = somme des lignes ouvertes (réalisé + latent par token)
if not positions_live.empty:
    positions_live["invested_real"] = positions_live["buy_cost_gross"].fillna(0) - positions_live["sell_proceeds_gross"].fillna(0)
    positions_live["profit_total_$"] = positions_live["value_live"].fillna(0) - positions_live["invested_real"].fillna(0)
    positions_live["profit_total_%"] = np.where(
        positions_live["invested_real"] > 0,
        (positions_live["profit_total_$"] / positions_live["invested_real"]) * 100,
        np.nan,
    )
else:
    positions_live["invested_real"] = []
    positions_live["profit_total_$"] = []
    positions_live["profit_total_%"] = []

profit_open_positions_real = float(np.nansum(positions_live["profit_total_$"].to_numpy())) if not positions_live.empty else 0.0
realized_pnl_total = float(sales_df["realized_pnl"].sum()) if not sales_df.empty else 0.0

# Ici on retire le réalisé déjà compté dans les positions ouvertes pour éviter le double count
pnl_total_real = realized_pnl_total + profit_open_positions_real

total_current_value = cash_total + (float(np.nansum(positions_live["value_live"].to_numpy())) if not positions_live.empty else 0.0)

# Ancien latent comptable gardé seulement si besoin un jour, mais non affiché en haut
pnl_unrealized_total_accounting = (
    float(np.nansum(positions_live["pnl_unrealized_$"].to_numpy()))
    if not positions_live.empty else 0.0
)

# ---------------------------
# Top metrics
# ---------------------------
pnl_color = "#22c55e" if pnl_total_real > 0 else "#ef4444" if pnl_total_real < 0 else "#e5e7eb"
open_positions_color = "#22c55e" if profit_open_positions_real > 0 else "#ef4444" if profit_open_positions_real < 0 else "#e5e7eb"

cards = [
    {
        "label": "Profit net total actuel",
        "value": money(pnl_total_real),
        "value_color": pnl_color,
        "value_opacity": 1.0,
        "detail_html": f"""
            <div style="
                font-size: 10px;
                line-height: 1.45;
                margin-top: 8px;
                color: #e5e7eb;
            ">
                <span style="color: rgba(229,231,235,0.70);">
                    dont :
                    <span style="font-weight:600; color: rgba(229,231,235,0.90);">
                        {("+" if realized_pnl_total > 0 else "")}{money(realized_pnl_total)}
                    </span>
                    encaissé
                    <span style="color: rgba(229,231,235,0.45);">•</span>
                    <span style="font-weight:600; color: rgba(229,231,235,0.90);">
                        {money(profit_open_positions_real)}
                    </span>
                    restant sur positions ouvertes
                </span>
                <br><br><br>
                <span style="font-size:14px; color: rgba(229,231,235,0.70);">
                    Total actuel (cash + positions en cours) :
                </span>
                <b style="font-size:14px; color:#ffffff;">
                    {money_rounded(total_current_value)}
                </b>
            </div>
        """,
    },
    {
        "label": "Cash disponible (rakbank + stablecoins)",
        "value": money_rounded(cash_total),
        "value_color": "#e5e7eb",
        "value_opacity": 1.0,
        "detail_html": "",
    },
    {
        "label": "Gain / perte positions en cours",
        "value": money(profit_open_positions_real),
        "value_color": open_positions_color,
        "value_opacity": 0.20,
        "detail_html": "",
    },
]

cols = st.columns(3)

for col, card in zip(cols, cards):
    with col:
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 14px;
                padding: 18px 16px 16px 16px;
                min-height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                box-sizing: border-box;
            ">
                <div style="
                    font-size: 14px;
                    line-height: 1.2;
                    opacity: {0.85 if card["value_opacity"] == 1.0 else 0.60};
                    margin-bottom: 10px;
                    color: #e5e7eb;
                    font-weight: 500;
                ">
                    {card["label"]}
                </div>
                <div style="
                    font-size: 32px;
                    line-height: 1.15;
                    font-weight: 700;
                    color: {card["value_color"]};
                    opacity: {card["value_opacity"]};
                    margin: 0;
                    padding: 0;
                ">
                    {card["value"]}
                </div>
                {card["detail_html"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div style="height: 25px;"></div>', unsafe_allow_html=True)

tab_portefeuille, tab_sales = st.tabs(["📊 Portefeuille", "✅ Ventes réalisées"])

positions_all = positions_live.copy()
if not cash_positions_df.empty:
    positions_all = pd.concat([positions_all, cash_positions_df], ignore_index=True)

all_labels_for_colors = positions_all["project"].astype(str).tolist() if not positions_all.empty else []
palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(all_labels_for_colors)}

color_map["RAKBANK"] = "#60a5fa"

# ---------------------------
# TAB 1 — Portefeuille
# ---------------------------
with tab_portefeuille:
    st.subheader("📌 Positions")

    if positions_all.empty:
        st.info("Aucune position ouverte.")
    else:
        df_show = positions_all.copy()

        df_show["Quantité de tokens"] = df_show["qty_current"].map(qty_tokens)
        df_show["Prix achat moyen"] = np.where(
            df_show["qty_current"] > 0,
            df_show["invested_real"] / df_show["qty_current"],
            np.nan,
        )
        df_show["Prix achat moyen"] = pd.Series(df_show["Prix achat moyen"], index=df_show.index).map(price)
        df_show["Prix actuel"] = df_show["price_live"].map(price)
        df_show["Investi"] = df_show["invested_real"].map(money)
        df_show["Valeur actuelle"] = df_show["value_live"].map(money)
        df_show["Profit en cours"] = df_show["profit_total_$"].map(pnl_color_html)
        df_show["Profit %"] = df_show["profit_total_%"].map(pct_color_html)

        is_cash_row = df_show["project"].isin(list(cash_assets))
        df_show.loc[is_cash_row, ["Prix achat moyen", "Profit en cours", "Profit %"]] = ["—", "—", "—"]
        df_show.loc[is_cash_row, "Valeur actuelle"] = df_show.loc[is_cash_row, "value_live"].map(money_rounded)

        cols = [
            "project",
            "Quantité de tokens",
            "Prix achat moyen",
            "Prix actuel",
            "Investi",
            "Valeur actuelle",
            "Profit en cours",
            "Profit %",
        ]

        positions_html = df_show[cols].rename(columns={"project": "Projet"})
        st.markdown(make_html_table(positions_html), unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown('<div style="height: 25px;"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("📊 Répartition")
            pie_df = positions_all.dropna(subset=["value_live"]).copy()
            if pie_df.empty:
                st.info("Pas de données de valorisation.")
            else:
                fig = px.pie(
                    pie_df,
                    names="project",
                    values="value_live",
                    hole=0.45,
                    color="project",
                    color_discrete_map=color_map,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📉 Profit en cours par token")
            bar_df = positions_live.copy()
            bar_df = bar_df.dropna(subset=["profit_total_$"])

            if not bar_df.empty:
                fig2 = px.bar(
                    bar_df,
                    x="project",
                    y="profit_total_$",
                    color="project",
                    color_discrete_map=color_map,
                )
                fig2.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis_title="Token",
                    yaxis_title="Profit en cours total ($)",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Profit en cours indisponible.")

    st.markdown('<div style="height: 75px;"></div>', unsafe_allow_html=True)

    if show_transactions:
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.subheader("🧾 Journal complet")

        tx_show = transactions.copy().sort_values("date", ascending=False)
        tx_show["Date"] = tx_show["date"].dt.strftime("%Y-%m-%d")
        tx_show["Type"] = tx_show["type"].map(tx_badge_html)
        tx_show["Quantité"] = tx_show["quantity"].map(qty_tokens)
        tx_show["Prix unitaire"] = tx_show["unit_price_usd"].map(price)
        tx_show["Montant brut"] = (tx_show["quantity"] * tx_show["unit_price_usd"]).map(money)

        tx_html = tx_show[[
            "Date", "project", "Type", "Quantité", "Prix unitaire", "Montant brut", "note"
        ]].rename(columns={
            "project": "Token",
            "note": "Note",
        })

        st.markdown(make_html_table(tx_html), unsafe_allow_html=True)


# ---------------------------
# TAB 2 — Ventes réalisées
# ---------------------------
with tab_sales:
    pnl_realized_html = pnl_html(realized_pnl_total)
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 14px 16px 12px 16px;
            margin-bottom: 18px;
            max-width: 420px;
        ">
            <div style="font-size: 14px; opacity: 0.85; margin-bottom: 6px;">Profit encaissé total</div>
            <div style="font-size: 24px; font-weight: 700;">{pnl_realized_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sales_df.empty:
        st.info("Aucune vente enregistrée.")
    else:
        st.subheader("📊 Synthèse par token")

        summary = sales_df.groupby("project", as_index=False).agg(
            quantity_sold=("quantity", "sum"),
            net_proceeds=("net_proceeds", "sum"),
            cost_basis_sold=("cost_basis_sold", "sum"),
            realized_pnl=("realized_pnl", "sum"),
        )
        
        # Trier du plus gros gain au plus gros loss
        summary = summary.sort_values("realized_pnl", ascending=False).reset_index(drop=True)
        
        summary["Quantité vendue"] = summary["quantity_sold"].map(qty_tokens)
        summary["Argent récupéré"] = summary["net_proceeds"].map(money)
        summary["Montant initial investi"] = summary["cost_basis_sold"].map(money)
        summary["Gain / Perte"] = summary["realized_pnl"].map(pnl_color_html)
        
        summary_html = summary[[
            "project",
            "Quantité vendue",
            "Argent récupéré",
            "Montant initial investi",
            "Gain / Perte",
        ]].rename(columns={"project": "Token"})
        
        st.markdown(make_html_table(summary_html), unsafe_allow_html=True)
        
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.subheader("🧾 Historique des ventes")

        sales_show = sales_df.copy()
        sales_show["Date"] = sales_show["date"].dt.strftime("%Y-%m-%d")
        sales_show["Type"] = sales_show["type"].map(tx_badge_html)
        sales_show["Quantité vendue"] = sales_show["quantity"].map(qty_tokens)
        sales_show["Prix de vente"] = sales_show["sell_price"].map(price)
        sales_show["Argent récupéré"] = sales_show["net_proceeds"].map(money)
        sales_show["Montant initial investi"] = sales_show["cost_basis_sold"].map(money)
        sales_show["Gain / Perte"] = sales_show["realized_pnl"].map(pnl_html)

        sales_html = sales_show[[
            "Date",
            "project",
            "Type",
            "Quantité vendue",
            "Prix de vente",
            "Argent récupéré",
            "Montant initial investi",
            "Gain / Perte",
            "note",
        ]].rename(columns={
            "project": "Token",
            "note": "Note",
        })

        st.markdown(make_html_table(sales_html), unsafe_allow_html=True)
