import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
h1 { margin-bottom: -40px !important; }
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

.block-container {
    padding-top: 0.5rem !important;
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
    """Build open positions and realized sales with trade-cycle awareness.

    Cycle rule:
    - A cycle starts with a BUY when the token quantity is 0.
    - A partial SELL stays inside the same cycle.
    - When quantity returns to 0, the cycle is closed.
    - A later BUY starts a new cycle.

    Important:
    - Positions table uses ONLY the currently open cycle.
    - Ventes réalisées keeps ALL historical sales, with cycle_id available for summaries.
    """
    position_columns = [
        "project",
        "cycle_id",
        "cycle_start_date",
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
    ]
    sales_columns = [
        "date",
        "project",
        "cycle_id",
        "type",
        "quantity",
        "sell_price",
        "gross_proceeds",
        "fees_usd",
        "net_proceeds",
        "cost_basis_sold",
        "realized_pnl",
        "note",
    ]

    if transactions.empty:
        return pd.DataFrame(columns=position_columns), pd.DataFrame(columns=sales_columns), []

    positions_rows = []
    sales_rows = []
    warnings_list: List[str] = []

    for project, grp in transactions.groupby("project", sort=True):
        grp = grp.sort_values("date").reset_index(drop=True)

        cycle_id = 1
        cycle_start_date = None

        qty_held = 0.0
        cost_basis_held = 0.0

        qty_bought = 0.0
        qty_sold = 0.0
        buy_cost_gross = 0.0
        sell_proceeds_gross = 0.0
        fees_total = 0.0
        realized_pnl_total = 0.0
        last_tx_date = None

        def reset_cycle(next_cycle_id: int):
            return {
                "cycle_id": next_cycle_id,
                "cycle_start_date": None,
                "qty_held": 0.0,
                "cost_basis_held": 0.0,
                "qty_bought": 0.0,
                "qty_sold": 0.0,
                "buy_cost_gross": 0.0,
                "sell_proceeds_gross": 0.0,
                "fees_total": 0.0,
                "realized_pnl_total": 0.0,
                "last_tx_date": None,
            }

        for _, row in grp.iterrows():
            tx_type = row["type"]
            qty = float(row["quantity"])
            px = float(row["unit_price_usd"])
            fees = float(row["fees_usd"]) if is_number(row["fees_usd"]) else 0.0
            note = row.get("note", "")
            tx_date = row["date"]

            if qty <= 0:
                continue

            if tx_type == "BUY":
                if qty_held <= 1e-12 and qty_bought <= 1e-12:
                    cycle_start_date = tx_date

                gross = qty * px
                total_cost = gross + fees

                qty_bought += qty
                buy_cost_gross += total_cost
                fees_total += fees

                qty_held += qty
                cost_basis_held += total_cost
                last_tx_date = tx_date

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
                last_tx_date = tx_date

                qty_held = qty_held - qty_to_sell
                cost_basis_held = cost_basis_held - cost_basis_sold

                if abs(qty_held) < 1e-12:
                    qty_held = 0.0
                if abs(cost_basis_held) < 1e-12:
                    cost_basis_held = 0.0

                sales_rows.append({
                    "date": tx_date,
                    "project": project,
                    "cycle_id": cycle_id,
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

                # SELL 100% => cycle closes. Next BUY starts a new cycle.
                if qty_held <= 1e-12:
                    cycle_id += 1
                    cycle_start_date = None
                    qty_held = 0.0
                    cost_basis_held = 0.0
                    qty_bought = 0.0
                    qty_sold = 0.0
                    buy_cost_gross = 0.0
                    sell_proceeds_gross = 0.0
                    fees_total = 0.0
                    realized_pnl_total = 0.0
                    last_tx_date = None

        # Only the currently open cycle belongs in Positions.
        if qty_held > 1e-12:
            avg_entry_all_buys = (buy_cost_gross / qty_bought) if qty_bought > 0 else np.nan
            avg_cost_current = (cost_basis_held / qty_held) if qty_held > 0 else np.nan

            positions_rows.append({
                "project": project,
                "cycle_id": cycle_id,
                "cycle_start_date": cycle_start_date,
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
                "last_tx_date": last_tx_date if last_tx_date is not None else grp["date"].max(),
            })

    positions = pd.DataFrame(positions_rows, columns=position_columns)
    sales = pd.DataFrame(sales_rows, columns=sales_columns)

    if not sales.empty:
        sales = sales.sort_values("date", ascending=False).reset_index(drop=True)

    if not positions.empty:
        positions = positions.sort_values("project").reset_index(drop=True)

    return positions, sales, warnings_list



def montant_investi_affichage(row: pd.Series, transactions: pd.DataFrame) -> float:
    """Montant total investi affiché dans Positions.

    Règle volontairement UX / informative, sans impact sur les calculs :
    - Si le trade ouvert a eu une vente puis un rachat ensuite, on affiche le capital net injecté :
      buy_cost_gross - sell_proceeds_gross.
    - Sinon, on affiche le total des achats du cycle ouvert : buy_cost_gross.

    Exemple :
    - NOCK : achats puis prises de profits, pas de rachat après vente => total BUY.
    - OCT : achat, prise de profit, puis rachat => BUY - SELL.
    """
    project = str(row.get("project", "")).upper().strip()

    if not project or project in cash_assets:
        return np.nan

    tx = transactions[transactions["project"] == project].copy()

    cycle_start_date = row.get("cycle_start_date", None)
    if pd.notna(cycle_start_date):
        tx = tx[tx["date"] >= cycle_start_date]

    tx = tx.sort_values("date").reset_index(drop=True)

    seen_sell = False
    has_buy_after_sell = False

    for _, t in tx.iterrows():
        tx_type = str(t["type"]).upper().strip()
        if tx_type == "SELL":
            seen_sell = True
        elif tx_type == "BUY" and seen_sell:
            has_buy_after_sell = True
            break

    buy_total = float(row.get("buy_cost_gross", 0) or 0)
    sell_total = float(row.get("sell_proceeds_gross", 0) or 0)

    if has_buy_after_sell:
        return buy_total - sell_total

    return buy_total

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Dashboard BW", page_icon="📈", layout="wide")
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

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
                "mise_tokens_restants": np.nan,
                "value_live": amount,
                "pnl_unrealized_$": np.nan,
                "pnl_unrealized_%": np.nan,
                "realized_pnl": np.nan,
                "gain_position_en_cours_$": np.nan,
                "gain_position_en_cours_%": np.nan,
                "profit_global_trade_si_vente_now_$": np.nan,
                "roi_global_trade_si_vente_now_%": np.nan,
            })

cash_positions_df = pd.DataFrame(cash_rows)

if not positions_live.empty:
    # Logique d'affichage retenue pour l'onglet Portefeuille :
    # - Prix achat moyen = moyenne brute de tous les BUY du token.
    # - Gain sur position restante (en cours) = valeur actuelle restante des tokens restants
    #   moins leur base de lecture BUY-only.
    # - Profit global du trade (si vente now) = profit déjà réalisé du cycle ouvert + gain sur position restante (en cours).
    #   Cette colonne évite de croire qu'un token est perdant globalement
    #   quand la position actuelle est rouge mais que des profits ont déjà été encaissés.
    positions_live["mise_tokens_restants"] = positions_live["qty_current"] * positions_live["avg_entry_all_buys"]
    positions_live["gain_position_en_cours_$"] = positions_live["value_live"].fillna(0) - positions_live["mise_tokens_restants"].fillna(0)
    positions_live["gain_position_en_cours_%"] = np.where(
        positions_live["mise_tokens_restants"] > 0,
        (positions_live["gain_position_en_cours_$"] / positions_live["mise_tokens_restants"]) * 100,
        np.nan,
    )
    positions_live["profit_global_trade_si_vente_now_$"] = (
        positions_live["realized_pnl"].fillna(0) + positions_live["gain_position_en_cours_$"].fillna(0)
    )
    positions_live["roi_global_trade_si_vente_now_%"] = np.where(
        positions_live["buy_cost_gross"] > 0,
        (positions_live["profit_global_trade_si_vente_now_$"] / positions_live["buy_cost_gross"]) * 100,
        np.nan,
    )
else:
    positions_live["mise_tokens_restants"] = []
    positions_live["gain_position_en_cours_$"] = []
    positions_live["gain_position_en_cours_%"] = []
    positions_live["profit_global_trade_si_vente_now_$"] = []
    positions_live["roi_global_trade_si_vente_now_%"] = []

profit_open_positions_real = float(np.nansum(positions_live["gain_position_en_cours_$"].to_numpy())) if not positions_live.empty else 0.0
realized_pnl_total = float(sales_df["realized_pnl"].sum()) if not sales_df.empty else 0.0
pnl_total_real = realized_pnl_total + profit_open_positions_real

crypto_current_value = float(np.nansum(positions_live["value_live"].to_numpy())) if not positions_live.empty else 0.0
total_current_value = cash_total + crypto_current_value

pnl_color = "#22c55e" if pnl_total_real > 0 else "#ef4444" if pnl_total_real < 0 else "#e5e7eb"

# ---------------------------
# Top metrics
# ---------------------------
cards = [
    {
        "label": "Profit net total actuel → si on vendait tout now",
        "value": money(pnl_total_real),
        "value_color": pnl_color,
        "detail_html": f"""
            <div style="
                font-size: 10px;
                line-height: 1.45;
                margin-top: 8px;
                color: #e5e7eb;
            ">
                <br><br>
                <span style="font-weight:600; color: rgba(229,231,235,0.90);">
                    {("+" if realized_pnl_total > 0 else "")}{money(realized_pnl_total)}
                </span>
                <span style="color: rgba(229,231,235,0.70);"> déjà encaissés</span>
                <br>
                <span style="font-weight:600; color: rgba(229,231,235,0.90);">
                    {money(profit_open_positions_real)}
                </span>
                <span style="color: rgba(229,231,235,0.70);"> gain sur positions restantes (en cours)</span>
            </div>
        """,
    },
    {
        "label": "Valeur crypto → positions en cours",
        "value": money_rounded(crypto_current_value),
        "value_color": "#e5e7eb",
        "detail_html": "",
    },
    {
        "label": "Cash disponible → rakbank + stablecoins",
        "value": money_rounded(cash_total),
        "value_color": "#e5e7eb",
        "detail_html": """
            <div style="
                font-size: 10px;
                line-height: 1.45;
                margin-top: 8px;
                color: rgba(229,231,235,0.70);
            ">
            </div>
        """,
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
                    opacity: 0.85;
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

st.markdown(
    f"""
<div style="
margin-top:14px;
margin-bottom:12px;
background:rgba(255,255,255,0.03);
border:1px solid rgba(255,255,255,0.06);
border-radius:14px;
padding:12px 18px;
box-sizing:border-box;
">

<div style="
font-size:13px;
color:rgba(229,231,235,0.72);
margin-bottom:8px;
font-weight:500;
">
Total actuel → cash + positions en cours
</div>

<div style="
font-size:28px;
line-height:1.1;
font-weight:700;
color:#ffffff;
">
{money_rounded(total_current_value)}
</div>

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

        df_show["Quantité"] = df_show["qty_current"].map(qty_tokens)
        df_show["Prix achat moyen"] = df_show["avg_entry_all_buys"].map(price)
        df_show["Prix actuel"] = df_show["price_live"].map(price)
        df_show["Montant total investi"] = df_show.apply(
            lambda row: money(montant_investi_affichage(row, transactions)),
            axis=1,
        )
        df_show["Valeur actuelle restante"] = df_show["value_live"].map(money)
        df_show["Gain sur position restante (en cours)"] = df_show["gain_position_en_cours_$"].map(pnl_color_html)
        df_show["Profit global du trade (si vente now)"] = df_show["profit_global_trade_si_vente_now_$"].map(pnl_color_html)
        df_show["ROI global du trade"] = df_show["roi_global_trade_si_vente_now_%"].map(pct_color_html)

        is_cash_row = df_show["project"].isin(list(cash_assets))
        df_show.loc[is_cash_row, ["Prix achat moyen", "Montant total investi", "Gain sur position restante (en cours)", "Profit global du trade (si vente now)", "ROI global du trade"]] = ["—", "—", "—", "—", "—"]
        df_show.loc[is_cash_row, "Valeur actuelle restante"] = df_show.loc[is_cash_row, "value_live"].map(money_rounded)

        cols = [
            "project",
            "Quantité",
            "Prix achat moyen",
            "Prix actuel",
            "Montant total investi",
            "Valeur actuelle restante",
            "Gain sur position restante (en cours)",
            "Profit global du trade (si vente now)",
            "ROI global du trade",
        ]

        positions_html = df_show[cols].rename(columns={"project": "Projet"})
        st.markdown(make_html_table(positions_html), unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown('<div style="height: 2px;"></div>', unsafe_allow_html=True)

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
            st.subheader("📉 Gain sur position restante (en cours)")
            bar_df = positions_live.copy()
            bar_df = bar_df.dropna(subset=["gain_position_en_cours_$"])

            if not bar_df.empty:
                fig2 = px.bar(
                    bar_df,
                    x="project",
                    y="gain_position_en_cours_$",
                    color="project",
                    color_discrete_map=color_map,
                )
                fig2.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis_title="Token",
                    yaxis_title="Gain sur position restante (en cours) ($)",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Gain sur position restante indisponible.")

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

    # Vitesse de gain : calculée depuis le premier BUY jusqu'à la dernière vente.
    if not sales_df.empty:
        buy_dates_for_speed = transactions.loc[transactions["type"] == "BUY", "date"]
        first_buy_date_for_speed = (
            buy_dates_for_speed.min() if not buy_dates_for_speed.empty else sales_df["date"].min()
        )
        last_sale_date_for_speed = sales_df["date"].max()
        days_active = (last_sale_date_for_speed.normalize() - first_buy_date_for_speed.normalize()).days
        days_active = max(int(days_active), 1)
        profit_per_day = realized_pnl_total / days_active
        profit_per_month = profit_per_day * 30
        speed_html = f"""
            <div style="
                margin-top: 8px;
                font-size: 12px;
                color: rgba(229,231,235,0.72);
                line-height: 1.45;
            ">
                en <span style="font-weight:700; color:#e5e7eb;">{days_active} jours</span>
                → ~<span style="font-weight:700; color:#e5e7eb;">{money(profit_per_day)}/jour</span>
                | ~<span style="font-weight:700; color:#e5e7eb;">{money(profit_per_month)}/mois</span>
            </div>
        """
    else:
        speed_html = ""

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
            <div style="font-size: 14px; opacity: 0.85; margin-bottom: 6px;">Profits réalisés cumulés</div>
            <div style="font-size: 24px; font-weight: 700;">{pnl_realized_html}</div>
            {speed_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------
    # Graph — Profit réalisé cumulé
    # ---------------------------
    if not sales_df.empty:
        sales_curve = sales_df.copy().sort_values("date", ascending=True).reset_index(drop=True)

        # Décale légèrement les ventes faites le même jour pour éviter un mur vertical.
        # Exemple : plusieurs ventes NOCK le même jour deviennent un vrai escalier visuel.
        sales_curve["date_chart"] = (
            sales_curve["date"]
            + pd.to_timedelta(sales_curve.groupby("date").cumcount() * 10, unit="m")
        )

        # Profit cumulé réel, vente après vente.
        sales_curve["profit_cumule"] = sales_curve["realized_pnl"].cumsum()

        # Point initial à 0.
        # Important : on le place à la date du premier BUY de ton journal,
        # pas juste avant la première vente.
        # Comme ça le graph raconte vraiment : achat initial → ventes → profits réalisés.
        buy_dates = transactions.loc[transactions["type"] == "BUY", "date"]
        first_buy_date = buy_dates.min() if not buy_dates.empty else sales_curve["date_chart"].min()

        start_row = pd.DataFrame({
            "date": [first_buy_date],
            "date_chart": [first_buy_date],
            "project": ["Départ"],
            "cycle_id": [0],
            "realized_pnl": [0.0],
            "profit_cumule": [0.0],
        })
        sales_curve = pd.concat([start_row, sales_curve], ignore_index=True)
        sales_curve = sales_curve.sort_values("date_chart", ascending=True).reset_index(drop=True)

        sales_curve["Date"] = sales_curve["date_chart"].dt.strftime("%Y-%m-%d")
        sales_curve["Vente"] = sales_curve["realized_pnl"].map(money)
        sales_curve["Profit cumulé"] = sales_curve["profit_cumule"].map(money)
        sales_curve["Token"] = sales_curve["project"].astype(str)
        sales_curve["Cycle"] = sales_curve["cycle_id"].map(lambda x: "" if int(x) == 0 else f"#{int(x)}")

        st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)
        st.subheader("📈 Évolution des profits réalisés")

        fig_realized = go.Figure()
        fig_realized.add_trace(
            go.Scatter(
                x=sales_curve["date_chart"],
                y=sales_curve["profit_cumule"],
                mode="lines+markers",
                line=dict(color="#22c55e", width=3),
                marker=dict(size=8, color="#22c55e"),
                customdata=sales_curve[["Token", "Cycle", "Vente", "Profit cumulé"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
                    "Date : %{x|%Y-%m-%d}<br>"
                    "Vente : %{customdata[2]}<br>"
                    "Profit cumulé : %{customdata[3]}"
                    "<extra></extra>"
                ),
            )
        )
        fig_realized.add_hline(y=0, line_width=1, line_color="rgba(229,231,235,0.25)")
        fig_realized.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis_title="Date",
            yaxis_title="Profit réalisé cumulé ($)",
            font=dict(color="#e5e7eb"),
            hoverlabel=dict(
                bgcolor="#111827",
                bordercolor="rgba(255,255,255,0.12)",
                font_size=13,
            ),
        )
        fig_realized.update_xaxes(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.12)",
        )
        fig_realized.update_yaxes(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.12)",
            tickprefix="$",
            separatethousands=True,
            rangemode="tozero",
        )
        st.plotly_chart(fig_realized, use_container_width=True)

        # ---------------------------
        # Contribution par token sous le graph
        # ---------------------------
        contrib = (
            sales_df.groupby("project", as_index=False)["realized_pnl"]
            .sum()
            .sort_values("realized_pnl", ascending=False)
        )
        contrib = contrib[contrib["realized_pnl"] > 0].copy()

        if not contrib.empty:
            total_positive_pnl = float(contrib["realized_pnl"].sum())
            contrib["contribution_%"] = np.where(
                total_positive_pnl > 0,
                (contrib["realized_pnl"] / total_positive_pnl) * 100,
                0,
            )

            rows_html = ""
            for _, row in contrib.iterrows():
                token = str(row["project"])
                pct_val = float(row["contribution_%"])
                rows_html += f"""
                <div style="
                    display:grid;
                    grid-template-columns: 58px 1fr 52px;
                    align-items:center;
                    gap:10px;
                    margin: 7px 0;
                    max-width: 620px;
                ">
                    <div style="font-size:13px; font-weight:700; color:#e5e7eb;">{token}</div>
                    <div style="height:8px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden;">
                        <div style="height:8px; width:{pct_val:.2f}%; background:#22c55e; border-radius:999px;"></div>
                    </div>
                    <div style="font-size:13px; font-weight:700; color:#e5e7eb; text-align:right;">{pct_val:.0f}%</div>
                </div>
                """

            st.markdown(
                f"""
                <div style="
                    margin-top: 2px;
                    margin-bottom: 22px;
                    padding: 12px 14px;
                    background: rgba(255,255,255,0.025);
                    border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 14px;
                    max-width: 700px;
                ">
                    <div style="
                        font-size: 13px;
                        font-weight: 700;
                        color: rgba(229,231,235,0.82);
                        margin-bottom: 8px;
                    ">
                        Contribution aux profits réalisés
                    </div>
                    {rows_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("""
📌 Note :
Un cycle = un trade complet sur un token.

→ Tu achètes
→ Tu peux vendre en plusieurs fois
→ Quand tu as tout vendu (quantité = 0), le cycle est terminé

→ Si tu rachètes ensuite le même token, un nouveau cycle commence
    """)

    if sales_df.empty:
        st.info("Aucune vente enregistrée.")
    else:
        st.subheader("📊 Synthèse globale par token")

        summary_token = sales_df.groupby("project", as_index=False).agg(
            cycles=("cycle_id", "nunique"),
            quantity_sold=("quantity", "sum"),
            net_proceeds=("net_proceeds", "sum"),
            cost_basis_sold=("cost_basis_sold", "sum"),
            realized_pnl=("realized_pnl", "sum"),
        )

        summary_token = summary_token.sort_values("realized_pnl", ascending=False).reset_index(drop=True)

        summary_token["roi_sur_ventes_%"] = np.where(
            summary_token["cost_basis_sold"] > 0,
            (summary_token["realized_pnl"] / summary_token["cost_basis_sold"]) * 100,
            np.nan,
        )

        summary_token["Cycles"] = summary_token["cycles"].map(lambda x: f"{int(x)}")
        summary_token["Quantité vendue"] = summary_token["quantity_sold"].map(qty_tokens)
        summary_token["Argent récupéré"] = summary_token["net_proceeds"].map(money)
        summary_token["Mise vendue"] = summary_token["cost_basis_sold"].map(money)
        summary_token["Gain / Perte"] = summary_token["realized_pnl"].map(pnl_color_html)
        summary_token["ROI sur ventes"] = summary_token["roi_sur_ventes_%"].map(pct_color_html)

        summary_token_html = summary_token[[
            "project",
            "Cycles",
            "Quantité vendue",
            "Argent récupéré",
            "Mise vendue",
            "Gain / Perte",
            "ROI sur ventes",
        ]].rename(columns={"project": "Token"})

        st.markdown(make_html_table(summary_token_html), unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.subheader("🧩 Synthèse par cycle")

        summary_cycle = sales_df.groupby(["project", "cycle_id"], as_index=False).agg(
            quantity_sold=("quantity", "sum"),
            net_proceeds=("net_proceeds", "sum"),
            cost_basis_sold=("cost_basis_sold", "sum"),
            realized_pnl=("realized_pnl", "sum"),
        )

        summary_cycle = summary_cycle.sort_values("realized_pnl", ascending=False).reset_index(drop=True)

        summary_cycle["roi_sur_ventes_%"] = np.where(
            summary_cycle["cost_basis_sold"] > 0,
            (summary_cycle["realized_pnl"] / summary_cycle["cost_basis_sold"]) * 100,
            np.nan,
        )

        summary_cycle["Cycle"] = summary_cycle["cycle_id"].map(lambda x: f"#{int(x)}")
        summary_cycle["Quantité vendue"] = summary_cycle["quantity_sold"].map(qty_tokens)
        summary_cycle["Argent récupéré"] = summary_cycle["net_proceeds"].map(money)
        summary_cycle["Mise vendue"] = summary_cycle["cost_basis_sold"].map(money)
        summary_cycle["Gain / Perte"] = summary_cycle["realized_pnl"].map(pnl_color_html)
        summary_cycle["ROI sur ventes"] = summary_cycle["roi_sur_ventes_%"].map(pct_color_html)

        summary_cycle_html = summary_cycle[[
            "project",
            "Cycle",
            "Quantité vendue",
            "Argent récupéré",
            "Mise vendue",
            "Gain / Perte",
            "ROI sur ventes",
        ]].rename(columns={"project": "Token"})

        st.markdown(make_html_table(summary_cycle_html), unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.subheader("🧾 Historique des ventes")

        sales_show = sales_df.copy()
        sales_show["Date"] = sales_show["date"].dt.strftime("%Y-%m-%d")
        sales_show["Type"] = sales_show["type"].map(tx_badge_html)
        sales_show["Cycle"] = sales_show["cycle_id"].map(lambda x: f"#{int(x)}")
        sales_show["Quantité vendue"] = sales_show["quantity"].map(qty_tokens)
        sales_show["Prix de vente"] = sales_show["sell_price"].map(price)
        sales_show["Argent récupéré"] = sales_show["net_proceeds"].map(money)
        sales_show["Mise vendue"] = sales_show["cost_basis_sold"].map(money)
        sales_show["Gain / Perte"] = sales_show["realized_pnl"].map(pnl_html)
        sales_show["ROI sur ventes"] = np.where(
            sales_show["cost_basis_sold"] > 0,
            (sales_show["realized_pnl"] / sales_show["cost_basis_sold"]) * 100,
            np.nan,
        )
        sales_show["ROI sur ventes"] = sales_show["ROI sur ventes"].map(pct_color_html)

        sales_html = sales_show[[
            "Date",
            "project",
            "Cycle",
            "Type",
            "Quantité vendue",
            "Prix de vente",
            "Argent récupéré",
            "Mise vendue",
            "Gain / Perte",
            "ROI sur ventes",
            "note",
        ]].rename(columns={
            "project": "Token",
            "note": "Note",
        })

        st.markdown(make_html_table(sales_html), unsafe_allow_html=True)
