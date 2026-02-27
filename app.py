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
TRADES_FILE = "data_trades.csv"
TARGETS_FILE = "data_targets.csv"        # multiples (x2/x4) based on PRU
CASH_FILE = "data_cash.csv"              # stables: USDC/USDT/DAI
EXEC_FILE = "data_execution.csv"         # persistent execution journal (edit in GitHub)
HARDWARE_FILE = "data_hardware.csv"      # hardware liquidity

DEFAULT_VS_CURRENCY = "usd"

# We keep CoinGecko mapping as optional fallback, but TAO will use Binance.
COINGECKO_ID_BY_PROJECT = {
    "TAO": "bittensor",
    "NOCK": "nockchain",
}

# TAO live price source: Binance spot ticker (USD via USDT)
BINANCE_SYMBOL_BY_PROJECT = {
    "TAO": "TAOUSDT",
}

# NOCK live price source: Dexscreener pair (Aerodrome Base NOCK/USDC)
DEXSCREENER_PAIR_BY_PROJECT = {
    "NOCK": {
        "chain": "base",
        "pair": "0x85f1aa3a70fedd1c52705c15baed143e675cd626",
    }
}

FALLBACK_PRICE_BY_PROJECT: Dict[str, float] = {}  # optional manual fallback


# ---------------------------
# TAO staking (TaoStats)
# ---------------------------
# ✅ Mets ta clé ici (ou mieux: st.secrets["TAOSTATS_API_KEY"])
TAOSTATS_API_KEY = "tao-07f6bab3-e23d-4285-b9d5-a094cdb8e392:7eba670d"
TAO_COLDKEY = "5Fjkt5yxYyBNbRAWpVtwaj4RBG3txjAJEv7CU7Yx3Uomyw7X"

TAO_PROJECT = "TAO"
TAO_SUBNET_NAME = "root"
TAO_SUBNET_NETUID = 0

RAO_PER_TAO = 1e9


# ---------------------------
# Styles (premium)
# ---------------------------
PREMIUM_CSS = """
<style>
h1, h2, h3 { letter-spacing: -0.02em; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* some versions wrap the icon in a button */
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
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
}
.muted { opacity: 0.75; }
.ok { color: #22c55e; }
.warn { color: #f59e0b; }
.bad { color: #ef4444; }

/* Hide Streamlit heading anchor (copy link icon) */
/* Hide Streamlit heading anchor / copy-link icon (robust) */
a.stMarkdownAnchor,
a[data-testid="stMarkdownAnchor"],
.stMarkdown a[href^="#"],
h1 a[href^="#"], h2 a[href^="#"], h3 a[href^="#"] {
  display: none !important;
}
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
    return f"{x:,.2f}"


def qty_tokens_tao_precise(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    return f"{float(x):,.6f}"


def pct(x: Optional[float]) -> str:
    if not is_number(x):
        return "—"
    return f"{float(x):,.2f}%"


def progress(current: Optional[float], target: float) -> float:
    if not is_number(current) or target <= 0:
        return 0.0
    return float(np.clip(float(current) / target, 0.0, 1.0))


def parse_taostats_amount_to_tao(x) -> Optional[float]:
    """
    TaoStats peut renvoyer des montants en RAO (1e9) même si le champ s'appelle *_as_tao.
    On normalise ici en TAO.

    Heuristique robuste :
    - si c'est très grand, c'est du RAO => /1e9
    - sinon on prend tel quel
    """
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        try:
            v = float(str(x).strip())
        except Exception:
            return None

    # 165.9 TAO => OK ; 165946061804 => RAO déguisé
    if v >= 1_000_000:
        v = v / RAO_PER_TAO
    return v


# ---------------------------
# TaoStats API fetchers
# ---------------------------
@st.cache_data(ttl=60, show_spinner=False)
def taostats_get_stake_balance_latest(coldkey_ss58: str, api_key: str) -> Optional[dict]:
    if not api_key or api_key.startswith("YOUR_"):
        return None
    url = "https://api.taostats.io/api/dtao/stake_balance/latest/v1"
    headers = {"accept": "application/json", "authorization": api_key}
    r = requests.get(url, headers=headers, params={"coldkey": coldkey_ss58}, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    rows = data.get("data") or []
    return rows[0] if rows else None


@st.cache_data(ttl=60, show_spinner=False)
def taostats_get_validator_yield_latest(api_key: str, page: int = 1, per_page: int = 200) -> Optional[dict]:
    if not api_key or api_key.startswith("YOUR_"):
        return None
    url = "https://api.taostats.io/api/dtao/validator/yield/latest/v1"
    headers = {"accept": "application/json", "authorization": api_key}
    r = requests.get(url, headers=headers, params={"page": page, "per_page": per_page}, timeout=25)
    if r.status_code != 200:
        return None
    return r.json()


def find_root_apy_for_hotkey(api_key: str, hotkey_ss58: str, max_pages: int = 109) -> Optional[float]:
    """
    Cherche la ligne du hotkey sur netuid=0 (root) et retourne thirty_day_apy (≈ ce que tu vois sur le dash).
    On stop dès qu'on trouve.
    """
    if not api_key or api_key.startswith("YOUR_"):
        return None

    for p in range(1, max_pages + 1):
        payload = taostats_get_validator_yield_latest(api_key, page=p, per_page=200)
        if not payload:
            return None

        # payload d'erreur type {"status_code":429,...}
        if isinstance(payload, dict) and "status_code" in payload:
            return None

        for row in (payload.get("data") or []):
            if row.get("hotkey", {}).get("ss58") == hotkey_ss58 and int(row.get("netuid", -1)) == TAO_SUBNET_NETUID:
                v = row.get("thirty_day_apy")
                return float(v) if v is not None else None

        time.sleep(0.6)  # évite 429
    return None


# ---------------------------
# Price fetchers
# ---------------------------
@st.cache_data(ttl=20, show_spinner=False)
def fetch_binance_price(symbol: str) -> Optional[float]:
    """
    Fetch last price from Binance public API (spot) with robust fallbacks.
    Example: symbol='TAOUSDT'
    """
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
    """
    Optional fallback (not used for TAO/NOCK in normal operation).
    Return only pickle-serializable types for st.cache_data.
    Returns: (prices_by_id, source, as_of_epoch)
    """
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
    """
    Fetch priceUsd from Dexscreener for a specific pair.
    Used for NOCK (Aerodrome Base).
    """
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


# ---------------------------
# Data loaders
# ---------------------------
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
    Expected:
    project,stage,multiple,sell_pct,note
    """
    df = pd.read_csv(path)
    df["project"] = df["project"].astype(str).str.upper().str.strip()
    df["stage"] = df["stage"].astype(str).str.strip()
    df["multiple"] = pd.to_numeric(df["multiple"], errors="coerce")
    df["sell_pct"] = pd.to_numeric(df["sell_pct"], errors="coerce")
    df["note"] = df.get("note", "").astype(str)
    df = df.dropna(subset=["project", "stage", "multiple", "sell_pct"])
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


def load_execution(path: str) -> pd.DataFrame:
    """
    project,stage,executed,sell_price,executed_at
    """
    try:
        df = pd.read_csv(path)
        df["project"] = df["project"].astype(str).str.upper().str.strip()
        df["stage"] = df["stage"].astype(str).str.strip()
        df["executed"] = df["executed"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df["sell_price"] = pd.to_numeric(df.get("sell_price", np.nan), errors="coerce")
        df["executed_at"] = df.get("executed_at", "").astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["project", "stage", "executed", "sell_price", "executed_at"])


def load_hardware(path: str) -> pd.DataFrame:
    """
    Expected:
    item,unit_price_usd,qty_total,qty_pending_payment
    """
    try:
        df = pd.read_csv(path)
        df["item"] = df["item"].astype(str).str.strip()
        df["unit_price_usd"] = pd.to_numeric(df["unit_price_usd"], errors="coerce")
        df["qty_total"] = pd.to_numeric(df["qty_total"], errors="coerce")
        if "qty_pending_payment" not in df.columns:
            df["qty_pending_payment"] = 0
        df["qty_pending_payment"] = pd.to_numeric(df["qty_pending_payment"], errors="coerce").fillna(0)
        df = df.dropna(subset=["item", "unit_price_usd", "qty_total"])
        df["qty_total"] = df["qty_total"].astype(int)
        df["qty_pending_payment"] = df["qty_pending_payment"].astype(int)
        df["qty_available"] = (df["qty_total"] - df["qty_pending_payment"]).clip(lower=0)
        df["value_total_usd"] = df["unit_price_usd"] * df["qty_total"]
        df["value_available_usd"] = df["unit_price_usd"] * df["qty_available"]
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "item", "unit_price_usd", "qty_total", "qty_pending_payment",
            "qty_available", "value_total_usd", "value_available_usd"
        ])


# ---------------------------
# Portfolio computations
# ---------------------------
def consolidate_positions(trades: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ On garde 2 colonnes internes :
    - tokens_principal = somme des achats DCA (sert au PRU)
    - tokens_total = tokens à valoriser (par défaut = principal, override TAO via wallet)
    """
    g = trades.groupby("project", as_index=False).agg(
        tokens_principal=("tokens", "sum"),
        invested_total=("amount_invested", "sum"),
    )
    g["tokens_total"] = g["tokens_principal"]
    g["avg_entry"] = np.where(
        g["tokens_principal"] > 0,
        g["invested_total"] / g["tokens_principal"],
        np.nan
    )
    return g


def attach_live_prices(pos: pd.DataFrame, vs_currency: str) -> Tuple[pd.DataFrame, str]:
    """
    Live pricing logic:
    - NOCK: Dexscreener (USD only)
    - TAO: Binance (USD only via TAOUSDT)
    - Others: optional CoinGecko fallback
    Also includes "last known good price" fallback via st.session_state.
    """
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

    # ✅ valeur basée sur tokens_total (wallet pour TAO, DCA pour les autres)
    out["value_live"] = out["tokens_total"] * out["price_live"]
    out["pnl_$"] = out["value_live"] - out["invested_total"]
    out["pnl_%"] = np.where(
        out["invested_total"] > 0,
        (out["pnl_$"] / out["invested_total"]) * 100,
        np.nan,
    )
    return out, "live"


def get_target_price(pru: Optional[float], multiple: float) -> Optional[float]:
    if not is_number(pru) or not is_number(multiple):
        return None
    return float(pru) * float(multiple)


def execution_row(exe: pd.DataFrame, project: str, stage: str) -> pd.Series:
    match = exe[(exe["project"] == project) & (exe["stage"] == stage)]
    if match.empty:
        return pd.Series({"executed": False, "sell_price": np.nan, "executed_at": ""})
    return match.iloc[0]


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
        st.info("Note: NOCK/TAO sont pricés en USD (Dexscreener/Binance). En EUR, certains prix peuvent être indisponibles.")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=True)
    manual_refresh = st.button("🔄 Rafraîchir maintenant")
    st.divider()
    show_trades = st.toggle("Voir le détail des achats (DCA)", value=True)
    st.caption("Modifie data_trades.csv / data_targets.csv / data_cash.csv / data_execution.csv / data_hardware.csv")

if auto_refresh:
    st_autorefresh(interval=60_000, key="autorefresh_60s")

if manual_refresh:
    st.cache_data.clear()
    st.rerun()

trades = load_trades(TRADES_FILE)
targets = load_targets(TARGETS_FILE)
cash_df = load_cash(CASH_FILE)
exe_df = load_execution(EXEC_FILE)
hw_df = load_hardware(HARDWARE_FILE)

positions = consolidate_positions(trades)

# ---------------------------
# ✅ TAO wallet override (tokens_total) + staking info
# ---------------------------
tao_wallet_tokens: Optional[float] = None
tao_hotkey: Optional[str] = None
tao_root_apy: Optional[float] = None

stake_row = taostats_get_stake_balance_latest(TAO_COLDKEY, TAOSTATS_API_KEY)
if stake_row:
    tao_hotkey = (stake_row.get("hotkey") or {}).get("ss58")
    # balance_as_tao est parfois en RAO -> on normalise
    tao_wallet_tokens = parse_taostats_amount_to_tao(stake_row.get("balance_as_tao", None))
    if tao_wallet_tokens is None:
        tao_wallet_tokens = parse_taostats_amount_to_tao(stake_row.get("balance", None))

    # override tokens_total pour TAO (mais on garde tokens_principal pour PRU)
    if tao_wallet_tokens is not None and TAO_PROJECT in positions["project"].values:
        positions.loc[positions["project"] == TAO_PROJECT, "tokens_total"] = float(tao_wallet_tokens)

# ---------------------------
# Live prices + portfolio totals
# ---------------------------
positions_live, _price_source_hidden = attach_live_prices(positions, vs_currency)

stable_assets = {"USDC", "USDT", "DAI"}
cash_total = 0.0
if not cash_df.empty:
    for asset in cash_df["asset"].unique():
        amt = float(cash_df.loc[cash_df["asset"] == asset, "amount"].sum())
        if asset in stable_assets:
            cash_total += amt

invested_total = float(positions_live["invested_total"].sum())
value_positions_live = float(np.nansum(positions_live["value_live"].to_numpy()))
value_total_live = value_positions_live + cash_total
pnl_positions = value_positions_live - invested_total
pnl_positions_pct = (pnl_positions / invested_total * 100) if invested_total > 0 else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Investi", money(invested_total))
k2.metric("Cash (stablecoins)", money(cash_total))
k3.metric("Valeur totale (live)", money(value_total_live))
k4.metric("PnL (positions)", money(pnl_positions))
k5.metric("PnL % (positions)", pct(pnl_positions_pct))

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

tab_portefeuille, tab_plan, tab_exec, tab_hw = st.tabs(
    ["📊 Portefeuille", "🎯 Plan de vente", "✅ Ventes réalisées", "🖥️ Matériel"]
)

cash_label = "USDC (cash)"
cash_row = pd.DataFrame([{
    "project": cash_label,
    "tokens_principal": cash_total,
    "tokens_total": cash_total,
    "invested_total": 0.0,
    "avg_entry": np.nan,
    "price_live": 1.0,
    "value_live": cash_total,
    "pnl_$": np.nan,
    "pnl_%": np.nan,
}])
positions_all = pd.concat([positions_live, cash_row], ignore_index=True)

pru_by_proj = dict(zip(positions_live["project"], positions_live["avg_entry"]))
cur_price_by_proj = dict(zip(positions_live["project"], positions_live["price_live"]))
inv_by_proj = dict(zip(positions_live["project"], positions_live["invested_total"]))
tok_by_proj = dict(zip(positions_live["project"], positions_live["tokens_total"]))

all_labels_for_colors = positions_all["project"].astype(str).tolist()
palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(all_labels_for_colors)}


# ---------------------------
# TAB 1 — Portefeuille
# ---------------------------
with tab_portefeuille:
    st.subheader("📌 Positions")

    df_show = positions_all.copy()

    # ✅ colonne Tokens (visuel inchangé) — mais TAO en format précis
    def fmt_tokens_row(r: pd.Series) -> str:
        if r["project"] == cash_label:
            return money(r["tokens_total"])
        if r["project"] == TAO_PROJECT:
            return qty_tokens_tao_precise(r["tokens_total"])
        return qty_tokens(r["tokens_total"])

    df_show["Tokens"] = df_show.apply(fmt_tokens_row, axis=1)

    df_show["PRU (DCA)"] = df_show["avg_entry"].map(price)
    df_show.loc[df_show["project"] == cash_label, "PRU (DCA)"] = "—"
    df_show["Prix live"] = df_show["price_live"].map(price)

    df_show["Investi"] = df_show["invested_total"].map(money)
    df_show.loc[df_show["project"] == cash_label, "Investi"] = "—"

    df_show["Valeur"] = df_show["value_live"].map(money)

    df_show["PnL"] = df_show["pnl_$"].map(money)
    df_show["PnL %"] = df_show["pnl_%"].map(pct)
    df_show.loc[df_show["project"] == cash_label, ["PnL", "PnL %"]] = ["—", "—"]

    cols = ["project", "Tokens", "PRU (DCA)", "Prix live", "Investi", "Valeur", "PnL", "PnL %"]

    st.dataframe(
        df_show[cols].rename(columns={"project": "Projet"}),
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------
    # ✅ Staking section (TAO only for now)
    # ---------------------------
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("🪵 Staking")

    tao_row_live = positions_live[positions_live["project"] == TAO_PROJECT]
    staking_table = pd.DataFrame(columns=["Projet", "Subnet", "APY", "Staking rewards"])

    if not tao_row_live.empty and stake_row:
        hotkey = tao_hotkey
        if hotkey:
            tao_root_apy = find_root_apy_for_hotkey(TAOSTATS_API_KEY, hotkey)

        tokens_total_wallet = float(tao_row_live.iloc[0]["tokens_total"])
        tokens_principal = float(tao_row_live.iloc[0]["tokens_principal"])
        rewards_tao = max(tokens_total_wallet - tokens_principal, 0.0)

        tao_price_live = cur_price_by_proj.get(TAO_PROJECT)
        rewards_usd = (rewards_tao * float(tao_price_live)) if is_number(tao_price_live) else np.nan

        apy_str = pct((tao_root_apy * 100.0) if tao_root_apy is not None else None)
        rewards_str = f"{qty_tokens_tao_precise(rewards_tao)} TAO ({money(rewards_usd)})"

        staking_table = pd.DataFrame([{
            "Projet": "TAO",
            "Subnet": TAO_SUBNET_NAME,
            "APY": apy_str,
            "Staking rewards": rewards_str,
        }])
    else:
        staking_table = pd.DataFrame([{
            "Projet": "TAO",
            "Subnet": TAO_SUBNET_NAME,
            "APY": "—",
            "Staking rewards": "—",
        }])

    st.dataframe(staking_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📊 Répartition")
        pie_df = positions_all.dropna(subset=["value_live"]).copy()
        if pie_df.empty or (len(pie_df) == 1 and pie_df.iloc[0]["project"] == cash_label and len(positions_live) > 0):
            st.warning("Prix manquants pour certaines positions : la répartition peut être incomplète.")
        else:
            fig = px.pie(
                pie_df, names="project", values="value_live", hole=0.45,
                color="project", color_discrete_map=color_map
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 PnL par token")
        bar_df = positions_live.dropna(subset=["pnl_$"]).copy()
        if not bar_df.empty:
            fig2 = px.bar(
                bar_df, x="project", y="pnl_$",
                color="project", color_discrete_map=color_map
            )
            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
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


# ---------------------------
# TAB 2 — Plan de vente (théorique)
# ---------------------------
with tab_plan:
    st.subheader("🎯 Plan de vente")
    st.caption("Les cibles (x2/x4) se recalculent automatiquement quand tu ajoutes un buy (PRU actuel).")

    for proj in positions_live["project"].tolist():
        st.markdown(f"### {proj}")

        cur = cur_price_by_proj.get(proj)
        invested_proj = float(inv_by_proj.get(proj, 0.0))
        tokens_total_proj = float(tok_by_proj.get(proj, 0.0))
        pru = pru_by_proj.get(proj)

        t = targets[targets["project"] == proj].copy().sort_values("stage")
        if t.empty:
            st.info("Pas de plan configuré.")
            continue

        remaining = tokens_total_proj
        cumulative_cash = 0.0

        for _, row in t.iterrows():
            stage = str(row["stage"])
            multiple = float(row["multiple"])
            sell_pct = float(row["sell_pct"])
            note = str(row.get("note", "")).strip()

            tgt = get_target_price(pru, multiple)
            if not is_number(tgt):
                st.warning("PRU manquant : impossible de calculer la cible.")
                continue

            sold_tokens = remaining * sell_pct
            remaining_after = remaining - sold_tokens

            cash_if_hit = sold_tokens * float(tgt)
            cumulative_cash += cash_if_hit

            st.write(f"**{stage}** • **x{int(multiple)}** • cible: **{price(tgt)}** • vente: **{int(sell_pct*100)}% du restant**")
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
                    st.metric("À la cible, récup. mise ?", "Oui" if float(tgt) >= be_price else "Non")
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


# ---------------------------
# TAB 3 — Ventes réalisées (tracker persistant) — VERSION "NET"
# ---------------------------
with tab_exec:
    st.subheader("✅  Ventes réalisées")
    st.caption("Persistant via data_execution.csv (tu modifies executed/sell_price dans GitHub après une vente).")

    total_cash_realized = 0.0
    total_net_realized = 0.0

    for proj in positions_live["project"].tolist():
        st.markdown(f"### {proj}")

        tokens_total_proj = float(tok_by_proj.get(proj, 0.0))
        invested_proj = float(inv_by_proj.get(proj, 0.0))
        cur_live = cur_price_by_proj.get(proj)

        t = targets[targets["project"] == proj].copy().sort_values("stage")
        if t.empty:
            st.info("Pas de plan configuré.")
            continue

        remaining = tokens_total_proj
        cash_realized = 0.0

        for _, row in t.iterrows():
            stage = str(row["stage"])
            sell_pct = float(row["sell_pct"])

            ex = execution_row(exe_df, proj, stage)
            executed = bool(ex.get("executed", False))
            sell_price = ex.get("sell_price", np.nan)
            executed_at = str(ex.get("executed_at", ""))

            stage_tokens_to_sell = remaining * sell_pct

            c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 1.2])
            with c1:
                st.metric("Étape", stage)
            with c2:
                st.metric("Exécutée", "Oui" if executed else "Non")
            with c3:
                st.metric("Prix vente réel", price(sell_price) if is_number(sell_price) else "—")
            with c4:
                st.metric("Tokens concernés", qty_tokens(stage_tokens_to_sell))

            if executed and is_number(sell_price):
                stage_cash = float(stage_tokens_to_sell) * float(sell_price)
                cash_realized += stage_cash
                remaining = remaining - stage_tokens_to_sell

                dt = f" • {executed_at}" if executed_at and executed_at != "nan" else ""
                st.markdown(
                    f"<span class='badge'><span class='ok'>✔ Étape exécutée</span>{dt} • Cash encaissé: {money(stage_cash)}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"<span class='badge'><span class='muted'>En attente</span></span>", unsafe_allow_html=True)

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        net_realized = cash_realized - invested_proj
        is_mise_recup = invested_proj > 0 and cash_realized >= invested_proj

        bag_value_live = None
        if is_number(cur_live):
            bag_value_live = float(remaining) * float(cur_live)

        total_cash_realized += cash_realized
        total_net_realized += net_realized

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Cash encaissé (réel)", money(cash_realized))
        with s2:
            st.metric("Bénéfice net réalisé", money(net_realized))
        with s3:
            st.metric("Mise récupérée ?", "Oui" if is_mise_recup else "Non")
        with s4:
            st.metric("Valeur bag restant (live)", money(bag_value_live) if is_number(bag_value_live) else "—")

        st.caption(f"Tokens restants (réel) : {qty_tokens(remaining)}")
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.subheader("Résumé global (réalisé)")
    g1, g2 = st.columns(2)
    g1.metric("Cash total encaissé", money(total_cash_realized))
    g2.metric("Bénéfice net réalisé total", money(total_net_realized))

    st.info(
        "📝 Pour mettre à jour : ouvre data_execution.csv sur GitHub et passe 'executed' à true + renseigne 'sell_price' "
        "(et optionnellement 'executed_at')."
    )


# ---------------------------
# TAB 4 — Matériel (mini-portfolio) — no table when single item
# ---------------------------
with tab_hw:
    st.subheader("🖥️  Matériel")

    if hw_df.empty:
        st.warning("Aucune donnée matériel. Crée data_hardware.csv (voir modèle).")
    else:
        total_qty = int(hw_df["qty_total"].sum())
        pending_qty = int(hw_df["qty_pending_payment"].sum())
        available_qty = int(hw_df["qty_available"].sum())

        total_value = float((hw_df["unit_price_usd"] * hw_df["qty_total"]).sum())
        pending_value = float((hw_df["unit_price_usd"] * hw_df["qty_pending_payment"]).sum())
        available_value = float((hw_df["unit_price_usd"] * hw_df["qty_available"]).sum())

        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px;
                padding: 18px 18px 14px 18px;
                margin-bottom: 14px;
            ">
              <div style="font-size: 13px; opacity: 0.75; margin-bottom: 6px;">Valeur matériel totale</div>
              <div style="font-size: 34px; font-weight: 700; letter-spacing: -0.02em;">{money(total_value)}</div>
              <div style="margin-top: 8px; font-size: 12px; opacity: 0.70;">
                {total_qty:,} GPU(s) au total • {available_qty:,} dispo • {pending_qty:,} en attente paiement
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        avg_unit = float(total_value / total_qty) if total_qty > 0 else np.nan

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total GPUs", f"{total_qty:,}")
            st.caption(f"Valeur : {money(total_value)}")
        with c2:
            st.metric("Prix unitaire", price(avg_unit))
            st.caption("Moyenne pondérée")
        with c3:
            st.metric("En attente paiement", f"{pending_qty:,}")
            st.caption(f"Montant en attente : {money(pending_value)}")
        with c4:
            st.metric("Dispo (vendable)", f"{available_qty:,}")
            st.caption(f"Valeur vendable : {money(available_value)}")

        if hw_df.shape[0] > 1:
            show = hw_df.copy()
            show["Prix unitaire"] = show["unit_price_usd"].map(price)
            show["Valeur totale"] = show["value_total_usd"].map(money)
            show["Valeur dispo"] = show["value_available_usd"].map(money)

            st.dataframe(
                show[["item", "Prix unitaire", "qty_total", "qty_pending_payment", "qty_available", "Valeur totale", "Valeur dispo"]]
                    .rename(columns={
                        "item": "Matériel",
                        "qty_total": "Qté totale",
                        "qty_pending_payment": "En attente",
                        "qty_available": "Dispo",
                    }),
                use_container_width=True,
                hide_index=True,
            )

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.subheader("📊 Répartition valeur (total)")
                fig_hw = px.pie(hw_df, names="item", values="value_total_usd", hole=0.45)
                fig_hw.update_traces(textposition="inside", textinfo="percent+label")
                fig_hw.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig_hw, use_container_width=True)
            with c2:
                st.subheader("📦 Dispo vs attente")
                tmp2 = pd.DataFrame({
                    "item": hw_df["item"],
                    "Dispo": hw_df["qty_available"],
                    "En attente": hw_df["qty_pending_payment"],
                }).melt(id_vars=["item"], var_name="statut", value_name="qty")
                fig_hw2 = px.bar(tmp2, x="item", y="qty", color="statut", barmode="group")
                fig_hw2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_hw2, use_container_width=True)
        else:
            pass
