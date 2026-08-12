"""Configuration partagée pour la récupération de prix.

Utilisé à la fois par app.py (le dashboard Streamlit) et par
alert_watchlist.py (le script d'alerte Telegram lancé via GitHub Actions).
Centraliser ces mappings ici évite qu'ils dérivent entre les deux fichiers :
un token ajouté ici est immédiatement disponible partout.
"""

from typing import Dict

COINGECKO_ID_BY_PROJECT: Dict[str, str] = {
    "TAO": "bittensor",
    "NOCK": "nockchain",
}

BINANCE_SYMBOL_BY_PROJECT: Dict[str, str] = {
    "TAO": "TAOUSDT",
    "ZEC": "ZECUSDT",
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

OKX_INST_ID_BY_PROJECT: Dict[str, str] = {
    "HYPE": "HYPE-USDT",
}

SAFETRADE_MARKET_BY_PROJECT: Dict[str, str] = {
    "PRL": "prlusdt",
}

DEXSCREENER_PAIR_BY_PROJECT: Dict[str, Dict[str, str]] = {
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
    "TIG": {
        "chain": "base",
        "pair": "0x3f5e98c7ebff35056ab4346bccd722a537c1aefa",
    },
    "COP": {
        "chain": "base",
        "pair": "0xa51b3a0f976c3fe1054ccaa42cc3b807416f02f0db6724b2c72e99c72e572c24",
    },
    "TSG": {
        "chain": "base",
        "pair": "0x5e4c78bf666d78fa1e751abc84cf9933d17b1736d4605f400173ac63ac52b1f8",
    },
    "RAIL": {
        "chain": "ethereum",
        "pair": "0xac86903cdda380f20a06cc8a2dea7749f1558c68",
    },
    "FWA": {
        "chain": "ethereum",
        "pair": "0x230ecd3c25b44af30db59c15f70df7794eb13f67a200f230b7400daa96fe804d",
    },
    "PONS": {
        "chain": "robinhood",
        "pair": "0x10cc6bd38112cac182db90b6a71d8bb5939526ba",
    },
    "STONKBROKER": {
        "chain": "robinhood",
        "pair": "0xd33c8fd38b06e989cdbd4dffdefab71c4bdd415b24964c8d69e38ff35b068f92",
    },
    "PUMP": {
        "chain": "solana",
        "pair": "2uf4xh61rdwxng9woyxsvqp7zua6klfpb3nvnrqeoisd",
    },
}

DEXSCREENER_URL_BY_PROJECT: Dict[str, str] = {
    "PONS": "https://dexscreener.com/robinhood/0x10cc6bd38112cac182db90b6a71d8bb5939526ba",
    "STONKBROKER": "https://dexscreener.com/robinhood/0xd33c8fd38b06e989cdbd4dffdefab71c4bdd415b24964c8d69e38ff35b068f92",
    "FWA": "https://dexscreener.com/ethereum/0x230ecd3c25b44af30db59c15f70df7794eb13f67a200f230b7400daa96fe804d",
    "PUMP": "https://dexscreener.com/solana/2uf4xh61rdwxng9woyxsvqp7zua6klfpb3nvnrqeoisd",
}

FALLBACK_PRICE_BY_PROJECT: Dict[str, float] = {}
