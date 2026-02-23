# BW Crypto Dashboard (Streamlit)

Dashboard crypto consolidé avec :
- prix live (CoinGecko)
- positions (DCA)
- PnL / PnL %
- camembert allocation
- objectifs TP (x2 / x4 etc.)

## Lancer en local

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

## Mettre à jour tes données

- `data_trades.csv` : chaque ligne = un achat (DCA)
- `data_targets.csv` : tes objectifs de vente (TP)

## Déployer (simple)

### Streamlit Community Cloud
1. Push ce dossier sur GitHub
2. Sur Streamlit Cloud: New app -> sélectionner le repo -> `app.py` -> Deploy

