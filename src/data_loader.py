"""
data_loader.py — Étape 2 : récupération et nettoyage des données

Ce script fait deux choses, volontairement séparées en deux fonctions :
  1. download_raw()  -> télécharge les données brutes et les sauvegarde telles quelles
  2. clean_data()    -> part des données brutes et produit une série propre, exploitable

Pourquoi séparer les deux ? Parce qu'en pratique, tu ne veux JAMAIS re-télécharger
les données à chaque fois que tu modifies ta logique de nettoyage : le téléchargement
brut est lent et dépend d'un service externe (Yahoo Finance via yfinance), le nettoyage
est rapide et doit pouvoir être rejoué autant de fois que tu changes d'avis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# --- Configuration ---------------------------------------------------------

# Univers de démarrage volontairement réduit et simple : 5 grosses valeurs liquides
# + un ETF actions large (SPY) qui servira de référence ("benchmark") plus tard.
# On commence petit : mieux vaut 5 actifs bien compris que 50 actifs mal vérifiés.
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "PG", "SPY"]

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")


def download_raw() -> None:
    """
    Télécharge les données OHLCV brutes pour chaque ticker et les sauvegarde
    en CSV, un fichier par ticker, sans aucune transformation.

    On sauvegarde le brut séparément du propre pour une raison simple :
    si tu changes ta méthode de nettoyage dans 3 mois, tu dois pouvoir repartir
    du brut sans re-télécharger — Yahoo Finance peut changer, être indisponible,
    ou limiter le nombre de requêtes.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"Téléchargement de {ticker}...")
        # auto_adjust=False : on garde Close ET Adj Close séparément.
        # C'est volontaire : on veut voir la différence entre les deux
        # pour comprendre l'effet des dividendes/splits, pas juste la subir.
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            print(f"  -> ATTENTION : aucune donnée reçue pour {ticker}. "
                  f"Vérifie le ticker ou ta connexion.")
            continue

        # Les versions récentes de yfinance renvoient des colonnes à DEUX niveaux
        # (ex. "Adj Close" / "AAPL"), même pour un seul ticker. Si on sauvegarde ça
        # tel quel en CSV, on obtient deux lignes d'en-tête au lieu d'une, et la
        # relecture confond la 2e ligne d'en-tête avec une ligne de données.
        # On aplatit donc les colonnes en un seul niveau avant de sauvegarder.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.to_csv(RAW_DIR / f"{ticker}.csv")
        print(f"  -> {len(df)} lignes sauvegardées dans {RAW_DIR / f'{ticker}.csv'}")


def clean_data() -> pd.DataFrame:
    """
    Part des CSV bruts et produit UNE seule table propre :
    lignes = dates, colonnes = tickers, valeurs = rendements log journaliers
    calculés sur le Adj Close.

    Retourne ce DataFrame et le sauvegarde aussi dans data/clean/returns.csv.
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    adj_close_by_ticker = {}

    for ticker in TICKERS:
        raw_path = RAW_DIR / f"{ticker}.csv"
        if not raw_path.exists():
            print(f"Pas de fichier brut pour {ticker}, lance download_raw() d'abord.")
            continue

        # index_col=0 + parse_dates : la colonne des dates devient l'index du DataFrame,
        # ce qui permet d'aligner facilement plusieurs actifs sur les mêmes dates ensuite.
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

        # On ne garde QUE le Adj Close : c'est la colonne qui neutralise
        # dividendes et splits (voir l'explication donnée à côté de ce code).
        adj_close_by_ticker[ticker] = df["Adj Close"]

    # On assemble tous les actifs dans une seule table (une colonne par ticker).
    # pandas aligne automatiquement sur l'index (les dates) ; si un actif n'a pas
    # coté un jour donné (jour férié local, IPO plus tardive...), la case sera NaN.
    prices = pd.DataFrame(adj_close_by_ticker)

    # Gestion des valeurs manquantes : on comble par la dernière valeur connue
    # (forward-fill), mais seulement sur 3 jours max. Au-delà, on considère
    # que l'absence de donnée est un vrai problème à ne pas masquer artificiellement.
    prices = prices.ffill(limit=3)

    # On calcule les rendements log journaliers : log(prix_t / prix_t-1).
    # np.log(prices / prices.shift(1)) est l'écriture vectorisée pandas/numpy —
    # shift(1) décale toute la série d'un jour vers le bas, donc prices.shift(1)
    # à la ligne du 5 mars contient le prix du 4 mars.
    log_returns = np.log(prices / prices.shift(1))

    # La toute première ligne est forcément NaN (pas de "jour précédent" pour le
    # tout premier jour de la série) : on la retire.
    log_returns = log_returns.dropna(how="all")

    log_returns.to_csv(CLEAN_DIR / "returns.csv")
    print(f"Rendements propres sauvegardés : {CLEAN_DIR / 'returns.csv'} "
          f"({log_returns.shape[0]} lignes, {log_returns.shape[1]} actifs)")

    return log_returns


if __name__ == "__main__":
    download_raw()
    returns = clean_data()
    print(returns.head())
    print(returns.describe())
