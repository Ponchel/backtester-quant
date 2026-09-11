"""
engine.py — Étape 3 : moteur de backtest

Ce moteur ne connaît AUCUNE stratégie précise. Il prend en entrée un tableau
de "signaux" (ce qu'une stratégie voudrait faire) et transforme ça en
performance réaliste, en gérant deux choses que les stratégies elles-mêmes
ne devraient jamais avoir à gérer :
  1. Le décalage temporel qui empêche le look-ahead bias
  2. Les coûts de transaction

Convention IMPORTANTE, à ne jamais casser dans tout le reste du projet :
  - Un "signal" à la date t est calculé avec les rendements connus JUSQU'À
    la date t incluse (donc via une fonction qui ne regarde jamais "après").
  - La "position" réellement tenue à la date t est le signal de la date t-1
    (on décale d'un jour AVANT d'appliquer aux rendements). C'est ce décalage
    qui garantit qu'on ne peut jamais profiter d'une information qu'on
    n'avait pas encore au moment de décider.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def shift_signals_to_positions(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme des signaux (décision prise à t, basée sur l'info jusqu'à t)
    en positions réellement tenues (appliquées au rendement du jour t).

    signals.shift(1) : décale toute la série d'une ligne vers le bas.
    Concrètement, la position tenue le 5 mars devient le signal calculé le 4 mars.
    Sans ce décalage, on appliquerait le signal du 5 mars au rendement du 5 mars
    lui-même — c'est exactement le look-ahead bias qu'on veut éviter.

    fillna(0) : le tout premier jour n'a pas de signal précédent (pas de "t-1"
    avant le début de la série) donc pas de position possible -> on force à 0
    (aucune position tenue) plutôt que de laisser un NaN qui casserait les calculs.
    """
    return signals.shift(1).fillna(0)


def compute_transaction_costs(positions: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """
    Calcule le coût, en rendement négatif, de chaque changement de position.

    positions.diff() : la différence entre la position d'aujourd'hui et celle
    d'hier. Si tu étais à position 0 (rien) et que tu passes à 1 (position pleine),
    diff() = 1 : tu as "tourné" 100% de ton capital sur cet actif -> coût plein.
    Si tu restes à la même position d'un jour sur l'autre, diff() = 0 -> pas de coût.

    .abs() : un changement de +1 à -1 (on inverse totalement sa position) coûte
    le même prix qu'un changement de 0 à 1 en proportion du capital bougé —
    le signe du mouvement n'a pas d'importance pour le coût, seule l'ampleur compte.

    cost_bps est en points de base : 1 bp = 0.01% = 0.0001 en décimal.
    Ex. cost_bps=5 -> chaque unité de position tournée coûte 0.05%.
    """
    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0].abs()  # le tout premier trade a aussi un coût
    return turnover * (cost_bps / 10_000)


def run_backtest(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """
    Lance un backtest complet.

    Paramètres :
      returns   : rendements log journaliers par actif (sortie de data_loader.py)
      signals   : signal par actif, même forme que returns. Valeurs attendues :
                  1 = position longue pleine, 0 = pas de position, -1 = position
                  courte pleine (les stratégies de l'étape 4 produiront ça).
      cost_bps  : coût de transaction en points de base par unité de turnover.

    Retourne un dict avec toutes les étapes intermédiaires, pas seulement le
    résultat final — utile pour déboguer ("à quelle étape le calcul part en vrille ?")
    et pour montrer, en entretien, que tu comprends chaque maillon de la chaîne.
    """
    # Aligner les deux tableaux sur les mêmes dates et les mêmes actifs, au cas
    # où l'un contiendrait des colonnes ou des dates que l'autre n'a pas.
    returns, signals = returns.align(signals, join="inner", axis=None)

    positions = shift_signals_to_positions(signals)

    # Rendement brut par actif = position tenue x rendement réalisé ce jour-là.
    gross_returns = positions * returns

    costs = compute_transaction_costs(positions, cost_bps)

    net_returns = gross_returns - costs

    # Rendement du portefeuille = moyenne équipondérée des actifs chaque jour.
    # (Étape volontairement simple : pondération plus fine viendra si besoin,
    # pas nécessaire pour valider que le moteur fonctionne correctement.)
    portfolio_net_return = net_returns.mean(axis=1)

    # Courbe de capital : on part de 1 (100% du capital initial) et on
    # accumule les rendements log. exp(cumsum(...)) transforme une somme de
    # rendements log en facteur multiplicatif de capital — c'est exactement
    # pour cette propriété qu'on a choisi les rendements log à l'étape 2.
    equity_curve = np.exp(portfolio_net_return.cumsum())

    return {
        "positions": positions,
        "gross_returns": gross_returns,
        "costs": costs,
        "net_returns": net_returns,
        "portfolio_net_return": portfolio_net_return,
        "equity_curve": equity_curve,
    }


if __name__ == "__main__":
    # Test de fumée ("smoke test") : on ne teste PAS encore une vraie stratégie
    # (ça, c'est l'étape 4) — on vérifie juste que la plomberie du moteur
    # fonctionne, avec la stratégie la plus simple possible : buy-and-hold,
    # c'est-à-dire signal = 1 tout le temps, sur tous les actifs.
    returns = pd.read_csv("data/clean/returns.csv", index_col=0, parse_dates=True)

    signals = pd.DataFrame(1, index=returns.index, columns=returns.columns)

    result = run_backtest(returns, signals, cost_bps=5.0)

    print("Positions (5 premières lignes) :")
    print(result["positions"].head())
    print("\nRendement net du portefeuille (5 premières lignes) :")
    print(result["portfolio_net_return"].head())
    print("\nValeur finale du capital (départ = 1.0) :")
    print(result["equity_curve"].iloc[-1])
