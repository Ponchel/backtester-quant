"""
evaluation.py — Étape 5 : métriques et comparaison rigoureuse

Trois métriques, et une comparaison sur fenêtre commune.

Convention : toutes les fonctions prennent une série de rendements LOG
journaliers de portefeuille (la sortie "portfolio_net_return" du moteur).
"""

import pandas as pd
import numpy as np

# Nombre de jours de bourse par an. Utilisé pour annualiser des statistiques
# calculées sur des données journalières. 252 est la convention standard du
# secteur (365 jours moins week-ends et jours fériés).
TRADING_DAYS = 252


def annualized_return(portfolio_returns: pd.Series) -> float:
    """
    Rendement annualisé composé, exprimé en décimal (0.164 = 16.4%/an).

    On somme les rendements log (ils s'additionnent, propriété vue à l'étape 2),
    on ramène à une base annuelle, puis exp() - 1 convertit le rendement log
    en rendement "normal" lisible par un humain.
    """
    total_log_return = portfolio_returns.sum()
    n_years = len(portfolio_returns) / TRADING_DAYS
    return np.exp(total_log_return / n_years) - 1


def annualized_volatility(portfolio_returns: pd.Series) -> float:
    """
    Volatilité annualisée = écart-type journalier x racine(252).

    Pourquoi racine(252) et pas 252 : la volatilité croît avec la RACINE du
    temps, pas linéairement (conséquence du fait que les variations
    journalières se compensent partiellement au lieu de s'additionner).
    C'est une règle à connaître, elle revient constamment en finance.
    """
    return portfolio_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(portfolio_returns: pd.Series) -> float:
    """
    Rendement annualisé / volatilité annualisée.

    SIMPLIFICATION ASSUMÉE : le Sharpe rigoureux soustrait le taux sans risque
    du rendement avant de diviser. On l'ignore ici (implicitement supposé nul).
    Sur 2015-2025 les taux ont varié de ~0% à ~5%, donc cette approximation
    surestime légèrement le Sharpe. À mentionner dans le rapport, pas à cacher.
    """
    vol = annualized_volatility(portfolio_returns)
    if vol == 0:
        return np.nan
    return annualized_return(portfolio_returns) / vol


def max_drawdown(portfolio_returns: pd.Series) -> float:
    """
    Pire chute depuis un sommet historique, en décimal négatif (-0.35 = -35%).

    Méthode : on reconstruit la courbe de capital, on garde en mémoire le
    sommet atteint à chaque instant (cummax), et on mesure l'écart entre la
    valeur courante et ce sommet. Le minimum de tous ces écarts est le
    max drawdown.
    """
    equity = np.exp(portfolio_returns.cumsum())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def summarize(portfolio_returns: pd.Series) -> dict:
    """Rassemble toutes les métriques d'une stratégie en un seul dict."""
    return {
        "rendement_annualise": annualized_return(portfolio_returns),
        "volatilite_annualisee": annualized_volatility(portfolio_returns),
        "sharpe": sharpe_ratio(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_returns),
        "capital_final": np.exp(portfolio_returns.sum()),
    }


if __name__ == "__main__":
    from engine import run_backtest
    from strategies import (
        moving_average_crossover,
        momentum,
        mean_reversion,
        ml_logistic,
        to_long_only,
    )

    returns = pd.read_csv("data/clean/returns.csv", index_col=0, parse_dates=True)

    print("Entraînement du modèle ML...")
    ml_signal = ml_logistic(returns)

    # FENÊTRE COMMUNE : le modèle ML ne produit de signal que sur sa période de
    # test (les 30% de dates les plus récentes). Comparer son capital final à
    # des stratégies qui tradent sur 10 ans n'a aucun sens — c'est comparer un
    # coureur qui a fait 3 km à un autre qui en a fait 10.
    # On identifie donc la première date où le ML prend réellement position,
    # et on évalue TOUT LE MONDE à partir de cette date.
    ml_active_dates = ml_signal[(ml_signal != 0).any(axis=1)].index
    common_start = ml_active_dates[0]
    print(f"\nFenêtre commune d'évaluation : {common_start.date()} -> {returns.index[-1].date()}")
    print(f"({len(returns.loc[common_start:])} jours de bourse)\n")

    signals = {
        "buy_and_hold": pd.DataFrame(1.0, index=returns.index, columns=returns.columns),
        "moving_average": to_long_only(moving_average_crossover(returns)),
        "momentum": to_long_only(momentum(returns)),
        "mean_reversion": to_long_only(mean_reversion(returns)),
        "ml_logistic": to_long_only(ml_signal),
    }

    rows = {}
    for name, signal in signals.items():
        result = run_backtest(returns, signal, cost_bps=5.0)
        # On tronque APRÈS le backtest : les stratégies ont besoin de tout
        # l'historique pour calculer leurs signaux (une moyenne mobile 50 jours
        # a besoin des 50 jours précédents), on ne coupe donc qu'au moment
        # de mesurer la performance.
        windowed = result["portfolio_net_return"].loc[common_start:]
        rows[name] = summarize(windowed)

    table = pd.DataFrame(rows).T
    table["rendement_annualise"] = (table["rendement_annualise"] * 100).round(2)
    table["volatilite_annualisee"] = (table["volatilite_annualisee"] * 100).round(2)
    table["max_drawdown"] = (table["max_drawdown"] * 100).round(2)
    table["sharpe"] = table["sharpe"].round(3)
    table["capital_final"] = table["capital_final"].round(3)
    table.columns = ["rendement %/an", "volatilite %/an", "sharpe", "max DD %", "capital final"]

    print("--- Toutes stratégies en long-only, sur fenêtre commune ---")
    print(table.to_string())
