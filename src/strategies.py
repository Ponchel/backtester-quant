"""
strategies.py — Étape 4 : les stratégies

Chaque fonction ci-dessous prend les rendements (sortie de data_loader.py) et
retourne un tableau de SIGNAUX (même forme que returns, valeurs 1 / 0 / -1),
calculés avec l'information disponible JUSQU'AU jour t inclus.

Rappel de convention (posée dans engine.py) : ces fonctions ne décalent JAMAIS
elles-mêmes leurs signaux. C'est le moteur (engine.py) qui applique le décalage
d'un jour avant d'exécuter. Une stratégie n'a donc jamais à se soucier du
look-ahead bias au moment de l'exécution — seulement à ne jamais utiliser,
dans son calcul au jour t, une donnée qui ne serait connue qu'après t.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def reconstruct_price_index(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruit un indice de prix relatif (parti de 1.0) à partir des
    rendements log. Utile pour les stratégies qui ont besoin d'un NIVEAU de
    prix (moyennes mobiles, mean reversion), pas seulement d'un rendement.

    Pourquoi c'est valable d'utiliser un indice relatif plutôt que le vrai
    prix en dollars : une moyenne mobile ou un z-score ne s'intéressent qu'à
    la FORME de la série (est-ce que ça monte, est-ce que c'est loin de sa
    moyenne récente) — pas à son échelle absolue. Multiplier toute une série
    par une constante ne change ni où se croisent deux moyennes mobiles, ni
    où se situe un z-score. C'est une simplification légitime, pas un raccourci
    dangereux.
    """
    return np.exp(returns.cumsum())


def moving_average_crossover(
    returns: pd.DataFrame, short_window: int = 20, long_window: int = 50
) -> pd.DataFrame:
    """Long quand la moyenne courte dépasse la longue, short sinon."""
    prices = reconstruct_price_index(returns)
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()

    signal = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    signal[ma_short > ma_long] = 1.0
    signal[ma_short < ma_long] = -1.0
    # Pendant la période de "chauffe" (avant d'avoir 50 jours d'historique),
    # ma_long est NaN -> la comparaison est automatiquement False des deux
    # côtés -> signal reste à 0 (pas de position). Comportement voulu, pas
    # besoin de le gérer explicitement.
    return signal


def momentum(returns: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Long si le rendement cumulé sur `lookback` jours est positif, short sinon."""
    # La somme de rendements LOG sur une fenêtre = le rendement total log sur
    # cette fenêtre (propriété d'additivité vue à l'étape 2).
    cum_return = returns.rolling(lookback).sum()

    signal = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    signal[cum_return > 0] = 1.0
    signal[cum_return < 0] = -1.0
    return signal


def mean_reversion(
    returns: pd.DataFrame, lookback: int = 5, z_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Long si le prix est "survendu" (loin en-dessous de sa moyenne récente),
    short si "suracheté" (loin au-dessus) — le pari inverse du momentum.
    """
    prices = reconstruct_price_index(returns)
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std = prices.rolling(lookback).std()
    z_score = (prices - rolling_mean) / rolling_std

    signal = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    signal[z_score < -z_threshold] = 1.0   # survendu -> on parie sur un rebond
    signal[z_score > z_threshold] = -1.0   # suracheté -> on parie sur un repli
    return signal


def ml_logistic(
    returns: pd.DataFrame, n_lags: int = 5, train_frac: float = 0.7
) -> pd.DataFrame:
    """
    Un modèle de régression logistique PAR ACTIF, entraîné à prédire le SIGNE
    du rendement du LENDEMAIN à partir des `n_lags` derniers rendements connus.

    Le split est délibérément chronologique et non aléatoire : train_frac=0.7
    veut dire "les 70% de dates les plus anciennes servent à entraîner, les
    30% les plus récentes à tester" — jamais de mélange. C'est la règle la
    plus importante de cette fonction, plus importante que le choix du modèle
    lui-même.
    """
    signal = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

    for ticker in returns.columns:
        series = returns[ticker].dropna()

        # Features : le rendement d'aujourd'hui (lag_0) et des n_lags-1 jours
        # précédents. Tout ça est connu au moment de la clôture du jour t.
        features = pd.concat(
            {f"lag_{i}": series.shift(i) for i in range(n_lags)}, axis=1
        )
        # Cible : le SIGNE du rendement de DEMAIN (shift(-1) regarde en avant
        # dans la série -> c'est volontaire, c'est la cible qu'on veut prédire,
        # pas une feature. Ne jamais faire ça sur une colonne de features.)
        target = (series.shift(-1) > 0).astype(int)

        data = features.join(target.rename("target")).dropna()

        split_idx = int(len(data) * train_frac)
        train, test = data.iloc[:split_idx], data.iloc[split_idx:]

        if len(train) < 100 or len(test) < 20:
            print(f"  {ticker} : pas assez de données pour entraîner, ignoré.")
            continue

        model = LogisticRegression(max_iter=1000)
        model.fit(train.drop(columns="target"), train["target"])

        predictions = model.predict(test.drop(columns="target"))
        # predict() renvoie 0 ou 1 -> on convertit en signal -1 / +1 attendu
        # par le moteur.
        pred_signal = np.where(predictions == 1, 1.0, -1.0)

        signal.loc[test.index, ticker] = pred_signal
        train_accuracy = model.score(train.drop(columns="target"), train["target"])
        test_accuracy = model.score(test.drop(columns="target"), test["target"])
        print(f"  {ticker} : accuracy train={train_accuracy:.3f}  test={test_accuracy:.3f}")

    return signal


def to_long_only(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit un signal long/short en signal long-only : tout -1 devient 0.

    Pourquoi c'est nécessaire pour une comparaison honnête : le buy-and-hold
    est long à 100% du temps. Sur une décennie de marché haussier, une
    stratégie autorisée à être short est structurellement pénalisée — la
    comparer directement au buy-and-hold reviendrait à lui reprocher un
    handicap qu'on lui a imposé. La version long-only teste une question
    plus propre : la stratégie sait-elle au moins ÉVITER les mauvaises
    périodes, en restant simplement hors du marché ?

    Effet secondaire à noter : passer de -1 à 0 réduit aussi le turnover
    (moins d'allers-retours de position), donc les coûts de transaction
    baissent. Ça joue en faveur des stratégies actives — raison de plus
    pour que cette comparaison soit la bonne à présenter.
    """
    return signal.clip(lower=0)


if __name__ == "__main__":
    from engine import run_backtest

    returns = pd.read_csv("data/clean/returns.csv", index_col=0, parse_dates=True)

    strategies = {
        "moving_average": moving_average_crossover(returns),
        "momentum": momentum(returns),
        "mean_reversion": mean_reversion(returns),
    }

    print("Entraînement du modèle ML (régression logistique par actif)...")
    strategies["ml_logistic"] = ml_logistic(returns)

    buy_and_hold = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    bh_capital = run_backtest(returns, buy_and_hold, cost_bps=5.0)["equity_curve"].iloc[-1]

    print("\n--- Capital final (départ = 1.0), sur 10 ans ---")
    print(f"  {'buy_and_hold':16s} : {bh_capital:.3f}  (référence)")
    print(f"\n  {'stratégie':16s}   {'long/short':>10s}  {'long-only':>10s}")

    for name, signal in strategies.items():
        ls = run_backtest(returns, signal, cost_bps=5.0)["equity_curve"].iloc[-1]
        lo = run_backtest(returns, to_long_only(signal), cost_bps=5.0)["equity_curve"].iloc[-1]
        print(f"  {name:16s} {ls:>10.3f}  {lo:>10.3f}")
