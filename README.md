# Backtester — projet quant

## Objectif
Ce dépôt n'a pas pour but de produire une stratégie rentable. Il sert de
démonstration d'un **protocole de test rigoureux**, présentable en entretien
de stage (data/ML, finance quantitative).

Un backtest qui affiche un Sharpe de 3 se fait démonter en dix minutes.
Une analyse honnête de *pourquoi* des stratégies simples ne survivent pas
hors échantillon est plus difficile à réfuter et plus utile à discuter.

## Roadmap
- [ ] **1. Socle** — structure du dépôt, environnement, dépôt Git (cette étape)
- [ ] **2. Données** — récupération et nettoyage (source, fréquence, ajustement des dividendes/splits)
- [ ] **3. Moteur de backtest** — exécution avec coûts de transaction et slippage
- [ ] **4. Stratégies simples** — moyennes mobiles, momentum, mean reversion
- [ ] **5. Évaluation** — Sharpe, max drawdown, test hors échantillon, rapport écrit

## Biais à traquer à chaque étape
- **Look-ahead bias** : aucune information non disponible à l'instant *t* ne
  doit entrer dans une décision prise à *t*. Vérifier systématiquement le
  décalage (shift) entre signal et exécution.
- **Survivorship bias** : ne pas tester uniquement sur les sociétés encore
  cotées aujourd'hui — l'univers d'actifs doit refléter ce qui existait
  réellement à chaque date historique.

## Structure du dépôt
```
backtester-quant/
├── data/           # données brutes et nettoyées (non versionnées, voir .gitignore)
├── src/
│   ├── data_loader.py     # étape 2 : récupération / nettoyage
│   ├── engine.py          # étape 3 : moteur de backtest
│   ├── strategies.py      # étape 4 : stratégies
│   └── evaluation.py      # étape 5 : métriques
├── notebooks/      # exploration, non destinée à la prod
├── reports/        # rapport écrit final
├── requirements.txt
└── README.md
```

## Environnement
Python 3.14. Voir `requirements.txt`.
