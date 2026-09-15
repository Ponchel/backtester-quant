"""
main.py -- Single entry point. Reproduces the whole study end to end.

Run from the repository root:

    .venv\\Scripts\\activate
    python main.py

Outputs written to reports/:
    results.csv          performance table, machine readable
    results.md           same table, ready to paste into the README
    ml_accuracy.md       out-of-sample accuracy, per asset
    equity_curves.png    cumulative capital over the common evaluation window

Design note: strategy signals need the full price history to be computed
(a 50-day moving average needs the 50 preceding days), so the backtest runs
on the entire sample and performance is only measured from `common_start`
onwards. Truncating the returns beforehand would start every strategy blind
and penalise it for a purely mechanical reason.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a CI box or a fresh clone
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import clean_data, download_raw
from src.engine import run_backtest
from src.evaluation import summarize
from src.strategies import (
    ml_logistic,
    mean_reversion,
    momentum,
    moving_average_crossover,
    to_long_only,
)

ROOT = Path(__file__).resolve().parent
CLEAN_CSV = ROOT / "data" / "clean" / "returns.csv"
REPORTS = ROOT / "reports"
COST_BPS = 5.0

COLUMNS = {
    "rendement_annualise": "return %/yr",
    "volatilite_annualisee": "vol %/yr",
    "sharpe": "sharpe",
    "max_drawdown": "max DD %",
    "capital_final": "final capital",
}


def load_returns() -> pd.DataFrame:
    """Load cached returns, downloading them first if the cache is absent.

    data/ is git-ignored, so a fresh clone has no cache and this triggers a
    download. That is what makes `python main.py` enough to reproduce the
    study from scratch.
    """
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV, index_col=0, parse_dates=True)

    print("No cached returns found -- downloading from Yahoo Finance.")
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)

    download_raw()
    returns = clean_data()

    if not CLEAN_CSV.exists():
        returns.to_csv(CLEAN_CSV)
    return returns


def build_signals(returns: pd.DataFrame, ml_signal: pd.DataFrame) -> dict:
    """All strategies long-only, so the comparison against buy-and-hold is fair.

    See the `to_long_only` docstring: a strategy allowed to go short is
    structurally penalised over a decade-long bull market, which would make
    the comparison meaningless.
    """
    return {
        "buy_and_hold": pd.DataFrame(
            1.0, index=returns.index, columns=returns.columns
        ),
        "moving_average": to_long_only(moving_average_crossover(returns)),
        "momentum": to_long_only(momentum(returns)),
        "mean_reversion": to_long_only(mean_reversion(returns)),
        "ml_logistic": to_long_only(ml_signal),
    }


def format_table(rows: dict) -> pd.DataFrame:
    table = pd.DataFrame(rows).T
    for key in ("rendement_annualise", "volatilite_annualisee", "max_drawdown"):
        table[key] = (table[key] * 100).round(2)
    table["sharpe"] = table["sharpe"].round(3)
    table["capital_final"] = table["capital_final"].round(3)
    return table[list(COLUMNS)].rename(columns=COLUMNS)


def to_markdown(table: pd.DataFrame, index_name: str) -> str:
    """Render a DataFrame as a markdown table.

    Hand-rolled on purpose: pandas' own `to_markdown` needs the `tabulate`
    package, and one more dependency is not worth a report header.
    """
    headers = [index_name] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for name, row in table.iterrows():
        cells = [str(name)] + [f"{v}" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_curves(curves: dict, start, end, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, curve in curves.items():
        style = "-" if name == "buy_and_hold" else "--"
        width = 2.4 if name == "buy_and_hold" else 1.4
        ax.plot(curve.index, curve.values, style, linewidth=width, label=name)

    ax.set_title(
        f"Cumulative capital, long-only, {COST_BPS:.0f}bp per unit of turnover\n"
        f"common evaluation window {start.date()} to {end.date()}"
    )
    ax.set_ylabel("capital (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)

    returns = load_returns()
    print(f"Universe : {', '.join(returns.columns)}")
    print(
        f"History  : {returns.index[0].date()} to {returns.index[-1].date()} "
        f"({len(returns)} trading days)\n"
    )

    print("Training the logistic model -- one per asset, chronological split:")
    ml_signal = ml_logistic(returns)
    accuracy = ml_signal.attrs.get("accuracy", {})

    signals = build_signals(returns, ml_signal)

    # The ML model only holds a position over its test period (the most recent
    # 30% of dates). Comparing its final capital against strategies that traded
    # for ten years would compare a 3 km runner to a 10 km one, so every
    # strategy is measured from the first date the model is actually active.
    active_dates = ml_signal[(ml_signal != 0).any(axis=1)].index
    common_start = active_dates[0]
    common_end = returns.index[-1]
    window_length = len(returns.loc[common_start:])
    print(
        f"\nCommon evaluation window: {common_start.date()} to "
        f"{common_end.date()} ({window_length} trading days)\n"
    )

    rows, curves = {}, {}
    for name, signal in signals.items():
        result = run_backtest(returns, signal, cost_bps=COST_BPS)
        rows[name] = summarize(result["portfolio_net_return"].loc[common_start:])
        curve = result["equity_curve"].loc[common_start:]
        curves[name] = curve / curve.iloc[0]

    table = format_table(rows)
    print("--- All strategies long-only, common window ---")
    print(table.to_string())

    table.to_csv(REPORTS / "results.csv")
    (REPORTS / "results.md").write_text(
        f"Common evaluation window: {common_start.date()} to {common_end.date()} "
        f"({window_length} trading days). Long-only, {COST_BPS:.0f}bp per unit "
        f"of turnover.\n\n" + to_markdown(table, "strategy") + "\n",
        encoding="utf-8",
    )

    if accuracy:
        acc = pd.DataFrame(accuracy).T.round(3)
        acc.index.name = "asset"
        print("\n--- Logistic regression accuracy (0.500 = coin flip) ---")
        print(acc.to_string())
        (REPORTS / "ml_accuracy.md").write_text(
            "Directional accuracy of the per-asset logistic regression. "
            "The test split is the most recent 30% of dates, never shuffled.\n\n"
            + to_markdown(acc, "asset")
            + "\n",
            encoding="utf-8",
        )

    plot_curves(curves, common_start, common_end, REPORTS / "equity_curves.png")
    print(f"\nWritten to {REPORTS.name}/: results.csv, results.md, equity_curves.png")


if __name__ == "__main__":
    main()
