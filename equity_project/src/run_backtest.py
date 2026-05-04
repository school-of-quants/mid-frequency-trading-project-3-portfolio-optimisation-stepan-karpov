import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml

logger = logging.getLogger(__name__)

project_path = Path(__file__).parent.parent

K = 5  # частота ребалансировки (торговых дней)


def run_inference() -> None:
    """Прогоняет обе модели на бэктест-данных, сохраняет predictions_backtest.parquet."""
    ld         = project_path / "data" / "learning-dataset"
    models_dir = project_path / "models"

    logger.info("[run_inference] загрузка моделей")
    model_tb = joblib.load(models_dir / "model_tb.joblib")
    model_q  = joblib.load(models_dir / "model_q.joblib")

    logger.info("[run_inference] загрузка бэктест-фич")
    X_bt_tb = pd.read_parquet(ld / "X_tb_backtest.parquet")
    X_bt_q  = pd.read_parquet(ld / "X_q_backtest.parquet")

    logger.info("[run_inference] инференс Triple Barrier")
    proba_tb = model_tb.predict_proba(X_bt_tb)
    classes  = model_tb.classes_.astype(int)
    col_map  = {-1: "prob_short", 0: "prob_neutral", 1: "prob_long"}

    df_preds = pd.DataFrame(
        proba_tb,
        index=X_bt_tb.index,
        columns=[col_map[c] for c in classes],
    )
    df_preds["pred_tb"] = model_tb.predict(X_bt_tb).ravel().astype(int)

    logger.info("[run_inference] инференс Quintile")
    df_preds["pred_q"] = model_q.predict(X_bt_q)

    df_preds["net_signal"] = df_preds["prob_long"] - df_preds["prob_short"]

    out_path = ld / "predictions_backtest.parquet"
    df_preds.to_parquet(out_path)
    logger.info(f"[run_inference] сохранено → {out_path}  {df_preds.shape}")


def run_backtest() -> None:
    """Запускает бэктест стратегии, сохраняет метрики и PnL-график."""
    with open(project_path.parent / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    BT_START  = pd.Timestamp(cfg["backtest_start_date"])
    BT_END    = pd.Timestamp(cfg["backtest_end_date"])
    INIT_CASH = cfg["init_cash"]
    FEES      = cfg["fees"]

    ld = project_path / "data" / "learning-dataset"

    logger.info("[run_backtest] загрузка предсказаний и цен")
    preds = pd.read_parquet(ld / "predictions_backtest.parquet")
    all_data = pd.read_csv(
        project_path / "data" / "processed" / "all_data_features_extended.csv",
        index_col=0, header=[0, 1],
        parse_dates=True,
    )
    all_data.index = pd.to_datetime(all_data.index)
    all_data.index.name = "Date"

    # Фильтрация по диапазону бэктеста
    bt_mask  = (preds.index.get_level_values("Date") >= BT_START) & \
               (preds.index.get_level_values("Date") <= BT_END)
    preds    = preds[bt_mask]
    bt_dates = preds.index.get_level_values("Date").unique().sort_values()

    # Цена исполнения — Open следующего дня
    tickers = preds.index.get_level_values("Ticker").unique()
    open_prices = (
        all_data["Open"][tickers]
        .shift(-1)
        .loc[bt_dates]
        .dropna(how="all")
    )
    daily_ratio = open_prices / open_prices.shift(1)
    open_prices = open_prices.where((daily_ratio >= 0.5) & (daily_ratio <= 2.0), other=np.nan)
    open_prices = open_prices.ffill()

    # Формирование весов
    net_signal = preds["net_signal"].unstack("Ticker").reindex(columns=tickers)
    pred_q     = preds["pred_q"].unstack("Ticker").reindex(columns=tickers)

    score = net_signal * pred_q
    score[score <= 0] = 0

    common_dates   = score.index.intersection(open_prices.index)
    common_tickers = score.columns.intersection(open_prices.columns)
    score       = score.loc[common_dates, common_tickers]
    open_prices = open_prices.loc[common_dates, common_tickers]
    open_prices.index.name = None
    open_prices.columns.name = None

    dates = score.index
    rebal_mask = np.zeros(len(dates), dtype=bool)
    rebal_mask[::K] = True

    weights = score.div(score.sum(axis=1).replace(0, np.nan), axis=0)
    weights = weights.where(pd.Series(rebal_mask, index=dates), other=np.nan)
    weights = weights.ffill()
    weights = weights.reindex_like(open_prices).fillna(0)

    logger.info(f"[run_backtest] K={K}, дат: {len(dates)}, ребалансировок: {rebal_mask.sum()}")

    # Бэктест
    pf = vbt.Portfolio.from_orders(
        close=open_prices,
        size=weights,
        size_type="targetpercent",
        init_cash=INIT_CASH,
        fees=FEES,
        freq="1D",
        group_by=True,
        cash_sharing=True,
    )

    # Метрики
    equity = pf.value()
    rets   = pf.returns()
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n_years   = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr      = (1 + total_ret) ** (1 / n_years) - 1
    sharpe    = rets.mean() / rets.std() * np.sqrt(252)
    max_dd    = (equity / equity.cummax() - 1).min()
    calmar    = cagr / abs(max_dd)

    metrics = {
        "K": K,
        "CAGR": round(cagr, 4),
        "Sharpe": round(sharpe, 4),
        "Max_Drawdown": round(max_dd, 4),
        "Calmar": round(calmar, 4),
        "Total_Return": round(total_ret, 4),
    }
    logger.info(f"[run_backtest] CAGR={cagr:.1%}  Sharpe={sharpe:.2f}  MaxDD={max_dd:.1%}")

    # Сохранение артефактов
    artifacts = project_path / "artifacts"
    (artifacts / "metrics").mkdir(parents=True, exist_ok=True)
    (artifacts / "plots").mkdir(parents=True, exist_ok=True)

    with open(artifacts / "metrics" / "backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"[run_backtest] метрики сохранены → {artifacts / 'metrics' / 'backtest_metrics.json'}")

    try:
        pf.plot().write_image(str(artifacts / "plots" / "pnl.png"))
        logger.info(f"[run_backtest] график сохранён → {artifacts / 'plots' / 'pnl.png'}")
    except Exception as e:
        logger.warning(f"[run_backtest] не удалось сохранить график: {e}")


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    run_inference()
    run_backtest()
