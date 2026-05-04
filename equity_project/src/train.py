import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

project_path = Path(__file__).parent.parent

PARAM_GRID = [
    {"depth": 4, "learning_rate": 0.10, "l2_leaf_reg":  3},
    {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg":  3},
    {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 10},
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg":  5},
    {"depth": 6, "learning_rate": 0.10, "l2_leaf_reg":  3},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg":  3},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg":  5},
    {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg":  3},
    {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10},
    {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg":  5},
]


def tscv_with_embargo(dates, n_splits=3, embargo_days=22):
    """TimeSeriesSplit по уникальным датам с embargo между фолдами."""
    unique_dates = np.sort(dates.unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for d_train_idx, d_val_idx in tscv.split(unique_dates):
        last_train = pd.Timestamp(unique_dates[d_train_idx[-1]])
        cutoff     = last_train + pd.Timedelta(days=embargo_days)
        val_dates  = unique_dates[d_val_idx]
        val_dates  = val_dates[val_dates > np.datetime64(cutoff)]
        if len(val_dates) == 0:
            continue
        train_mask = dates.isin(unique_dates[d_train_idx])
        val_mask   = dates.isin(val_dates)
        yield np.where(train_mask)[0], np.where(val_mask)[0]


def compute_ic(y_true: pd.Series, y_pred) -> float:
    """Mean per-date Pearson correlation между предсказаниями и таргетом."""
    df = pd.DataFrame({"pred": y_pred, "true": y_true.values}, index=y_true.index)
    per_date = df.groupby(level="Date").apply(lambda g: g["pred"].corr(g["true"]))
    return per_date.mean()


def train_tb(ld: Path, models_dir: Path) -> None:
    """Обучает Triple Barrier CatBoostClassifier с grid search."""
    logger.info("[train_tb] загрузка данных")
    X_train = pd.read_parquet(ld / "X_tb_train.parquet")
    y_train = pd.read_parquet(ld / "y_tb_train.parquet")["TripleBarrier"].astype(int)

    dates  = X_train.index.get_level_values("Date")
    splits = list(tscv_with_embargo(dates, n_splits=3, embargo_days=22))
    logger.info(f"[train_tb] CV фолдов: {len(splits)}")

    logger.info("[train_tb] grid search")
    gs_results = []
    for params in PARAM_GRID:
        fold_scores = []
        for train_idx, val_idx in splits:
            model = CatBoostClassifier(
                loss_function="MultiClass",
                iterations=1000,
                early_stopping_rounds=30,
                thread_count=-1,
                random_seed=42,
                verbose=0,
                **params,
            )
            model.fit(
                X_train.iloc[train_idx], y_train.iloc[train_idx],
                eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
            )
            preds = model.predict(X_train.iloc[val_idx]).ravel().astype(int)
            fold_scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))

        row = {**params, "mean_f1": np.mean(fold_scores), "std_f1": np.std(fold_scores)}
        gs_results.append(row)
        logger.info(f"  depth={params['depth']}, lr={params['learning_rate']:.2f}"
                    f"  →  F1={row['mean_f1']:.4f} ± {row['std_f1']:.4f}")

    gs_df = pd.DataFrame(gs_results).sort_values("mean_f1", ascending=False).reset_index(drop=True)
    best = gs_df.iloc[0]
    logger.info(f"[train_tb] лучшие параметры: depth={int(best['depth'])}, lr={best['learning_rate']}")

    # Финальная модель — последние 15% дат как holdout для early stopping
    unique_dates  = np.sort(X_train.index.get_level_values("Date").unique())
    n_holdout     = max(1, int(len(unique_dates) * 0.15))
    holdout_start = pd.Timestamp(unique_dates[-n_holdout])

    final_train_mask = X_train.index.get_level_values("Date") < holdout_start
    final_eval_mask  = X_train.index.get_level_values("Date") >= holdout_start

    model_tb = CatBoostClassifier(
        loss_function="MultiClass",
        depth=int(best["depth"]),
        learning_rate=float(best["learning_rate"]),
        l2_leaf_reg=int(best["l2_leaf_reg"]),
        iterations=2000,
        early_stopping_rounds=50,
        thread_count=-1,
        random_seed=42,
        verbose=100,
    )
    model_tb.fit(
        X_train[final_train_mask], y_train[final_train_mask],
        eval_set=(X_train[final_eval_mask], y_train[final_eval_mask]),
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_tb, models_dir / "model_tb.joblib")
    logger.info(f"[train_tb] модель сохранена → {models_dir / 'model_tb.joblib'}")
    logger.info(f"[train_tb] лучшая итерация: {model_tb.best_iteration_}")


def train_q(ld: Path, models_dir: Path) -> None:
    """Обучает Quintile CatBoostRegressor с grid search."""
    logger.info("[train_q] загрузка данных")
    X_train_q = pd.read_parquet(ld / "X_q_train.parquet")
    y_train_q = pd.read_parquet(ld / "y_q_train.parquet")["Quintile"]

    dates_q  = X_train_q.index.get_level_values("Date")
    splits_q = list(tscv_with_embargo(dates_q, n_splits=3, embargo_days=22))
    logger.info(f"[train_q] CV фолдов: {len(splits_q)}")

    logger.info("[train_q] grid search")
    gs_results_q = []
    for params in PARAM_GRID:
        fold_scores = []
        for train_idx, val_idx in splits_q:
            model = CatBoostRegressor(
                loss_function="RMSE",
                iterations=1000,
                early_stopping_rounds=30,
                thread_count=-1,
                random_seed=42,
                verbose=0,
                **params,
            )
            model.fit(
                X_train_q.iloc[train_idx], y_train_q.iloc[train_idx],
                eval_set=(X_train_q.iloc[val_idx], y_train_q.iloc[val_idx]),
            )
            preds = model.predict(X_train_q.iloc[val_idx])
            fold_scores.append(compute_ic(y_train_q.iloc[val_idx], preds))

        row = {**params, "mean_ic": np.mean(fold_scores), "std_ic": np.std(fold_scores)}
        gs_results_q.append(row)
        logger.info(f"  depth={params['depth']}, lr={params['learning_rate']:.2f}"
                    f"  →  IC={row['mean_ic']:.4f} ± {row['std_ic']:.4f}")

    gs_df_q = pd.DataFrame(gs_results_q).sort_values("mean_ic", ascending=False).reset_index(drop=True)
    best_q = gs_df_q.iloc[0]
    logger.info(f"[train_q] лучшие параметры: depth={int(best_q['depth'])}, lr={best_q['learning_rate']}")

    unique_dates_q  = np.sort(X_train_q.index.get_level_values("Date").unique())
    n_holdout_q     = max(1, int(len(unique_dates_q) * 0.15))
    holdout_start_q = pd.Timestamp(unique_dates_q[-n_holdout_q])

    final_train_mask_q = X_train_q.index.get_level_values("Date") < holdout_start_q
    final_eval_mask_q  = X_train_q.index.get_level_values("Date") >= holdout_start_q

    model_q = CatBoostRegressor(
        loss_function="RMSE",
        depth=int(best_q["depth"]),
        learning_rate=float(best_q["learning_rate"]),
        l2_leaf_reg=int(best_q["l2_leaf_reg"]),
        iterations=2000,
        early_stopping_rounds=50,
        thread_count=-1,
        random_seed=42,
        verbose=100,
    )
    model_q.fit(
        X_train_q[final_train_mask_q], y_train_q[final_train_mask_q],
        eval_set=(X_train_q[final_eval_mask_q], y_train_q[final_eval_mask_q]),
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_q, models_dir / "model_q.joblib")
    logger.info(f"[train_q] модель сохранена → {models_dir / 'model_q.joblib'}")
    logger.info(f"[train_q] лучшая итерация: {model_q.best_iteration_}")


def train() -> None:
    """Обучает обе модели (Triple Barrier + Quintile) и сохраняет в models/."""
    ld         = project_path / "data" / "learning-dataset"
    models_dir = project_path / "models"

    train_tb(ld, models_dir)
    train_q(ld, models_dir)


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    train()
