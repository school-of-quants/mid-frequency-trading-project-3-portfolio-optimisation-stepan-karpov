import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

project_path = Path(__file__).parent.parent

OUT_PATH = project_path / "data" / "processed" / "all_data_features_extended.csv"


# ── Таргеты ───────────────────────────────────────────────────────────────────

def compute_triple_barrier(
    close: pd.DataFrame,
    n_days: int,
    barrier_pct: float,
) -> pd.DataFrame:
    """Размечает каждую (дата, тикер) тройным барьером.

    Для точки t смотрим вперёд n_days дней и определяем какой барьер
    пробит первым: верхний (+barrier_pct), нижний (-barrier_pct)
    или вертикальный (время вышло).
    """
    arr = close.values.astype(float)
    T, N = arr.shape
    labels = np.full((T, N), np.nan)

    for i in range(T - n_days):
        p0 = arr[i]
        window = arr[i + 1: i + n_days + 1]
        ret = window / p0 - 1

        upper = ret >= barrier_pct
        lower = ret <= -barrier_pct

        first_up = np.where(upper.any(axis=0), upper.argmax(axis=0), n_days)
        first_dn = np.where(lower.any(axis=0), lower.argmax(axis=0), n_days)

        label = np.where(first_up < first_dn,  1.0,
                np.where(first_dn < first_up, -1.0, 0.0))
        label[np.isnan(p0)] = np.nan
        labels[i] = label

    return pd.DataFrame(labels, index=close.index, columns=close.columns)


def compute_quintile(close: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Размечает каждую (дата, тикер) квинтилем кросс-секционной доходности.

    Для каждой даты t считаем форвардную доходность за n_days дней,
    вискоризируем на уровне [1%, 99%], ранжируем по всем тикерам и
    делим на 5 равных бинов (1 = худшие 20%, 5 = лучшие 20%).
    """
    fwd_ret = close.shift(-n_days) / close - 1

    lo = fwd_ret.quantile(0.01, axis=1)
    hi = fwd_ret.quantile(0.99, axis=1)
    fwd_ret_clipped = fwd_ret.clip(lower=lo, upper=hi, axis=0)

    ranks = fwd_ret_clipped.rank(axis=1, pct=True)
    quintile = (ranks * 5).apply(np.floor).clip(upper=4) + 1

    quintile.iloc[-n_days:] = np.nan
    return quintile


# ── Фичи ──────────────────────────────────────────────────────────────────────

def add_ma_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет отклонения Close от скользящих средних (mean reversion фичи)."""
    c = df["Close"]
    new = pd.concat({
        "dev5":      (c - c.rolling(5).mean())   / c,
        "dev22":     (c - c.rolling(22).mean())  / c,
        "dev252":    (c - c.rolling(252).mean()) / c,
        "ma200vs50": (c.rolling(200).mean() - c.rolling(50).mean()) / c,
    }, axis=1)
    new.columns.names = ["Price", "Ticker"]
    df = pd.concat([df, new], axis=1).copy()
    df.to_csv(OUT_PATH)
    logger.info(f"[add_ma_deviations] +4 фичи → {OUT_PATH}  {df.shape}")
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет кросс-секционный ранг импульса (mom5, mom22, mom252)."""
    c = df["Close"]
    new = pd.concat({
        "mom5":   c.pct_change(5).rank(axis=1),
        "mom22":  c.pct_change(22).rank(axis=1),
        "mom252": (c.shift(22) / c.shift(252) - 1).rank(axis=1),
    }, axis=1)
    new.columns.names = ["Price", "Ticker"]
    df = pd.concat([df, new], axis=1).copy()
    df.to_csv(OUT_PATH)
    logger.info(f"[add_momentum] +3 фичи → {OUT_PATH}  {df.shape}")
    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет std дневных доходностей на разных горизонтах (vol5, vol22, vol252)."""
    c = df["Close"]
    ret = c.pct_change()
    new = pd.concat({
        "vol5":   ret.rolling(5).std(),
        "vol22":  ret.rolling(22).std(),
        "vol252": ret.rolling(252).std(),
    }, axis=1)
    new.columns.names = ["Price", "Ticker"]
    df = pd.concat([df, new], axis=1).copy()
    df.to_csv(OUT_PATH)
    logger.info(f"[add_volatility] +3 фичи → {OUT_PATH}  {df.shape}")
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет кросс-секционные индикаторы рыночного режима."""
    c = df["Close"]
    tickers = c.columns
    ret = c.pct_change()

    mkt_ret_22 = c.pct_change(22).median(axis=1)
    mkt_vol    = ret.std(axis=1)
    pct_above  = (c > c.rolling(200).mean()).mean(axis=1)

    def broadcast(series):
        return pd.DataFrame(
            np.tile(series.values[:, None], (1, len(tickers))),
            index=c.index, columns=tickers,
        )

    new = pd.concat({
        "market_ret_22":   broadcast(mkt_ret_22),
        "market_vol":      broadcast(mkt_vol),
        "pct_above_200ma": broadcast(pct_above),
    }, axis=1)
    new.columns.names = ["Price", "Ticker"]
    df = pd.concat([df, new], axis=1).copy()
    df.to_csv(OUT_PATH)
    logger.info(f"[add_market_regime] +3 фичи → {OUT_PATH}  {df.shape}")
    return df


def add_reversal_and_highs(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет краткосрочный реверсал и расстояние от 52-week high."""
    c = df["Close"]
    high_252 = c.rolling(252).max()
    new = pd.concat({
        "reversal": -c.pct_change(5),
        "dist_52h": (c - high_252) / high_252,
    }, axis=1)
    new.columns.names = ["Price", "Ticker"]
    df = pd.concat([df, new], axis=1).copy()
    df.to_csv(OUT_PATH)
    logger.info(f"[add_reversal_and_highs] +2 фичи → {OUT_PATH}  {df.shape}")
    return df


def add_amihud(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет Amihud illiquidity: rolling 22-day mean(|ret| / volume)."""
    c = df["Close"]
    v = df["Volume"].replace(0, np.nan)
    amihud = (c.pct_change().abs() / v).rolling(22).mean()
    new = pd.concat({"amihud": amihud}, axis=1)
    new.columns.names = ["Price", "Ticker"]
    df = pd.concat([df, new], axis=1).copy()
    df.to_csv(OUT_PATH)
    logger.info(f"[add_amihud] +1 фича → {OUT_PATH}  {df.shape}")
    return df


def add_beta(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет rolling 60-day бету к кросс-секциональному среднему."""
    c = df["Close"]
    ret = c.pct_change()
    mkt = ret.mean(axis=1)

    window = 60
    mkt_var = mkt.rolling(window).var()

    ri_rm = ret.multiply(mkt, axis=0)
    cov = (ri_rm.rolling(window).mean()
           - ret.rolling(window).mean().multiply(mkt.rolling(window).mean(), axis=0))
    beta = cov.divide(mkt_var, axis=0)

    new = pd.concat({"beta": beta}, axis=1)
    new.columns.names = ["Price", "Ticker"]
    df = pd.concat([df, new], axis=1).copy()
    df.to_csv(OUT_PATH)
    logger.info(f"[add_beta] +1 фича → {OUT_PATH}  {df.shape}")
    return df


# ── Главная функция ────────────────────────────────────────────────────────────

def build_features() -> None:
    """Строит таргеты, фичи и learning-dataset из data/processed/all_data.csv."""
    with open(project_path.parent / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    TRAIN_START    = cfg["train_start_date"]
    TRAIN_END      = cfg["train_end_date"]
    BACKTEST_START = cfg["backtest_start_date"]
    BACKTEST_END   = cfg["backtest_end_date"]

    N_DAYS      = 22
    BARRIER_PCT = 0.05

    # ── загрузка отфильтрованных данных ───────────────────────────────────────
    logger.info("[build_features] загрузка all_data.csv")
    all_data_filtered = pd.read_csv(
        project_path / "data" / "processed" / "all_data.csv",
        index_col=0,
        header=[0, 1],
    )
    all_data_filtered.index = pd.to_datetime(all_data_filtered.index)
    close_clean = all_data_filtered["Close"]

    # ── таргеты ───────────────────────────────────────────────────────────────
    logger.info("[build_features] вычисление Triple Barrier")
    tb = compute_triple_barrier(close_clean, N_DAYS, BARRIER_PCT)

    logger.info("[build_features] вычисление Quintile")
    quintile = compute_quintile(close_clean, N_DAYS)

    tb_wide = pd.concat({"TripleBarrier": tb}, axis=1)
    tb_wide.columns.names = ["Price", "Ticker"]
    q_wide = pd.concat({"Quintile": quintile}, axis=1)
    q_wide.columns.names = ["Price", "Ticker"]

    all_data_labeled = pd.concat([all_data_filtered, tb_wide, q_wide], axis=1)
    all_data_labeled = all_data_labeled.sort_index(axis=1)
    all_data_labeled.to_csv(project_path / "data" / "processed" / "all_data_labeled.csv")
    logger.info(f"[build_features] all_data_labeled.csv сохранён  {all_data_labeled.shape}")

    # ── инициализация all_data_features_extended.csv ──────────────────────────
    logger.info("[build_features] инициализация all_data_features_extended.csv")
    df = all_data_labeled.copy()
    df.to_csv(OUT_PATH)

    # ── добавление фич ────────────────────────────────────────────────────────
    df = add_ma_deviations(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_market_regime(df)
    df = add_reversal_and_highs(df)
    df = add_amihud(df)
    df = add_beta(df)

    # ── приведение к тренировочному формату ───────────────────────────────────
    logger.info("[build_features] построение learning-dataset")
    all_price_types = df.columns.get_level_values("Price").unique().tolist()
    OHLCV   = ["Close", "High", "Low", "Open", "Volume"]
    TARGETS = ["TripleBarrier", "Quintile"]
    FEATURES = [pt for pt in all_price_types if pt not in OHLCV + TARGETS]

    X_wide    = df[FEATURES].shift(1).iloc[260:]
    y_tb_wide = df["TripleBarrier"].loc[X_wide.index]
    y_q_wide  = df["Quintile"].loc[X_wide.index]

    X    = X_wide.stack(level=1).rename_axis(["Date", "Ticker"])
    y_tb = y_tb_wide.stack().rename("TripleBarrier").rename_axis(["Date", "Ticker"])
    y_q  = y_q_wide.stack().rename("Quintile").rename_axis(["Date", "Ticker"])

    valid_tb = y_tb.dropna().index
    valid_q  = y_q.dropna().index

    X_tb = X.loc[X.index.intersection(valid_tb)]
    X_q  = X.loc[X.index.intersection(valid_q)]
    y_tb = y_tb.loc[valid_tb]
    y_q  = y_q.loc[valid_q]

    def split(df, start, end):
        d = df.index.get_level_values("Date")
        return df[(d >= start) & (d <= end)]

    X_tb_train    = split(X_tb, TRAIN_START, TRAIN_END)
    X_tb_backtest = split(X_tb, BACKTEST_START, BACKTEST_END)
    X_q_train     = split(X_q,  TRAIN_START, TRAIN_END)
    X_q_backtest  = split(X_q,  BACKTEST_START, BACKTEST_END)

    y_tb_train    = split(y_tb.to_frame(), TRAIN_START, TRAIN_END)
    y_tb_backtest = split(y_tb.to_frame(), BACKTEST_START, BACKTEST_END)
    y_q_train     = split(y_q.to_frame(),  TRAIN_START, TRAIN_END)
    y_q_backtest  = split(y_q.to_frame(),  BACKTEST_START, BACKTEST_END)

    out = project_path / "data" / "learning-dataset"
    out.mkdir(parents=True, exist_ok=True)

    X_tb_train.to_parquet(out / "X_tb_train.parquet")
    X_tb_backtest.to_parquet(out / "X_tb_backtest.parquet")
    X_q_train.to_parquet(out / "X_q_train.parquet")
    X_q_backtest.to_parquet(out / "X_q_backtest.parquet")
    y_tb_train.to_parquet(out / "y_tb_train.parquet")
    y_tb_backtest.to_parquet(out / "y_tb_backtest.parquet")
    y_q_train.to_parquet(out / "y_q_train.parquet")
    y_q_backtest.to_parquet(out / "y_q_backtest.parquet")

    logger.info(f"[build_features] learning-dataset сохранён в {out}")
    for name, d in [
        ("X_tb_train", X_tb_train), ("X_tb_backtest", X_tb_backtest),
        ("y_tb_train", y_tb_train), ("y_tb_backtest", y_tb_backtest),
        ("X_q_train",  X_q_train),  ("X_q_backtest",  X_q_backtest),
        ("y_q_train",  y_q_train),  ("y_q_backtest",  y_q_backtest),
    ]:
        logger.info(f"  {name+'.parquet':<25} {d.shape}")


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    build_features()
