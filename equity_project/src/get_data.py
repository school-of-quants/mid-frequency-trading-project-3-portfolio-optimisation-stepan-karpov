import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

project_path = Path(__file__).parent.parent

TIINGO_API_KEY = "<API-KEY>"
DELAY_SECS = 0.1


def get_sp500_tickers(csv_path: str, start_date: str) -> list[str]:
    """Возвращает отсортированный список тикеров, входивших в S&P 500 начиная с start_date."""
    historical = pd.read_csv(csv_path, index_col=0)
    historical.index = pd.to_datetime(historical.index)
    historical = historical[historical.index >= start_date]

    tickers: set[str] = set()
    for row in historical["tickers"]:
        for t in row.split(","):
            tickers.add(t.strip())

    for old, new in [("BF.B", "BF-B"), ("BRK.B", "BRK-B")]:
        if old in tickers:
            tickers.discard(old)
            tickers.add(new)

    return sorted(tickers)


def fetch_ticker(
    ticker: str,
    start: str,
    end: str,
    session: requests.Session,
) -> pd.DataFrame | None:
    """Скачивает скорректированные дневные OHLCV для одного тикера с Tiingo."""
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": start, "endDate": end, "format": "json"}

    try:
        resp = session.get(url, params=params, timeout=15)
    except requests.exceptions.Timeout:
        logger.warning(f"[timeout] {ticker}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"[conn err] {ticker}")
        return None

    if resp.status_code != 200:
        logger.warning(f"[HTTP {resp.status_code}] {ticker}: {resp.text[:120]}")
        return None

    data = resp.json()
    if not data:
        logger.warning(f"[empty] {ticker}: нет данных за {start}–{end}")
        return None

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date")
    df.index.name = "Date"
    df = df[["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]].rename(columns={
        "adjOpen": "Open", "adjHigh": "High", "adjLow": "Low",
        "adjClose": "Close", "adjVolume": "Volume",
    })
    return df.astype(float)


def get_raw_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Скачивает OHLCV данные с Tiingo для всех исторических компонент S&P 500."""
    csv_path = project_path / "data" / "pony" / "S&P_500_Historical_Components.csv"
    tickers = get_sp500_tickers(str(csv_path), start_date)
    logger.info(f"[get_raw_data] тикеров для скачивания: {len(tickers)}")

    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Authorization": f"Token {TIINGO_API_KEY}",
    })

    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for ticker in tqdm(tickers, desc="Скачивание OHLCV с Tiingo"):
        df = fetch_ticker(ticker, start_date, end_date, session)
        if df is not None:
            results[ticker] = df
        else:
            failed.append(ticker)
        time.sleep(DELAY_SECS)

    logger.info(f"[get_raw_data] успешно: {len(results)}, не найдено: {len(failed)}")
    if failed:
        logger.warning(f"[get_raw_data] failed tickers: {failed[:30]}")

    price_types = ["Open", "High", "Low", "Close", "Volume"]
    wide = pd.concat(
        {pt: pd.DataFrame({t: results[t][pt] for t in results}) for pt in price_types},
        axis=1,
    )
    wide.columns.names = ["Price", "Ticker"]
    wide = wide.sort_index(axis=1)
    wide.index = pd.to_datetime(wide.index)
    return wide


def get_data() -> None:
    """Скачивает сырые данные, сохраняет в data/raw и фильтрует в data/processed."""
    from equity_project.src.utils import load_config

    cfg = load_config(str(project_path.parent / "config.yaml"))
    start_date = cfg["train_start_date"]
    end_date = cfg["backtest_end_date"]

    logger.info("[get_data] скачивание данных с Tiingo")
    raw_out = project_path / "data" / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)

    wide = get_raw_data(start_date, end_date)
    wide.to_csv(raw_out / "all_data.csv")
    logger.info(f"[get_data] сохранено: {raw_out / 'all_data.csv'}  {wide.shape}")

    # Фильтрация: оставляем только тикеры без единого пропуска в Close
    close = wide["Close"]
    complete_tickers = close.columns[close.notna().all()].tolist()
    logger.info(f"[get_data] тикеров до фильтрации: {close.shape[1]}, после: {len(complete_tickers)}")

    all_data_filtered = wide.loc[:, (slice(None), complete_tickers)]

    processed_out = project_path / "data" / "processed"
    processed_out.mkdir(parents=True, exist_ok=True)
    all_data_filtered.to_csv(processed_out / "all_data.csv")
    logger.info(f"[get_data] сохранено: {processed_out / 'all_data.csv'}  {all_data_filtered.shape}")


if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    get_data()
