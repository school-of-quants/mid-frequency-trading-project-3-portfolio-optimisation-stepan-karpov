# ML-driven Long-Only Equity Strategy

## Как читать этот проект

Всё исследование — анализ данных, объяснение идей, визуализации и интерпретация результатов — находится в **[Strategy.ipynb](Strategy.ipynb)**.

Код в репозитории (`equity_project/src/`) — это перенос того же кода из ноутбука в модульную форму. Если хочешь понять, что и зачем сделано, читай ноутбук: там каждый шаг сопровождается объяснением и графиками.

**Рекомендуемый порядок чтения:** идти по `Strategy.ipynb` сверху вниз. Разделы идут последовательно и самодостаточны.

---

## Что сделано

### Данные

- **Источник:** Tiingo API (30$ за подписку :( )— скорректированные дневные OHLCV
- **Период:** 2017–2025, ~550 тикеров
- **Train:** 2017–2021 / **Backtest:** 2022–2025

### Признаки (17 штук)

Отклонения от скользящих средних, кросс-секциональный моментум (12-1 month), волатильность доходностей, рыночный режим, краткосрочный реверсал, расстояние от 52-week high, ликвидность по Amihud, бета к рынку.

### Модели

Обучены две независимые модели на `CatBoost` с `TimeSeriesSplit + embargo`:

| Модель             | Тип                | Таргет                                | Метрика         |
| ------------------ | ------------------ | ------------------------------------- | --------------- |
| **Triple Barrier** | CatBoostClassifier | short/neutral/long (±5%, 22 дня)      | Macro F1 ≈ 0.35 |
| **Quintile**       | CatBoostRegressor  | квантиль форвардной 22-дн. доходности | IC ≈ 0.02       |

### Стратегия

Long-only, fully invested. Каждые K торговых дней портфель ребалансируется:

1. **Score** = `(P(long) − P(short)) × pred_quintile` — комбинация сигналов обеих моделей
2. **Фильтр** — берём только бумаги с `score > 0`
3. **Веса** — пропорциональны score, нормированы в 1
4. **Исполнение** — по цене `Open` следующего дня

Результаты на бэктесте (2022–2025, fees=0.1%):

| K               | CAGR  | Sharpe | Max DD |
| --------------- | ----- | ------ | ------ |
| 1 (ежедневно)   | −4.0% | −0.09  | −35.1% |
| 5 (еженедельно) | +5.5% | 0.33   | −25.4% |
| 22 (ежемесячно) | +7.5% | 0.43   | −27.2% |

Подробный разбор результатов — в конце `Strategy.ipynb`.

---

## Установка и запуск

```bash
pip install poetry==1.8.5
poetry install
```

Полный пайплайн:

```bash
poetry run python main.py
```

Отдельные этапы:

```bash
poetry run python -m equity_project.src.get_data        # скачивание данных
poetry run python -m equity_project.src.build_features  # таргеты + фичи + learning-dataset
poetry run python -m equity_project.src.train           # обучение обеих моделей
poetry run python -m equity_project.src.run_backtest    # инференс + бэктест
```

---

## Структура репозитория

```
equity_project/
├── data/
│   ├── pony/               исторический состав S&P 500
│   ├── raw/                сырые OHLCV (all_data.csv)
│   ├── processed/          all_data_labeled.csv, all_data_features_extended.csv
│   └── learning-dataset/   X/y parquet файлы для train и backtest
├── models/
│   ├── model_tb.joblib     Triple Barrier classifier
│   └── model_q.joblib      Quintile regressor
└── src/
    ├── get_data.py         скачивание и фильтрация данных
    ├── build_features.py   таргеты, фичи, сборка learning-dataset
    ├── train.py            обучение двух моделей с grid search
    ├── run_backtest.py     инференс + построение портфеля + бэктест
    └── utils.py            вспомогательные функции

Strategy.ipynb              всё исследование с объяснениями
main.py                     запуск полного пайплайна
config.yaml                 даты train/backtest, начальный капитал, комиссии
```
