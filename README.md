# Прогнозирование оттока клиентов телеком-компании (Telco Customer Churn)

Портфолио-проект: предсказание оттока клиентов на основе датасета IBM Telco Customer
Churn. Полный цикл: EDA, инжиниринг признаков, бейзлайн + градиентный бустинг, борьба
с несбалансированностью классов, подбор гиперпараметров (Optuna), интерпретируемость
(SHAP), трекинг экспериментов (MLflow).

## Стек технологий

- Python 3.13 (pip + venv)
- pandas, scikit-learn, CatBoost, XGBoost, LightGBM
- Optuna, SHAP, MLflow, imbalanced-learn
- kagglehub для автоматической загрузки датасета

## Структура проекта

```
├── data/
│   ├── raw/            # исходный CSV (скачивается автоматически, не в git)
│   └── processed/      # очищенные и обогащённые данные
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── data_loader.py  # загрузка датасета через kagglehub
│   ├── features.py     # очистка, фичи, train/test split
│   └── train.py        # обучение моделей, MLflow, Optuna
├── models/              # сохранённая лучшая модель (не в git)
├── requirements.txt
└── README.md
```

## Установка (Windows PowerShell)

```powershell
# 1. Создать виртуальное окружение
python -m venv venv

# 2. Активировать его
.\venv\Scripts\Activate.ps1

# 3. Установить зависимости
pip install -r requirements.txt
```

> Примечание: для загрузки датасета используется Kaggle API через kagglehub.
> Предполагается, что файл с учётными данными Kaggle уже существует по пути
> `%USERPROFILE%\.kaggle\kaggle.json`. Если файла нет — создайте API-токен на
> https://www.kaggle.com/settings и сохраните его по этому пути.

## Порядок запуска

1. `python src/data_loader.py` — скачивает датасет и кладёт CSV в `data/raw/`.
2. `notebooks/01_eda.ipynb` — разведочный анализ данных.
3. `notebooks/02_feature_engineering.ipynb` — очистка и инжиниринг признаков.
4. `notebooks/03_modeling.ipynb` — обучение моделей, подбор гиперпараметров, MLflow.
5. `notebooks/04_explainability.ipynb` — SHAP, бизнес-интерпретация ошибок модели.

Чтобы посмотреть результаты экспериментов:

```powershell
mlflow ui
```

и открыть http://localhost:5000

## Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD | TBD |
| LightGBM (tuned) | TBD | TBD | TBD | TBD | TBD | TBD |

## Бизнес-интерпретация ошибок

TODO: заполняется по итогам notebook 04 — стоимость False Negative (потерянный
клиент, retention-предложение не сделано) vs. False Positive (скидка предложена
клиенту, который не собирался уходить), выбор оптимального порога.

## Известные ограничения и риски окружения

Python 3.13 — относительно новая версия. На момент написания все ключевые
библиотеки (`catboost>=1.2.8`, `lightgbm`, `shap>=0.48`) имеют готовые wheel-сборки
под Windows/cp313. Если в будущем установка какого-то пакета не найдёт wheel и
попытается собираться из исходников — резервный план:

1. Создать venv на Python 3.12 (если установлен): `py -3.12 -m venv venv`
2. Либо понизить версию проблемного пакета до последней, у которой есть готовый wheel.

## Выводы

TODO: заполнить после завершения анализа.
