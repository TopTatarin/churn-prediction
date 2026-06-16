"""Training pipeline: baseline -> gradient boosting models -> tuning -> MLflow tracking."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from src.features import build_processed_dataset, get_train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_EXPERIMENT = "telco-churn"


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def log_run(model_name, params, metrics, model, log_fn):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        log_fn(model, "model")


def train_baseline(X_train, y_train, X_test, y_test):
    params = {"class_weight": "balanced", "max_iter": 1000, "random_state": 42}
    model = LogisticRegression(**params).fit(X_train, y_train)
    metrics = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    log_run("logistic_regression", params, metrics, model, mlflow.sklearn.log_model)
    return model, metrics


def train_catboost(X_train, y_train, X_test, y_test):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {"iterations": 500, "depth": 6, "learning_rate": 0.05,
              "class_weights": [1, pos_weight], "random_seed": 42, "verbose": False}
    model = CatBoostClassifier(**params).fit(X_train, y_train)
    metrics = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    log_run("catboost", params, metrics, model, mlflow.catboost.log_model)
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
              "scale_pos_weight": pos_weight, "random_state": 42, "eval_metric": "logloss"}
    model = XGBClassifier(**params).fit(X_train, y_train)
    metrics = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    log_run("xgboost", params, metrics, model, mlflow.xgboost.log_model)
    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    params = {"n_estimators": 300, "max_depth": -1, "learning_rate": 0.05,
              "class_weight": "balanced", "random_state": 42}
    model = LGBMClassifier(**params).fit(X_train, y_train)
    metrics = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    log_run("lightgbm", params, metrics, model, mlflow.lightgbm.log_model)
    return model, metrics


def tune_with_optuna(X_train, y_train, X_test, y_test, n_trials: int = 30):
    """Optuna tuning for LightGBM. Logs each trial as a nested MLflow run."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "class_weight": "balanced",
            "random_state": 42,
        }
        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            model = LGBMClassifier(**params).fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, proba)
            mlflow.log_params(params)
            mlflow.log_metric("pr_auc", pr_auc)
        return pr_auc

    with mlflow.start_run(run_name="optuna_tuning_parent"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_pr_auc", study.best_value)

    return study.best_params, study.best_value


def save_best_model(model, model_name: str, metrics: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(model, model_path)
    metadata = {"model_name": model_name, "metrics": metrics}
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[train] Saved best model ({model_name}) to {model_path}")


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    df = build_processed_dataset(RAW_CSV)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    results = {}
    for name, fn in [
        ("logistic_regression", train_baseline),
        ("catboost", train_catboost),
        ("xgboost", train_xgboost),
        ("lightgbm", train_lightgbm),
    ]:
        model, metrics = fn(X_train, y_train, X_test, y_test)
        results[name] = (model, metrics)
        print(f"[train] {name}: {metrics}")

    best_name = max(results, key=lambda n: results[n][1]["pr_auc"])
    best_model, best_metrics = results[best_name]
    print(f"[train] Best model by PR-AUC: {best_name}")

    save_best_model(best_model, best_name, best_metrics)


if __name__ == "__main__":
    main()
