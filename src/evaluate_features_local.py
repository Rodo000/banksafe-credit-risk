import json
import joblib
import importlib
import numpy as np
import pandas as pd
import mlflow
import torch

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve
)
from pytorch_tabnet.tab_model import TabNetClassifier

from src.deep_tabular import load_data as deep_load

SEED = 42
np.random.seed(SEED)


def credit_age_years(df: pd.DataFrame) -> pd.Series:
    issue    = pd.to_datetime(df['issue_d'], format='%b-%Y')
    earliest = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    return (issue - earliest).dt.days / 365.25


engineered_features = {
    'open_total_ratio': lambda df: (
        (
            pd.to_numeric(df['open_acc'], errors='coerce')
            / pd.to_numeric(df['total_acc'], errors='coerce')
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    ),
    'inc_bin': lambda df: (
        pd.qcut(
            pd.to_numeric(df['annual_inc'], errors='coerce'),
            5,
            labels=False,
            duplicates='drop'
        ).fillna(0)
    ),
    'amnt_inc_ratio': lambda df: (
        (
            pd.to_numeric(df['loan_amnt'], errors='coerce')
            / pd.to_numeric(df['annual_inc'], errors='coerce')
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    ),
    'credit_age_years': credit_age_years,
}


def main():
    # 1) Experiment setup
    mlflow.set_experiment("banksafe-credit-risk-fe")

    # 2) Load preprocessing & data splits
    feat_mod = importlib.import_module('src.features')
    pre, X_tr, X_val, y_tr, y_val = feat_mod.get_data_and_preprocessor()

    # 3) TabNet metadata & device
    _, _, _, _, cat_meta, _ = deep_load('features', sample_frac=0.02)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_params = {
        "cat_idxs":    cat_meta["cat_idxs"],
        "cat_dims":    cat_meta["cat_dims"],
        "device_name": device,
        "seed":        SEED,
    }

    # 4) Load tuned hyperparameters
    lr_pipe   = joblib.load("models/model_lr.joblib")
    lgbm_pipe = joblib.load("models/model_lgbm.joblib")

    lr_params   = lr_pipe.named_steps['clf'].get_params()
    lgbm_params = lgbm_pipe.named_steps['clf'].get_params()
    lgbm_params['n_jobs'] = 1

    deep_params = json.load(open("models/deep_params.json"))
    max_epochs  = deep_params.pop("max_epochs", 50)

    # 5) Loop over your engineered features
    for feat_name, feat_fn in engineered_features.items():
        print(f"\n=== FEATURE: {feat_name} ===")

        # augment datasets
        Xtr = X_tr.copy()
        Xtr[feat_name] = feat_fn(Xtr)
        Xvl = X_val.copy()
        Xvl[feat_name] = feat_fn(Xvl)

        # —— 5a) Logistic Regression ——
        with mlflow.start_run(run_name=f"lr_{feat_name}"):
            pipe = Pipeline([
                ('prep', clone(pre)),
                ('clf', LogisticRegression(**lr_params))
            ])
            pipe.fit(Xtr, y_tr)
            proba = pipe.predict_proba(Xvl)[:, 1]

            fpr, tpr, _ = roc_curve(y_val, proba)
            ks = (tpr - fpr).max()
            metrics = {
                'auc':    roc_auc_score(y_val, proba),
                'pr_auc': average_precision_score(y_val, proba),
                'brier':  brier_score_loss(y_val, proba),
                'ks':     ks,
            }

            mlflow.log_params({**lr_params, 'feature': feat_name})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path=f"model_lr_{feat_name}")
            print("LR metrics:", metrics)

        # —— 5b) LightGBM ——
        with mlflow.start_run(run_name=f"lgbm_{feat_name}"):
            pipe = Pipeline([
                ('prep', clone(pre)),
                ('clf', LGBMClassifier(**lgbm_params))
            ])
            pipe.fit(Xtr, y_tr)
            proba = pipe.predict_proba(Xvl)[:, 1]

            fpr, tpr, _ = roc_curve(y_val, proba)
            ks = (tpr - fpr).max()
            metrics = {
                'auc':    roc_auc_score(y_val, proba),
                'pr_auc': average_precision_score(y_val, proba),
                'brier':  brier_score_loss(y_val, proba),
                'ks':     ks,
            }

            mlflow.log_params({**lgbm_params, 'feature': feat_name})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path=f"model_lgbm_{feat_name}")
            print("LGBM metrics:", metrics)

        # —— 5c) TabNet —— (commented out for performance limitation)
        # with mlflow.start_run(run_name=f"tabnet_{feat_name}"):
        #     pr       = clone(pre).fit(Xtr)
        #     Xtr_t    = pr.transform(Xtr)
        #     Xvl_t    = pr.transform(Xvl)

        #     params = {**base_params, **deep_params}
        #     clf    = TabNetClassifier(**params)
        #     clf.fit(
        #         Xtr_t, y_tr.astype('int64'),
        #         eval_set=[(Xvl_t, y_val.astype('int64'))],
        #         eval_metric=['auc'],
        #         max_epochs=max_epochs,
        #         patience=12,
        #     )
        #     proba = clf.predict_proba(Xvl_t)[:, 1]

        #     fpr, tpr, _ = roc_curve(y_val, proba)
        #     ks = (tpr - fpr).max()
        #     metrics = {
        #         'auc':    roc_auc_score(y_val, proba),
        #         'pr_auc': average_precision_score(y_val, proba),
        #         'brier':  brier_score_loss(y_val, proba),
        #         'ks':     ks,
        #     }

        #     mlflow.log_params({**deep_params, 'feature': feat_name})
        #     mlflow.log_metrics(metrics)
        #     clf.save_model(f"tabnet_{feat_name}.ckpt")
        #     mlflow.log_artifact(f"tabnet_{feat_name}.ckpt", artifact_path="model_tabnet")
        #     print("TabNet metrics:", metrics)


if __name__ == "__main__":
    main()
