from __future__ import annotations
import argparse, importlib, json, joblib, pathlib, optuna
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline

SEED = 42
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.INFO)

# argument parsing
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast Optuna tuning for LR & LGBM with one-time preprocessing"
    )
    p.add_argument('--features', default='features',
                   help='Feature module inside src/ (no .py)')
    p.add_argument('--model', choices=['lr', 'lgbm'], default='lr',
                   help='Which classifier to tune & train')
    p.add_argument('--trials', type=int, default=100,
                   help='Number of Optuna trials')
    p.add_argument('--sample', type=int, default=200_000,
                   help='Rows to subsample for tuning (0 = use full data)')
    return p.parse_args()

# data loading and prep
def load_data_and_preprocessor(mod_name: str):
    module = importlib.import_module(f'src.{mod_name}')
    return module.get_data_and_preprocessor()

def prepare_data(feature_mod: str, sample_rows: int):
    """
    1. Loads data & unfitted ColumnTransformer.
    2. Fits transformer **once** on full training data.
    3. Optionally samples rows (stratified) for Optuna tuning.
    """
    pre, X_train_df, X_test_df, y_train, y_test = load_data_and_preprocessor(feature_mod)

    # Fit transformer and transform
    pre.fit(X_train_df, y_train)
    X_train = pre.transform(X_train_df)
    X_test  = pre.transform(X_test_df)

    # Stratified subsample for tuning, if needed
    if sample_rows and sample_rows < X_train.shape[0]:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=sample_rows,
            stratify=y_train,
            random_state=SEED
        )

    return pre, X_train, X_test, y_train, y_test, X_train_df, X_test_df

# tuning helpers
def tune_lr(X, y, n_trials: int):
    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        w_pos = trial.suggest_float("w_pos", 1.0, 10.0)
        clf = LogisticRegression(
            solver='saga', penalty='l2',
            C=C, class_weight={0: 1, 1: w_pos},
            max_iter=1000, n_jobs=-1, random_state=SEED
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        return cross_val_score(clf, X, y,
                               cv=cv, scoring='roc_auc', n_jobs=-1).mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def tune_lgbm(X, y, n_trials: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            'num_leaves'       : trial.suggest_int('num_leaves', 31, 127),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators'     : trial.suggest_int('n_estimators', 200, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'subsample_freq'  : trial.suggest_int('subsample_freq', 1, 10),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 0.0, 5.0),
            'scale_pos_weight' : trial.suggest_float('scale_pos_weight', 1.0, 20.0),
            'max_depth'       : trial.suggest_int('max_depth', 3, 12),
            'min_split_gain'  : trial.suggest_float('min_split_gain', 0, 1),
        }
        clf = LGBMClassifier(
            **params,
            objective='binary',
            random_state=SEED,
            n_jobs=-1,
            verbose = 1,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return cross_val_score(clf, X, y,
                               cv=cv, scoring='roc_auc', n_jobs=-1).mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

# evaluation & saving
def evaluate_and_save(pipe: Pipeline, X_test, y_test, model_name: str):
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    ks = (tpr - fpr).max()

    metrics = {
        'auc'   : roc_auc_score(y_test, proba),
        'pr_auc': average_precision_score(y_test, proba),
        'brier' : brier_score_loss(y_test, proba),
        'ks'    : ks,
        'cm'    : confusion_matrix(y_test, preds).tolist()
    }

    print(f"\n=== {model_name.upper()} validation metrics ===")
    print(json.dumps(metrics, indent=2))

    pathlib.Path('models').mkdir(exist_ok=True)
    joblib.dump(pipe, f"models/model_{model_name}.joblib", compress=3)
    with open(f"models/metrics_{model_name}.json", 'w') as fp:
        json.dump(metrics, fp, indent=2)

# main
def main() -> None:
    args = parse_args()

    # preprocessing
    (pre, X_train, X_test, y_train, y_test,
    X_train_df, X_test_df) = prepare_data(args.features, args.sample)

    # tuning
    if args.model == 'lr':
        print("🔍  Tuning Logistic Regression …")
        best_params = tune_lr(X_train, y_train, args.trials)
        print("→ best LR params:", best_params)

        C = best_params['C']
        w_pos = best_params['w_pos']
        clf = LogisticRegression(
            C=C, class_weight={0: 1, 1: w_pos},
            solver='saga', penalty='l2',
            max_iter=1000, n_jobs=-1, random_state=SEED
        )
    else:
        print("🔍  Tuning LightGBM …")
        best_params = tune_lgbm(X_train, y_train, args.trials)
        print("→ best LGBM params:", best_params)
        clf = LGBMClassifier(
            **best_params,
            objective='binary',
            random_state=SEED,
            n_jobs=-1
        )

    # final fit on *all* (possibly sampled) training data
    pipe = Pipeline([('prep', pre), ('clf', clf)])
    pipe.fit(X_train_df, y_train)

    # evaluation & persistence
    evaluate_and_save(pipe, X_test_df, y_test, args.model)

if __name__ == "__main__":
    main()