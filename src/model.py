import argparse, importlib, json, joblib, pathlib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix,
                             roc_curve)
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
import optuna
import pandas as pd
import numpy as np

SEED = 42
np.random.seed(SEED)

def parse_args():
    p = argparse.ArgumentParser(description='train/tune Lending-Club model')
    p.add_argument('--features', default='features',
                   help='Feature module inside src/ (no .py)')
    p.add_argument('--model', choices=['lr','lgbm'], default='lgbm')
    p.add_argument('--study-name', default='lgbm_pd')
    return p.parse_args()

def load_data(feature_mod_name):
    feat_mod = importlib.import_module(f'src.{feature_mod_name}')
    return feat_mod.get_data_and_preprocessor()

def param_grid_from_trial(trial: optuna.Trial) -> dict:
    return {
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
    }

def tune_lgbm(X_tr, y_tr, X_val, y_val, pre, study_name):
    def objective(trial):
        params = param_grid_from_trial(trial)

        prep = clone(pre).fit(X_tr, y_tr)
        X_tr_t  = prep.transform(X_tr)
        X_val_t = prep.transform(X_val)

        clf = LGBMClassifier(
            **params,
            objective='binary',
            class_weight='balanced',
            random_state=SEED
        )
        es_cb = early_stopping(100, first_metric_only=True, verbose=False)
        clf.fit(
            X_tr_t, y_tr,
            eval_set=[(X_val_t, y_val)],
            eval_metric='auc',
            callbacks=[es_cb]
        )
        proba = clf.predict_proba(X_val_t)[:, 1]
        return roc_auc_score(y_val, proba)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='maximize',
                                study_name=study_name,
                                sampler=sampler)
    study.optimize(objective, n_trials=40, show_progress_bar=True)
    return study.best_params            


def main():
    args = parse_args()
    pre, X_tr, X_val, y_tr, y_val = load_data(args.features)

    # choose model
    if args.model == 'lr':
        clf = LogisticRegression(solver='saga',
                                 penalty='l2',
                                 max_iter=1000,
                                 c
                                 n_jobs=-1,
                                 class_weight='balanced',
                                 verbose=1,
                                 random_state=SEED)
    else:
        best = tune_lgbm(X_tr, y_tr, X_val, y_val, pre, args.study_name)
        clf = LGBMClassifier(**best,
                             objective='binary',
                             class_weight='balanced',
                             random_state=SEED)
        
    X_full = pd.concat([X_tr, X_val])
    y_full = pd.concat([y_tr, y_val])
    
    pipe = Pipeline([('prep', pre),('clf', clf)])
    pipe.fit(X_full, y_full)

    # metrics
    proba = pipe.predict_proba(X_val)[:,1]
    preds = (proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_val, proba)
    ks_stat  = (tpr - fpr).max()

    metrics = {
        'auc': roc_auc_score(y_val, proba),
        'pr_auc': average_precision_score(y_val, proba),
        'brier': brier_score_loss(y_val, proba),
        'ks': ks_stat,
        'cm': confusion_matrix(y_val, preds).tolist() 
    }
    print(json.dumps(metrics, indent=2))

    # save metrics
    pathlib.Path('models').mkdir(exist_ok=True)
    joblib.dump(pipe, f'models/model_{args.model}.joblib', compress=3)
    with open (f'models/metrics_{args.model}.json','w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()