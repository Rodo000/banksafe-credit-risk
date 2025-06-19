import argparse, importlib, json, joblib, pathlib
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import optuna
import pandas as pd
import numpy as np

SEED = 42
np.random.seed(SEED)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='features',
                   help='Which feature module to import (no .py)')
    p.add_argument('--model', choices=['lr','lgbm'], default='lr')
    p.add_argument('--study-name', default='lr_optuna')
    return p.parse_args()

def load_data(feature_mod_name):
    feat_mod = importlib.import_module(f'src.{feature_mod_name}')
    return feat_mod.get_data_and_preprocessor()

def tune_lgbm(X_tr, y_tr, pre):
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        pipe = Pipeline([('prep', pre),
                         ('clf', LGBMClassifier(**params,
                                                objective='binary',
                                                class_weight='balanced',
                                                random_state=SEED))])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict_proba(X_tr)[:,1]
        return roc_auc_score(y_tr, pred)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='maximize',
                                study_name='lgbm_pd',
                                sampler=sampler)
    study.optimize(objective, n_trials=40, show_progress_bar=True)
    return study.best_params

def main():
    args = parse_args()
    pre, X_tr, X_te, y_tr, y_te = load_data(args.features)

    # choose model
    if args.model == 'lr':
        clf = LogisticRegression(solver='saga',
                                 penalty='l2',
                                 max_iter=1000,
                                 n_jobs=-1,
                                 class_weight='balanced',
                                 verbose=1,
                                 random_state=SEED)
    else:
        best = tune_lgbm(X_tr, y_tr, pre)
        clf = LGBMClassifier(**best,
                             objective='binary',
                             class_weight='balanced',
                             random_state=42)
    
    pipe = Pipeline([('prep', pre),('clf', clf)])
    pipe.fit(X_tr, y_tr)

    # metrics
    proba = pipe.predict_proba(X_te)[:,1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        'auc': roc_auc_score(y_te, proba),
        'pr_auc': average_precision_score(y_te, proba),
        'brier': brier_score_loss(y_te, proba),
        'ks': max(abs(pd.Series(proba[y_te==1]).rank(pct=True) -
                      pd.Series(proba[y_te==0]).rank(pct=True))),
        'cm': confusion_matrix(y_te, preds).tolist() 
    }
    print(metrics)

    # save metrics
    pathlib.Path('models').mkdir(exist_ok=True)
    joblib.dump(pipe, f'models/model_{args.model}.joblib', compress=3)
    with open (f'models/metrics_{args.model}.json','w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()