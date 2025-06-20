import importlib, json, joblib, pathlib
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)
import torch
import pandas as pd

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_data(module):
    mod = importlib.import_module(f'src.{module}')
    pre, X_tr, X_te, y_tr, y_te = mod.get_data_and_preprocessor()
    X_tr_enc = pre.fit_transform(X_tr).toarray().astype(np.float32)
    X_te_enc = pre.transform(X_te).toarray().astype(np.float32)
    return X_tr_enc, X_te_enc, y_tr.values, y_te.values, pre

def main(features='features', epochs=30):
    X_tr, X_te, y_tr, y_te, pre = load_data(features)
    clf = TabNetClassifier(n_d=16, n_a=16,
                           n_steps=5, n_independent=2,
                           n_shared=2, seed=SEED,
                           verbose=1)
    pos_weight = (len(y_tr)-y_tr.sum()) / y_tr.sum()
    sample_w = np.where(y_tr==1, pos_weight, 1.0)

    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
            eval_metric=['auc'], max_epochs=epochs,
            patience=10, weights=sample_w)
    
    proba = clf.predict_proba(X_te)[:,1]
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

    pathlib.Path('models').mkdir(exist_ok=True)
    joblib.dump({'pre': pre, 'tabnet': clf},
                f'models/tabnet_{features}.joblib', compress=3)
    with open(f'models/metrics_tabnet_{features}.json','w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    import fire; fire.Fire(main)