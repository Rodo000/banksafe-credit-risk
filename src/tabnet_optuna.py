import os
import json
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent     
DATA_DIR    = ROOT / "data" / "processed"
STUDY_DIR   = ROOT / "optuna_studies"
CKPT_DIR    = ROOT / "tabnet_ckpt"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
STUDY_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

BEST_PARAMS_JSON  = MODELS_DIR / "deep_params.json"
BEST_METRICS_JSON = MODELS_DIR / "deep_metrics.json"
BEST_MODEL_FILE   = MODELS_DIR / "deep.pt"

from src.deep_tabular import load_data, SEED          

import numpy as np
import optuna
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score

print("🔹 Loading data ...")
X_tr, X_te, y_tr, y_te, cat_meta, encoder = load_data('features', sample_frac=None)
print(f"   Train: {X_tr.shape}, Test: {X_te.shape}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {DEVICE}")

base_params = {
    "cat_idxs"   : cat_meta["cat_idxs"],
    "cat_dims"   : cat_meta["cat_dims"],
    "device_name": DEVICE,
    "seed"       : SEED,
}

def objective(trial: optuna.Trial) -> float:
    params = {
        **base_params,
        "n_d"        : trial.suggest_categorical("n_d",     [16, 32, 64, 128]),
        "n_a"        : trial.suggest_categorical("n_a",     [16, 32, 64, 128]),
        "n_steps"    : trial.suggest_int        ("n_steps",  3, 10),
        "gamma"      : trial.suggest_float      ("gamma",    1.0, 2.0),
        "cat_emb_dim": trial.suggest_int        ("cat_emb_dim", 4, 16),
        "verbose"    : 0,
    }

    clf = TabNetClassifier(**params)
    clf.fit(
        X_tr, y_tr.astype("int64"),
        eval_set=[(X_te, y_te.astype("int64"))],
        eval_metric=["auc"],
        max_epochs=50,
        patience=15,
        batch_size=1024 if DEVICE == "cpu" else 4096,
    )

    # save checkpoint for this trial
    ckpt_path = CKPT_DIR / f"trial_{trial.number}.zip"
    clf.save_model(ckpt_path.as_posix())

    preds  = clf.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, preds)
    return auc

print("🔹 Creating/loading Optuna study ...")
study_path   = STUDY_DIR / "tabnet.db"
storage_uri  = f"sqlite:///{study_path}"
study = optuna.create_study(
    study_name     = "tabnet_banksafe",
    direction      = "maximize",
    storage        = storage_uri,
    load_if_exists = True,
    sampler        = optuna.samplers.TPESampler(seed=SEED),
    pruner         = optuna.pruners.MedianPruner(n_warmup_steps=5),
)

N_TRIALS = 40
print(f"🔹 Optimizing for {N_TRIALS} trials (existing: {len(study.trials)}) ...")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n🏆  Best AUC : {study.best_value:.6f}")
print(f"🏆  Best params:\n{json.dumps(study.best_params, indent=2)}")

print("\n🔹 Retraining best model on full data ...")
best_clf = TabNetClassifier(**{**base_params, **study.best_params, "verbose":0})
best_clf.fit(
    X_tr, y_tr.astype("int64"),
    eval_set=[(X_te, y_te.astype("int64"))],
    eval_metric=["auc"],
    max_epochs=50,
    patience=15,
    batch_size=1024 if DEVICE == "cpu" else 4096,
)
best_clf.save_model(BEST_MODEL_FILE.as_posix())

best_preds = best_clf.predict_proba(X_te)[:, 1]
best_auc   = roc_auc_score(y_te, best_preds)

with BEST_PARAMS_JSON.open("w") as fh:
    json.dump(study.best_params, fh, indent=2)

with BEST_METRICS_JSON.open("w") as fh:
    json.dump({"auc": best_auc}, fh, indent=2)

print("\n✅  All artefacts saved:")
print(f"    • {BEST_MODEL_FILE}")
print(f"    • {BEST_PARAMS_JSON}")
print(f"    • {BEST_METRICS_JSON}")
print("    • individual trial checkpoints in", CKPT_DIR)
print("\nDone.")
