from __future__ import annotations
import argparse, importlib, json, joblib, pathlib, yaml
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def _ordinal_encode(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], OrdinalEncoder]:
    # replace one-hot for ordinal enconding, return tabnet metadata
    cat_cols = X_train.select_dtypes(['object','category']).columns.tolist()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train.loc[:, cat_cols] = enc.fit_transform(X_train[cat_cols])
    X_test.loc[:, cat_cols] = enc.transform(X_test[cat_cols])

    cat_idxs = [X_train.columns.get_loc(c) for c in cat_cols]
    cat_dims = [int(X_train[c].nunique()) for c in cat_cols]

    X_train_arr = X_train.astype('float32').to_numpy()
    X_test_arr  = X_test.astype('float32').to_numpy()

    X_train_arr[X_train_arr < 0] = 0
    X_test_arr[X_test_arr   < 0] = 0

    return X_train_arr, X_test_arr, {'cat_idxs': cat_idxs, 'cat_dims': cat_dims}, enc

def load_data(module:str, sample_frac: float | None = None) -> Tuple:
    mod = importlib.import_module(f'src.{module}')
    _, X_tr_df, X_te_df, y_tr, y_te = mod.get_data_and_preprocessor()

    if sample_frac:
        X_tr_df, _, y_tr, _ = train_test_split(
            X_tr_df,
            y_tr,
            train_size=sample_frac,
            stratify=y_tr,
            random_state=SEED
        )

    X_tr_arr, X_te_arr, cat_meta, enc = _ordinal_encode(X_tr_df, X_te_df)

    return X_tr_arr, X_te_arr, y_tr.values, y_te.values, cat_meta, enc

# training rutine
def train(cfg: Dict[str, Any]) -> None:
    print("🛠  Effective config\n", yaml.dump(cfg, sort_keys=False))
    # paths and device
    fit_batch_size = cfg["tabnet"].pop("batch_size", 16384)
    out_dir = pathlib.Path(cfg.get("out_dir", "models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg["tabnet"].get("device_name")
    if device == "auto" or device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["tabnet"]["device_name"] = device

    # data
    X_tr, X_te, y_tr, y_te, cat_meta, encoder = load_data(
        cfg["data"]["module"],
        cfg["data"].get("sample_frac")
    )

    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    sample_w = np.where(y_tr == 1, pos_weight, 1.0)

    # model
    (out_dir / 'checkpoints').mkdir(exist_ok=True)
    tabnet_params = {
        "n_d": 32,
        "n_a": 32,
        "n_steps": 5,
        "gamma": 1.3,
        "device_name": device,
        "seed": SEED,
        "verbose": 1,
        "cat_idxs": cat_meta["cat_idxs"],
        "cat_dims": cat_meta["cat_dims"],
        "cat_emb_dim": 8,
        "scheduler_params": {"step_size": 10, "gamma": 0.9},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    }
    # allow overrides from YAML
    tabnet_params.update(cfg.get("tabnet", {}))
    
    clf = TabNetClassifier(**tabnet_params)

    clf.fit(
        X_tr,
        y_tr.astype('int64'),
        eval_set=[(X_te, y_te.astype('int64'))],
        eval_metric=["auc"],
        max_epochs=cfg.get("max_epochs", 50),
        patience=cfg.get("patience", 15),
        weights=sample_w,
        batch_size=fit_batch_size,
    )

    clf.save_model(str(out_dir / 'checkpoints' / 'tabnet_best'))

    # eval & saving
    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "brier": brier_score_loss(y_te, proba),
        "ks": float(
            np.max(
                np.abs(
                    pd.Series(proba[y_te == 1]).rank(pct=True)
                    - pd.Series(proba[y_te == 0]).rank(pct=True)
                )
            )
        ),
        "cm": confusion_matrix(y_te, preds).tolist(),
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump(
        {"encoder": encoder, "tabnet": clf},
        out_dir / f"tabnet_{cfg['data']['module']}.joblib",
        compress=3,
    )
    with (out_dir / f"metrics_tabnet_{cfg['data']['module']}.json").open("w") as fp:
        json.dump(metrics, fp, indent=2)


# CLI wrapper 
def _cli() -> None:
    parser = argparse.ArgumentParser(description="TabNet trainer (local & Colab)")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="Path to YAML config. If omitted, sensible defaults are used.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Inline overrides like key=value (e.g. tabnet.batch_size=8192)",
    )
    args = parser.parse_args()

    # defaults that work anywhere
    cfg: Dict[str, Any] = {
        "data": {"module": "features", "sample_frac": None},
        "tabnet": {"device_name": "auto"},
        "out_dir": "models",
        "max_epochs": 50,
        "patience": 15,
    }

    if args.config:
        cfg.update(yaml.safe_load(args.config.read_text()))

    for kv in args.override:
        k, v = kv.split("=", 1)
        section, key = k.split(".", 1)
        # naive type cast
        v_cast: Any = yaml.safe_load(v)
        cfg.setdefault(section, {})[key] = v_cast

    train(cfg)

if __name__ == "__main__":
    _cli()