from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import duckdb
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import re

# hyperparameters
DUCKDB_PATH: Path = Path("data/processed/loans_full.duckdb")
TABLE_NAME: str = 'loans'
HIGH_NA_THRESHOLD: float = 0.50
TRAIN_END_YEAR = 2017
TEST_YEAR = 2018

DROP_COLS = [  
    # IDs / free text
    'id','member_id','url','emp_title','title','desc',
    # Principal & payment snapshots
    'out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv',
    'total_rec_prncp','total_rec_int','total_rec_late_fee',
    'recoveries','collection_recovery_fee','last_pymnt_amnt',
    # Dates after origination
    'last_pymnt_d','next_pymnt_d','last_credit_pull_d','settlement_date',
    # Post-origination FICO / utilisation
    'last_fico_range_high','last_fico_range_low',
    'sec_app_fico_range_low','sec_app_fico_range_high',
    'sec_app_revol_util','revol_bal_joint',
    # Hardship / settlement
    'hardship_flag','hardship_type','hardship_reason','hardship_status',
    'hardship_start_date','hardship_end_date','hardship_length',
    'hardship_amount','hardship_dpd','hardship_loan_status',
    'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount','hardship_last_payment_amount',
    'debt_settlement_flag','debt_settlement_flag_date',
    'settlement_status','settlement_amount','settlement_percentage',
    'settlement_term',
    # Delinquency / recovery after issue
    'deferral_term','mths_since_last_major_derog',
    'sec_app_mths_since_last_major_derog',
    'mths_since_last_delinq','mths_since_last_record','mths_since_rcnt_il',
    'mths_since_recent_bc','mths_since_recent_bc_dlq',
    'mths_since_recent_inq','mths_since_recent_revol_delinq',
    # Post-issue performance aggregates
    'num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m',
    'chargeoff_within_12_mths','delinq_amnt',
    'collections_12_mths_ex_med','sec_app_collections_12_mths_ex_med'
]


# helper functions
def _clean_df(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLS, errors='ignore')
    na_ratio = df.isna().mean()
    return df.drop(columns=na_ratio[na_ratio>=threshold].index)

PERCENT_RE  = re.compile(r"%$")   # strip %
CURRENCY_RE = re.compile(r"^\$")  # strip $

def _coerce_numeric(col: pd.Series) -> pd.Series:
    """Try to convert an object column to numeric. Handles %, $, commas."""
    if col.dtype != "object":
        return col

    cleaned = (
        col
        .str.strip()
        .str.replace(",", "", regex=False)           
        .str.replace(PERCENT_RE, "", regex=True)      
        .str.replace(CURRENCY_RE, "", regex=True)     
        .replace({"": None, "n/a": None, "N/A": None})
    )
    return pd.to_numeric(cleaned, errors="ignore")   

def _coerce_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%b-%Y", errors="coerce")
    return df

def load_df(path: Path = DUCKDB_PATH, table: str = TABLE_NAME) -> pd.DataFrame:
    # read table
    con = duckdb.connect(str(path))
    df = con.execute(f"SELECT * FROM {table}").df()
    con.close()

    df = df.apply(_coerce_numeric)
    df = _coerce_dates(df, ['issue_d','earliest_cr_line'])

    # label
    df['target'] = (df['loan_status'] == 'Charged Off').astype(int)

    # sparcity cleanup
    df = _clean_df(df, HIGH_NA_THRESHOLD)

    return df

def temporal_split(
        df: pd.DataFrame,
        train_end_year: int = TRAIN_END_YEAR,
        test_year: int = TEST_YEAR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    df = df[df['issue_d'].notna()]
    df['issue_year'] = df['issue_d'].dt.year

    train_mask = df['issue_year'] <= train_end_year
    test_mask = df['issue_year'] == test_year

    X_train = df.loc[train_mask].drop(columns=['target','loan_status'])
    X_test = df.loc[test_mask].drop(columns=['target','loan_status'])

    y_train = df.loc[train_mask, 'target']
    y_test = df.loc[test_mask, 'target']

    return X_train, X_test, y_train, y_test

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    return ColumnTransformer(
        [
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
            ]), cat_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

# engineered features
def credit_age_years(df: pd.DataFrame) -> pd.Series:
    issue    = pd.to_datetime(df['issue_d'],    format='%b-%Y')
    earliest = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    return (issue - earliest).dt.days / 365.25

engineered_features = {
    'open_total_ratio': lambda df: (
        pd.to_numeric(df['open_acc'],  errors='coerce')
        / pd.to_numeric(df['total_acc'], errors='coerce')
    ).replace([np.inf,-np.inf], np.nan).fillna(0),
    'inc_bin': lambda df: (
        pd.qcut(
            pd.to_numeric(df['annual_inc'], errors='coerce'),
            5, labels=False, duplicates='drop'
        ).fillna(0)
    ),
    'amnt_inc_ratio': lambda df: (
        pd.to_numeric(df['loan_amnt'], errors='coerce')
        / pd.to_numeric(df['annual_inc'], errors='coerce')
    ).replace([np.inf,-np.inf], np.nan).fillna(0),
    'credit_age_years': credit_age_years,
}

def get_data_and_preprocessor(
    duckdb_path: Path = DUCKDB_PATH,
) -> Tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_df(duckdb_path)
    X_train, X_test, y_train, y_test = temporal_split(df)

    # add engineered features
    for feat_name, feat_fn in engineered_features.items():
        X_train[feat_name] = feat_fn(X_train)
        X_test[feat_name]  = feat_fn(X_test)

    preprocessor = build_preprocessor(X_train)
    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    pre, X_tr, X_te, y_tr, y_te = get_data_and_preprocessor()
    print(f'Train: {X_tr.shape}, Test: {X_te.shape}')
    print('Features after prep:', pre.fit(X_tr).get_feature_names_out().shape[0])

