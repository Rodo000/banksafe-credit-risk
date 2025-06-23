# alternative feature-pre helper, keeps *grade* variable for benchmarking

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# hyperparameters
DUCKDB_PATH: Path = Path('data/processed/loans.duckdb')
TABLE_NAME: str = 'loans'
HIGH_NA_THRESHOLD: float = 0.50
TRAIN_END_YEAR: float = 2017
TEST_YEAR: float = 2018
GRADE_ORDER: List[str] = list('ABCDEFG')

# helper functions
def _drop_high_na(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    na_ratio = df.isna().mean()
    return df.drop(columns=na_ratio[na_ratio>=threshold].index)

def load_df(path: Path = DUCKDB_PATH, table: str = TABLE_NAME) -> pd.DataFrame:
    # read table
    con = duckdb.connect(str(path))
    df = con.execute(f'SELECT * FROM {table}').df()
    con.close()

    # binary target
    df['target'] = (df['loan_status'] == 'Charged Off').astype(int)

    # sparcity cleanup
    df = _drop_high_na(df, HIGH_NA_THRESHOLD)

    # ordinal grade mapping
    if 'grade' in df.columns:
        df['grade_int'] = df['grade'].map({g: i for i, g in enumerate(GRADE_ORDER)}).astype('Int64')
    
    return df

def temporal_split(df: pd.DataFrame,
                   train_end: int = TRAIN_END_YEAR,
                   test_year: int = TEST_YEAR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    df['issue_year'] = df['issue_d'].str[-4:].astype(int)

    train_mask = df['issue_year'] <= train_end
    test_mask = df['issue_year'] == test_year

    X_train = df.loc[train_mask].drop(columns=['target','loan_status','issue_d'])
    X_test = df.loc[test_mask].drop(columns=['target','loan_status','issue_d'])

    y_train = df.loc[train_mask, 'target']
    y_test = df.loc[test_mask, 'target']

    return X_train, X_test, y_train, y_test

def build_processor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ]), cat_cols)
    ], remainder='drop', verbose_feature_names_out=False)

def get_data_and_preprocessor(
    duckdb_path: Path = DUCKDB_PATH
) -> Tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_df(duckdb_path)
    X_train, X_test, y_train, y_test = temporal_split(df)
    preprocessor = build_processor(X_train)
    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    pre, X_tr, X_te, y_tr, y_te = get_data_and_preprocessor()
    print(f'Train: {X_tr.shape}, Test: {X_te.shape}')
    print('Features after prep:', pre.fit(X_tr).get_feature_names_out().shape[0])
