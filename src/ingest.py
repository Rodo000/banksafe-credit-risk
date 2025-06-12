import duckdb
import pathlib
import subprocess
import pandas as pd

# create paths
RAW = pathlib.Path('data/raw')
PROC = pathlib.Path('data/processed')

# check if folders exist
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# file names
csv_file = RAW / 'accepted_2007_to_2018Q4.csv.gz'
parquet_file = PROC / 'accepted_100k.csv'
db_file = PROC / 'loans.duckdb'

if not csv_file.exists():
    subprocess.run(
        [
            'kaggle', 'datasets', 'download',
            '-d', 'wordsforthewise/lending-club',
            '-f', csv_file.name,
            '-p', str(RAW)
        ],
        check=True
    )

print('Sampling 100k rows -> Parquet')
df = pd.read_csv(csv_file, nrows=100000, low_memory=False)
df.to_parquet(parquet_file, index=False)

print('Writing DuckDB table...')
con = duckdb.connect(db_file)
con.execute(
    'CREATE or REPLACE TABLE loans AS SELECT * FROM parquet_scan(?)',
    [str(parquet_file)]
)
con.close()
print(f'Done! Parquet: {parquet_file}, DB: {db_file}')