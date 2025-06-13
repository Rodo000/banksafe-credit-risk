import duckdb
import pathlib
import subprocess
import shutil
import gzip
import pandas as pd

# create paths
RAW = pathlib.Path('data/raw')
PROC = pathlib.Path('data/processed')

# check if folders exist    
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# file names
gz_file = RAW / 'accepted_2007_to_2018Q4.csv.gz'
csv_file = RAW / 'accepted_2007_to_2018Q4.csv'
parquet_file = PROC / 'accepted_500k.parquet'
db_file = PROC / 'loans.duckdb'

if not gz_file.exists() and not csv_file.exists():
    print('Downloading LendingClub file...')
    subprocess.run(
        [
            'kaggle', 'datasets', 'download',
            '-d', 'wordsforthewise/lending-club',
            '-f', gz_file.name,
            '-p', str(RAW)
        ],
        check=True
    )

if gz_file.exists() and not csv_file.exists():
    print('Unzipping gz file...')
    with gzip.open(gz_file, 'rb') as f_in:
        with open(csv_file, 'wb') as f_out:
            shutil.copyfileobj(f_in,f_out)

print('Random-sampling 500k rows -> Parquet')
SAMPLE_SIZE = 500_000
sample_df = duckdb.query(f"""
    SELECT *
    FROM read_csv_auto('{csv_file}', AUTO_DETECT=TRUE, all_varchar=TRUE)
    USING SAMPLE {SAMPLE_SIZE} 
                         """).df()
print("Sample shape:", sample_df.shape)
sample_df.to_parquet(parquet_file, index=False)
csv_file.unlink()

print('Writing DuckDB table...')
con = duckdb.connect(db_file)
con.execute(
    'CREATE or REPLACE TABLE loans AS SELECT * FROM parquet_scan(?)',
    [str(parquet_file)]
)
con.close()
print(f'Done! Parquet: {parquet_file} | DB: {db_file}')