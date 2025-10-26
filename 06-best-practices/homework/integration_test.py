#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import subprocess
from datetime import datetime

# --- Configuration ---
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
options = {'client_kwargs': {'endpoint_url': S3_ENDPOINT_URL}}

BUCKET_INPUT = 'nyc-duration'
BUCKET_OUTPUT = 'nyc-duration-output'
YEAR, MONTH = 2023, 1
input_file = f's3://{BUCKET_INPUT}/in/{YEAR:04d}-{MONTH:02d}.parquet'
output_file = f's3://{BUCKET_OUTPUT}/predictions_{YEAR:04d}-{MONTH:02d}.parquet'

# --- Step 1: Create dummy data ---
data = [
    ('1', '6', datetime(2023, 1, 1, 1, 0), datetime(2023, 1, 1, 1, 8)),
    ('1', '6', datetime(2023, 1, 1, 1, 10), datetime(2023, 1, 1, 1, 15)),
    ('1', '6', datetime(2023, 1, 1, 1, 20), datetime(2023, 1, 1, 1, 25)),
    ('1', '6', datetime(2023, 1, 1, 1, 30), datetime(2023, 1, 1, 1, 32)),
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

print(f"🪣 Writing test data to LocalStack S3: {input_file}")
df_input.to_parquet(input_file, engine='pyarrow', index=False, storage_options=options)
print("✅ Test data saved successfully.\n")

# --- Step 2: Run the batch script ---
print("▶️ Running batch.py for January 2023 ...")
subprocess.run(['python', 'batch.py', str(YEAR), str(MONTH)], check=True)
print("\n✅ Batch script executed successfully.\n")

# --- Step 3: Read output from LocalStack ---
print(f"📥 Reading prediction results from {output_file}")
df_result = pd.read_parquet(output_file, storage_options=options)

# --- Step 4: Validate ---
sum_pred = df_result['predicted_duration'].sum()
print(f"🎯 Sum of predicted durations: {sum_pred:.2f}")
