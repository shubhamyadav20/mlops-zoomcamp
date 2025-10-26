#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os

# --- Helpers ---

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def read_data(filename, categorical):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    storage_options = {}
    if filename.startswith("s3://"):
        storage_options = {'client_kwargs': {'endpoint_url': S3_ENDPOINT_URL}}
    print(f"📥 Reading data from {filename}")
    df = pd.read_parquet(filename, storage_options=storage_options)
    return prepare_data(df, categorical)


def save_data(df, filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    storage_options = {}
    if filename.startswith("s3://"):
        storage_options = {'client_kwargs': {'endpoint_url': S3_ENDPOINT_URL}}
    df.to_parquet(filename, engine='pyarrow', index=False, storage_options=storage_options)
    print(f"💾 Saved results to {filename}")


def get_input_path(year, month):
    return f's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'


def get_output_path(year, month):
    return f's3://nyc-duration-output/predictions_{year:04d}-{month:02d}.parquet'


# --- Main ---

def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    # Load model
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    # Read data and prepare
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)

    # Predict
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f"✅ Predicted mean duration: {y_pred.mean():.2f}")

    # Save results
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})
    save_data(df_result, output_file)
    print("🎉 Batch job completed successfully.")


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
