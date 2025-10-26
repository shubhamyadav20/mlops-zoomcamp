import pandas as pd
import os
from datetime import datetime

# --- Configuration ---
# LocalStack typically runs on http://localhost:4566
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')

# Target file details for January 2023
YEAR = 2023
MONTH = 1
BUCKET = 'nyc-duration'
KEY = f'in/{YEAR:04d}-{MONTH:02d}.parquet'
input_file = f's3://{BUCKET}/{KEY}'

# Storage options pointing to LocalStack
options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}
# --- End Configuration ---

# 1. Create the dummy DataFrame (similar to the unit test in Q3)
# Note: These columns are minimal but sufficient for the integration test
data = [
    ('1', '6', datetime(2023, 1, 1, 1, 0), datetime(2023, 1, 1, 1, 8)),
    ('1', '6', datetime(2023, 1, 1, 1, 10), datetime(2023, 1, 1, 1, 15)),
    ('1', '6', datetime(2023, 1, 1, 1, 20), datetime(2023, 1, 1, 1, 25)),
    ('1', '6', datetime(2023, 1, 1, 1, 30), datetime(2023, 1, 1, 1, 32)),
]

columns = [
    'PULocationID', 
    'DOLocationID', 
    'tpep_pickup_datetime', 
    'tpep_dropoff_datetime'
]

df_input = pd.DataFrame(data, columns=columns)

# 2. Save the DataFrame to LocalStack S3
print(f"Attempting to save test data to LocalStack S3: {input_file}")

try:
    # Ensure the bucket exists (optional, but good practice for robustness)
    # This requires a separate AWS CLI or awslocal command, but we'll proceed 
    # assuming LocalStack is configured to auto-create, or a separate mb command was run.
    
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )
    print("Test data successfully saved.")

except Exception as e:
    print(f"Error saving file to LocalStack S3: {e}")
    print("Ensure LocalStack is running and the bucket exists.")