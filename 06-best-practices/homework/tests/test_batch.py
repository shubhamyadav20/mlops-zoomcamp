from datetime import datetime
import pandas as pd
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),         # 9 min ✅
        (1, 1, dt(1, 2), dt(1, 10)),               # 8 min ✅
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),      # <1 min ❌
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),          # >60 min ❌
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = prepare_data(df, categorical)

    # ✅ Expect only 2 valid rows
    assert len(actual_df) == 2
