import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def preprocess_dataframe(df):
    """
    Prepares the continuous factory process data for modeling.

    Steps:
    1. Select the primary goal relevant columns.
    2. Remove setpoint value.
    3. Convert timestamp to datetime.
    4. Set timestamp as index.

    Args:
      df: A Pandas DataFrame containing the raw data.

    Returns:
      A Pandas DataFrame containing the prepared data.
    """

    selected_columns = df.columns[:71]
    df = df[selected_columns]

    df = df.loc[:, ~df.columns.str.contains("Setpoint")]

    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.set_index("time_stamp")

    return df


df = pd.read_csv("../../data/raw/continuous_factory_process.csv")

# Prepare the data
processed_df = preprocess_dataframe(df)

processed_df.to_pickle("../../data/interim/data.processed.pkl")
