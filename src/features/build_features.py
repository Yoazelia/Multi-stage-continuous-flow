import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_pickle("../../data/interim/data.processed.pkl")

df["Stage1.Output.Measurement0.U.Actual"].plot()


def clean_time_series_data(series, window_size=25, num_std=3):
    """Cleans up time series data by removing extreme values and filling in missing values using linear interpolation.

    Args:
        series: A Pandas Series containing the time series data.
        window_size: The size of the window to use when calculating the rolling mean and standard deviation.
        num_std: The number of standard deviations to use to identify extreme values.

    Returns:
        A Pandas Series containing the cleaned time series data.
    """

    series = series.where(series != 0)

    # Calculate the rolling mean and standard deviation
    rolling_mean = series.rolling(window_size).mean()
    rolling_std = series.rolling(window_size).std()

    # Identify extreme values
    lower_bound = rolling_mean - (num_std * rolling_std)
    upper_bound = rolling_mean + (num_std * rolling_std)

    # Replace extreme values with missing values
    series = series.where((series > lower_bound) & (series < upper_bound))

    # Fill in missing values using linear interpolation
    series = series.interpolate(method="linear")

    return series


cleaned_series = clean_time_series_data(
    df["Stage1.Output.Measurement0.U.Actual"], window_size=100
)
cleaned_series.plot()

df["Stage1.Output.Measurement0.U.Actual"] = cleaned_series
df = df.iloc[:, :42]


def features_engineer(df, window_size=60, lag_features=None):
    """Adds basic time series features to a DataFrame using window size and lag features.

    Args:
      df: A Pandas DataFrame.
      window_size: The window size to use for the rolling mean and standard deviation features.
      lag_features: A list of column names to create lag features for. If None, all columns will be used.

    Returns:
      A Pandas DataFrame with the new features added.
    """

    # Create a copy of the DataFrame
    df_eng = df.copy()

    # Skip the 'Stage1.Output.Measurement0.U.Actual' column if it is in the lag features
    if lag_features is not None:
        lag_features = [
            col for col in lag_features if col != "Stage1.Output.Measurement0.U.Actual"
        ]

    # Check if the lag features variable is None
    if lag_features is not None:
        for col in lag_features:
            for i in range(1, window_size + 1):
                df_eng["{}_lag{}".format(col, i)] = df_eng[col].shift(i)

    # Create rolling mean and standard deviation features
    for col in df_eng.columns:
        if col != "Stage1.Output.Measurement0.U.Actual":
            df_eng["{}_rolling_mean".format(col)] = (
                df_eng[col].rolling(window=window_size).mean()
            )
            df_eng["{}_rolling_std".format(col)] = (
                df_eng[col].rolling(window=window_size).std()
            )

    # Drop any rows with NaN values
    df_eng = df_eng.dropna()

    # Return the DataFrame with the new features added
    return df_eng


# Engineer the features
df_eng = features_engineer(df, window_size=60)

df_eng.to_pickle("../../data/interim/data.engineer.pkl")
