import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif

warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_pickle("../../data/interim/data.engineer.pkl")


def select_best_features(
    df, target="Stage1.Output.Measurement0.U.Actual", k=10, metric="f_classif"
):
    """Selects the k best features from a DataFrame with respect to the target variable.

    Args:
      df: A Pandas DataFrame.
      target: The name of the target variable.
      k: The number of features to select.
      metric: The metric to use for feature selection. One of 'f_classif' or 'mutual_info_classif'.

    Returns:
      A list of the k best features.
    """

    # Create a SelectKBest selector
    selector = SelectKBest(
        f_classif if metric == "f_classif" else mutual_info_classif, k=k
    )

    # Fit the selector to the data
    selector.fit(df.drop(columns=target), df[target])

    # Get the selected features
    selected_features = selector.get_support(indices=True)

    # Return the selected features
    return list(df.columns[selected_features])


best_features = select_best_features(
    df, target="Stage1.Output.Measurement0.U.Actual", k=10, metric="f_classif"
)

bestfeatures_df = df[best_features]
bestfeatures_df.columns

bestfeatures_df = pd.concat(
    (bestfeatures_df, df["Stage1.Output.Measurement0.U.Actual"]), axis=1
)

bestfeatures_df.to_pickle("../../data/processed/best_features.pkl")
