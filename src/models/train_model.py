import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_pickle("../../data/processed/best_features.pkl")


def experiment_with_models(df, target_variable, train_test_split_ratio=0.75):
    """Experiments with different models to predict the target variable.

    Args:
      df: A Pandas DataFrame containing the data to be used.
      target_variable: The name of the target variable to be predicted.
      train_test_split_ratio: The proportion of the data to be used for training.

    Returns:
      A Pandas DataFrame containing the performance of the different models.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_variable, axis=1),
        df[target_variable],
        train_size=train_test_split_ratio,
    )

    # Create a list of models to experiment with
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
    }

    # Train and evaluate each model
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Plot the predicted values
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(y_pred, label="Predicted")

        # Plot the actual values
        ax.plot(y_test.values, label="Actual", color="black", linestyle="dashed")

        # Add a legend
        ax.legend()

        # Set the title and labels
        ax.set_title("Predicted vs Actual for {}".format(model_name))
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        # Set the x and y limits to be slightly larger than the range of the data
        ax.set_xlim(min(y_test) - 0.1, max(y_test) + 0.1)
        ax.set_ylim(min(y_pred) - 0.1, max(y_pred) + 0.1)

        # Set the y-axis label color to black
        ax.yaxis.label.set_color("black")

        # Show the plot
        plt.show()

        # Calculate the mean absolute error and R^2 score
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({"model_name": model_name, "mae": mae, "r2": r2})

    # Return a DataFrame with the results
    return pd.DataFrame(results)


results = experiment_with_models(
    df, target_variable="Stage1.Output.Measurement0.U.Actual"
)

# Get the MSE and R-squared scores from the results DataFrame
mse_scores = results["mae"]
r2_scores = results["r2"]
print(mse_scores, r2_scores)

# Sort the scores by MSE
sorted_results = results.sort_values(by="mae")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot the MSE scores in the first subplot
ax1.bar(sorted_results["model_name"], mse_scores)
ax1.set_title("Mean Squared Error Comparison")
ax1.set_ylabel("MSE")
ax1.grid(True)

# Plot the R-squared scores in the second subplot
ax2.plot(sorted_results["model_name"], r2_scores, "o-", color="red")
ax2.set_ylabel("R-Squared", color="red")
ax2.set_ylim(0, 1)

# Set the x-axis label for both subplots
fig.supxlabel("Model")

# Show the plot
plt.show()
