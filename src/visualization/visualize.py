import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_pickle("../../data/interim/data.processed.pkl")

ambient_columns = [col for col in df.columns if col.startswith("AmbientConditions")]
machine_columns = [col for col in df.columns if col.startswith("Machine")]
combiner_columns = [
    col for col in df.columns if col.startswith("FirstStage.CombinerOperation")
]
stage_output_columns = [col for col in df.columns if col.startswith("Stage1.Output")]


def plot_columns(columns, title):
    plt.figure(figsize=(20, 5))
    for col in columns:
        sns.lineplot(data=df, x=df.index, y=col, label=col)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.show()


def get_machine_properties(machine_columns):
    properties = set()
    for col in machine_columns:
        prop = ".".join(col.split(".")[1:])
        properties.add(prop)
    return properties


machine_properties = get_machine_properties(machine_columns)


def plot_machine_columns(properties, title):
    for prop in properties:
        plt.figure(figsize=(15, 6))
        for machine in range(1, 4):
            col = f"Machine{machine}.{prop}"
            sns.lineplot(data=df, x=df.index, y=col, label=col)
    plt.title(f"{title}: {prop}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.show()


def plot_individual_columns(columns, title):
    for col in columns:
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=df, x=df.index, y=col, label=col)
        plt.title(f"{title}: {col}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(loc="best")
        plt.show()


plot_columns(ambient_columns, "Ambient Conditions")
plot_machine_columns(machine_properties, "Machine Properties")
plot_columns(combiner_columns, "Stage 1 Combiner Operation")
plot_individual_columns(stage_output_columns, "Stage 1 Output Measurements")
