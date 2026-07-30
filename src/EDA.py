import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from load_material_profiles import load_datasets
from property_statistics import property_statistics



NUMERICAL_COLUMNS = (FEATURE_COLUMNS + TARGET_COLUMNS)

def plot_histograms(df):
    n_columns = 3
    n_rows = 4

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(18,16)
    )

    axes = axes.flatten()

    for i,column in enumerate(NUMERICAL_COLUMNS):
        axes[i].hist(df[column].dropna(), bins=50)

        axes[i].set_title(column)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    # removes the unused subplot
    for i in range(len(NUMERICAL_COLUMNS), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_boxplots(df):

    n_columns = 3
    n_rows = 4

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(18,16)
    )

    axes = axes.flatten()

    for i,column in enumerate(NUMERICAL_COLUMNS):
        axes[i].boxplot(df[column].dropna())
        axes[i].set_title(column)

    for i in range(len(NUMERICAL_COLUMNS), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def correlation_heatmap(df):
    corr = (df[NUMERICAL_COLUMNS].corr())
    plt.figure(figsize=(14,10))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def missing_values(df):
    print("MISSING VALUES")
    print(df[NUMERICAL_COLUMNS].isna().sum())