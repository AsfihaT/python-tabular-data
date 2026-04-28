#!/usr/bin/env python3

"""
iris_analysis.py

Perform linear regression of sepal length vs petal length
for each Iris species and generate plots.

Usage:
    python3 iris_analysis.py iris.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_data(file_path):
    """Load CSV data into pandas DataFrame."""
    return pd.read_csv(file_path)


def perform_regression(x, y):
    """Perform linear regression and return slope + intercept."""
    result = stats.linregress(x, y)
    return result.slope, result.intercept


def plot_species(dataframe, species_name):
    """Create scatter + regression plot for one species."""
    subset = dataframe[dataframe["species"] == species_name]

    x = subset["petal_length_cm"]
    y = subset["sepal_length_cm"]

    slope, intercept = perform_regression(x, y)

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x, slope * x + intercept, label="Regression")

    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.title(f"{species_name}")
    plt.legend()

    # save file
    filename = species_name.replace(" ", "_") + ".png"
    plt.savefig(filename)
    plt.close()


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python3 iris_analysis.py iris.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    df = load_data(file_path)

    species_list = df["species"].unique()

    for species in species_list:
        plot_species(df, species)

    print("Plots generated for all species.")


if __name__ == "__main__":
    main()
