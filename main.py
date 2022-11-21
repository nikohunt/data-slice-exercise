import numpy as np
import pandas as pd
from sklearn import datasets


def download_iris():
    """Downloads the popular iris dataset"""
    return datasets.load_iris()


def make_iris_dataframe(iris):
    """Returns a dataframe of the iris numpy dataset

    Args:
        iris (sklearn.dataset): iris dataset from sklearn.datasets
    """

    target = iris.target.reshape(len(iris.target), 1)
    iris_arr = np.concatenate((iris.data, target), axis=1)
    df_iris = pd.DataFrame(iris_arr, columns=iris.feature_names + ["target"])
    df_iris["target"] = df_iris.loc[:, "target"].map(
        dict(enumerate(iris.target_names))
    )
    return df_iris


def stats_by_feature(df, feature, iris_target_names):
    """Provides descriptive statistics for a given feature in a dataset.

    Args:
        df (pd.DataFrame): Input DataFrame
        feature (str): Name of
    """
    print(f"Feature: {feature}")
    for flower in iris_target_names:
        df_temp = df[df.target == flower][feature]
        mean = df_temp.mean()
        stddev = df_temp.std()
        print(f"    Flower: {flower}")
        print(f"        mean: {mean:.4f}")
        print(f"        stddev: {stddev:.4f}")


if __name__ == "__main__":
    iris_dataset = download_iris()
    df_iris = make_iris_dataframe(iris_dataset)

    for feature in iris_dataset.feature_names:
        stats_by_feature(df_iris, feature, iris_dataset.target_names)
