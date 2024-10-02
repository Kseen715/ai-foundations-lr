import pandas as pd
import numpy as np


def read_csv(file):
    data = pd.read_csv(file)
    return data


def save_csv(data, file):
    data.to_csv(file, index=False)


def convert_to_binary_dataset(data):
    """Map every unique value in a column to a set of binary columns.

    Args:
        data (pd.DataFrame): The dataset to be converted.
    """
    for column in data.columns:
        unique_values = data[column].unique()
        for value in unique_values:
            data[f"{column}_{value}"] = data[column].apply(
                lambda x: 1 if x == value else 0)
        data.drop(column, axis=1, inplace=True)
    print(data.head())
    return data


if __name__ == "__main__":
    data1 = read_csv("data/bank_churn_dataset/raw/test.csv")
    data1['Balance'] = data1['Balance'].apply(lambda x: 0 if x == 0 else 1)
    data1['IsActiveMember'] = data1['IsActiveMember'].apply(
        lambda x: 0 if x == 0 else 1)
    print(data1.head())
    data1 = convert_to_binary_dataset(data1)
    print(data1.head())
    data = read_csv("data/bank_churn_dataset/raw/train.csv")
    data['Balance'] = data['Balance'].apply(lambda x: 0 if x == 0 else 1)
    data['IsActiveMember'] = data['IsActiveMember'].apply(
        lambda x: 0 if x == 0 else 1)
    data.drop(['Exited'], axis=1, inplace=True)
    print(data.head())
    data = convert_to_binary_dataset(data)
    print(data.head())
    data = pd.concat([data, data1], ignore_index=True)
    # to ints 
    data = data.fillna(0)
    data = data.replace([np.inf, -np.inf], 0)
    data = data.astype(int)
    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    save_csv(data, "data/bank_churn_dataset/data.csv")
