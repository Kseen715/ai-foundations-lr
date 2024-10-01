import pandas as pd


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
    data = read_csv("data/portuguese_bank_marketing/raw/Bank Marketing.csv")
    data.drop(["Age", "Balance (euros)", "Last Contact Duration",
              "Pdays"], axis=1, inplace=True)
    print(data.head())
    data = convert_to_binary_dataset(data)
    print(data.head())
    data = data.sample(frac=1).reset_index(drop=True)
    save_csv(data, "data/portuguese_bank_marketing/Bank Marketing.csv")
