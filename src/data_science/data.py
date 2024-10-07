import pandas as pd


def split_data(df: pd.DataFrame,
               test_ratio: float,
               seed: int) -> tuple[pd.DataFrame]:
    """
    Split the dataset into training and test sets by randomly shuffling the order
    of the data while fixing the random seed.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        test_ratio (float): The ratio of the dataset to be used for testing.
        seed (int): The seed for the random number generator to ensure reproducibility.

    Returns:
        tuple[pd.DataFrame]: Returns a tuple containing X_train, y_train, X_test, and y_test.
    """
    pass

    df_train = df.sample(frac=(1 - test_ratio), random_state=seed)
    # df_test = df - df_train
    df_test = df.merge(df_train, how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])

    X_train = df_train.drop("price", axis=1)
    y_train = df_train["price"]
    X_test = df_test.drop("price", axis=1)
    y_test = df_test["price"]
    
    return X_train, X_test, y_train, y_test
