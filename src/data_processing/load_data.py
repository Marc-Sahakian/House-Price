import pandas as pd


def load_data(filename: str) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        filename (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        dataframe = pd.read_csv(filename)
        return dataframe
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
