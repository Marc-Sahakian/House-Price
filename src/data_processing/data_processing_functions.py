def get_value_counts(data, columnName, dropna=False):
    """
    Calculate the value counts of a specified column in a DataFrame.

    This function takes a DataFrame 'data' and the name of a column 'columnName'.
    It calculates and returns the count of unique values in the specified column.

    Parameters:
        data (DataFrame): The DataFrame containing the data.
        columnName (str): The name of the column for which to calculate value counts.
        dropna (bool, optional): Whether to exclude NaN values from the value counts.
            Defaults to False.

    Returns:
        pandas.Series: A Series with value counts of the specified column.
    """

    return data[columnName].value_counts(dropna=dropna)


def count_na(data, columnName):
    """
    Count the number of missing (NaN) values in a specified column of a DataFrame.

    This function takes a DataFrame 'data' and the name of a column 'columnName'.
    It calculates and returns the count of missing (NaN) values in the specified column.

    Parameters:
        data (DataFrame): The DataFrame containing the data.
        columnName (str): The name of the column for which to count missing values.

    Returns:
        int: The number of missing (NaN) values in the specified column.
    """

    return data[columnName].isna().sum()


def encoding_2_num(data):
    """
    Encode categorical variables to numerical values in a DataFrame.

    This function takes a DataFrame 'data' and encodes specified categorical variables
    to numerical values. It identifies binary categorical variables (e.g., 'yes' and 'no')
    and encodes them as 1 and 0. Additionally, it encodes the 'furnishing_status' column
    with custom mapping.

    Parameters:
       data (DataFrame): The DataFrame containing the data.

    Returns:
       pd.DataFrame: The DataFrame with categorical variables encoded as numerical values.
    """
    
    colonnes_categ = []
    for c in data.columns:
        if data[c].dtype == object:
            colonnes_categ.append(c)
    # Encode binary categorical variables (Yes/No) as 1 and 0
    for col in colonnes_categ[:-1]:
        data[col] = data[col].map({'yes': 1, 'no': 0})

    data["furnishing_status"] = data["furnishing_status"].map({"semi-furnished": 1, 'furnished': 2, 'unfurnished': 0})

    return data
