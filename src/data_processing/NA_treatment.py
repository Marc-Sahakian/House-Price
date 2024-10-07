from src.figures.plot_figures import correlation_matrix


def NA_treatment(data): 
    """
    Perform treatment for missing values in the input DataFrame.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the dataset.

    Returns:
    pd.DataFrame: A new DataFrame with missing values treated using the following steps:
        1. Drop the 'house_age' column due to its high number of missing values (mostly NA).
        2. Drop rows with NA values in the 'price' column.
        3. Generate a correlation matrix before filling NA values (using the 'correlation_matrix' function).
        4. Sort the DataFrame based on the 'price' column in descending order.
        5. Fill missing values using forward fill (method='ffill').
        6. Generate a correlation matrix after filling NA values (using the 'correlation_matrix' function).
    """
    # dropping house_age column because it is mostly NA and has only 5 non-null values:
    data1 = data.drop("house_age", axis=1)

    # dropping rows where we have na in price column
    data1 = data1.dropna(subset=["price"])
    
    corr_matrix1 = correlation_matrix(data1, 'Correlation Heatmap before filling NA values')
    
    data2 = data1.sort_values(by=['price'], ascending=False)
    data2 = data2.fillna(method='ffill')

    corr_matrix2 = correlation_matrix(data2, 'Correlation Heatmap after filling NA values using ffill')
    
    return data2
