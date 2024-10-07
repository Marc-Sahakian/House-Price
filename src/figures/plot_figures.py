import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def correlation_matrix(data, title='Correlation Heatmap'):
    """
    Generate and display a heatmap of the correlation matrix for a DataFrame.

    This function takes a DataFrame 'data' and calculates the correlation matrix
    between numerical columns. It then creates a heatmap to visualize the correlations
    and displays it using Matplotlib and Seaborn.

    Parameters:
        data (DataFrame): The DataFrame containing numerical columns for correlation analysis.
        title (str, optional): The title to be displayed on the correlation heatmap.
        Defaults to 'Correlation Heatmap'.

    Returns:
        pandas.DataFrame: The correlation matrix of the input DataFrame.
        (Displays a heatmap of the correlation matrix with the specified title.)
    """
    corr_matrix = data.corr()
    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()
    return corr_matrix


def univariate_plot(data, columns):
    """
    Create subplots with bar plots for the frequency of unique values in DataFrame columns.

    This function takes a DataFrame 'data' and a list of column names 'columns'. It calculates
    the frequency of each unique value in the specified columns and creates subplots of
    bar plots to visualize the distribution of values.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        columns (list): A list of column names for which to create the univariate plots.


    Returns:
        None: The function displays the subplot of bar plots.
    """
    num_columns = len(columns)
    num_rows = (num_columns + 1) // 2
    num_cols = 2
    
    figsize = (12, 4 * num_rows)
    plt.figure(figsize=figsize)
    for i, column in enumerate(columns, start=1):
        plt.subplot(num_rows, num_cols, i)
        
        value_counts = dict(Counter(data[column]))
        unique_values = list(value_counts.keys())
        value_frequencies = list(value_counts.values())
        
        plt.bar(unique_values, value_frequencies)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(column)
        
        for i, v in enumerate(value_frequencies):
            plt.text(unique_values[i], v, str(v), ha='center', va='bottom')
    plt.tight_layout(pad=2.0)
    plt.show()


def univariate_boxplots(data):
    """
    Create subplots with box plots for the spread of values in DataFrame columns.

    This function takes a DataFrame 'data' and creates subplots of box plots to visualize
    the distribution of values in each column.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None: The function displays the subplot of box plots.
    """
    col = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    num_rows = (len(col) + 1) // 2
    num_cols = 2
    
    figsize = (12, 4 * num_rows)
    
    plt.figure(figsize=figsize)
    
    for i, column in enumerate(col, start=1):
        plt.subplot(num_rows, num_cols, i)
        
        plt.boxplot(data[column])
        plt.xlabel(column)
        plt.ylabel('Values')
        plt.title(f'Box Plot for {column}')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.yticks(plt.yticks()[0])
    plt.tight_layout(pad=2.0)
    plt.show()


def bivariate_plot(data, column_list):
    """
    Create a 3x2 subplot with bar plots for each column in the list of columns.

    This function takes a DataFrame 'data' and a list of column names 'column_list'. It creates a 2x2 subplot
    with bar plots between the values of each column in the list.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_list (list[str]): The list of column names for which to create the bivariate plots.

    Returns:
        None: The function displays the 2x2 subplot of bar plots.
    """
    num_rows = 3
    num_cols = 2
    
    plt.figure(figsize=(10, 8))
    
    for i, column in enumerate(column_list, start=1):
        if i > num_rows * num_cols:
            break
        plt.subplot(num_rows, num_cols, i)
        plt.bar(data[column], data["price"], data=data)
        plt.xlabel(column)
        plt.ylabel("price")
        plt.title(f'{column} vs price')
    plt.tight_layout()
    plt.show()


def bivariate_boxplot(data, columns):
    """
    Create a 2x2 subplot with boxplots for the values of specific columns in the DataFrame.

    This function takes a DataFrame 'data' and a list of two column names 'columns'. It creates a 2x2 subplot
    with boxplots between the values of the two specified columns.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        columns (list): A list of two column names for which to create the bivariate boxplots.

    Returns:
        None: The function displays the 2x2 subplot of boxplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Define colors for each boxplot
    colors = ['red', 'yellow', 'green', 'blue', 'purple', 'magenta']

    for i, column in enumerate(columns):
        sns.boxplot(data, x=data[column], y=data["price"], ax=axes[i], color=colors[i])
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Price")   
    plt.tight_layout()
    plt.show()


def scatter_plot(data, x_column, y_column, title='', xlabel='', ylabel=''):
    """
    Create a scatter plot for two columns in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        x_column (str): The column name for the x-axis.
        y_column (str): The column name for the y-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        None: The function displays the scatter plot.
    """
    plt.scatter(data[x_column], data[y_column], alpha=0.5)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()
