import matplotlib.pyplot as plt
import seaborn as sns


def plot_performance(train_sizes, model_name, mae_train, mae_test, r2_train, r2_test):
    """
    Plots the performance metrics of a machine learning model on training and test sets.

    Parameters:
        train_sizes (list): List of training set sizes used for training the model.
        model_name (str): Name of the machine learning model for labeling the plots.
        mae_train (list): List of Mean Absolute Error (MAE) values on the training set for each training size.
        mae_test (list): List of Mean Absolute Error (MAE) values on the test set for each training size.
        r2_train (list): List of R-squared (R²) values on the training set for each training size.
        r2_test (list): List of R-squared (R²) values on the test set for each training size.

    Returns:
        None: The function generates and displays a 1x2 subplot figure with MAE and R² performance plots.
    """
    # Create a figure and a grid of subplots 1x2
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot MAE in the first subplot
    axes[0].plot(train_sizes, mae_train, marker='o', linestyle='-', label='MAE_Train')
    axes[0].plot(train_sizes, mae_test, marker='o', linestyle='-', label='MAE_Test')
    axes[0].set_xlabel("Training set size")
    axes[0].set_ylabel("MAE (Mean Absolute Error)")
    axes[0].set_title(f"{model_name} model performance")
    axes[0].legend()
    axes[0].grid(True)

    # Plot R² in the second subplot
    axes[1].plot(train_sizes, r2_train, marker='o', linestyle='-', label='R2_Train')
    axes[1].plot(train_sizes, r2_test, marker='o', linestyle='-', label='R2_Test')
    axes[1].set_xlabel("Training set size")
    axes[1].set_ylabel("R² (R-Square)")
    axes[1].set_title(f"{model_name} model performance")
    axes[1].legend()
    axes[1].grid(True)

    # Adjust the spacing between the subplots
    plt.tight_layout()

    # Show the subplots
    plt.show()


def plot_mse_vs_max_depth(max_depth_values, mse_train_scores, mse_test_scores):
    """
    Plots the Mean Squared Error (MSE) versus Max Depth for a Random Forest model.

    Parameters:
        max_depth_values (list): List of max depth values used for training the Random Forest model.
        mse_train_scores (list): List of MSE scores on the training set for each max depth value.
        mse_test_scores (list): List of MSE scores on the test set for each max depth value.

    Returns:
        None: The function generates and displays a plot depicting MSE versus Max Depth for Random Forest.
    """
    plt.figure()
    plt.plot(max_depth_values, mse_train_scores, marker='o', linestyle='-', label='MSE_Train')
    plt.plot(max_depth_values, mse_test_scores, marker='o', linestyle='-', label='MSE_Test')
    plt.title('MSE vs. Max Depth for Random Forest')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_mae_vs_max_depth(max_depth_values, mae_train_scores, mae_test_scores):
    """
    Plots the Mean Absolute Error (MAE) versus Max Depth for a Random Forest model.

    Parameters:
        max_depth_values (list): List of max depth values used for training the Random Forest model.
        mae_train_scores (list): List of MAE scores on the training set for each max depth value.
        mae_test_scores (list): List of MAE scores on the test set for each max depth value.

    Returns:
        None: The function generates and displays a plot depicting MAE versus Max Depth for Random Forest.
    """
    plt.figure()
    plt.plot(max_depth_values, mae_train_scores, marker='o', linestyle='-', label='MAE_Train')
    plt.plot(max_depth_values, mae_test_scores, marker='o', linestyle='-', label='MAE_Test')
    plt.title('MAE vs. Max Depth for Random Forest')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_r2_vs_max_depth(max_depth_values, r2_train_scores, r2_test_scores):
    """
    Plots the R-squared (R²) versus Max Depth for a Random Forest model.

    Parameters:
        max_depth_values (list): List of max depth values used for training the Random Forest model.
        r2_train_scores (list): List of R² scores on the training set for each max depth value.
        r2_test_scores (list): List of R² scores on the test set for each max depth value.

    Returns:
        None: The function generates and displays a plot depicting R² versus Max Depth for Random Forest.
    """
    plt.figure()
    plt.plot(max_depth_values, r2_train_scores, marker='o', linestyle='-', label='R2_Train')
    plt.plot(max_depth_values, r2_test_scores, marker='o', linestyle='-', label='R2_Test')
    plt.xlabel("Max Depth")
    plt.ylabel("R2 (R_Square)")
    plt.title("R2 vs. Max Depth for Random Forest")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mse_vs_n_estimators(n_estimators_values, mse_train_scores, mse_test_scores):
    """
    Plots the Mean Squared Error (MSE) versus the number of estimators (n_estimators) for a Random Forest model.

    Parameters:
        n_estimators_values (list): List of values for the number of estimators used for training the Random Forest
            model.
        mse_train_scores (list): List of MSE scores on the training set for each number of estimators.
        mse_test_scores (list): List of MSE scores on the test set for each number of estimators.

    Returns:
        None: The function generates and displays a plot depicting MSE versus n_estimators for Random Forest.
    """
    plt.figure()
    plt.plot(n_estimators_values, mse_train_scores, marker='o', linestyle='-', label='MSE_Train')
    plt.plot(n_estimators_values, mse_test_scores, marker='o', linestyle='-', label='MSE_Test')
    plt.title('MSE vs. n_estimators for Random Forest')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_mae_vs_n_estimators(n_estimators_values, mae_train_scores, mae_test_scores):
    """
    Plots the Mean Absolute Error (MAE) versus the number of estimators (n_estimators) for a Random Forest model.

    Parameters:
        n_estimators_values (list): List of values for the number of estimators used for training the Random Forest
            model.
        mae_train_scores (list): List of MAE scores on the training set for each number of estimators.
        mae_test_scores (list): List of MAE scores on the test set for each number of estimators.

    Returns:
        None: The function generates and displays a plot depicting MAE versus n_estimators for Random Forest.
    """
    plt.figure()
    plt.plot(n_estimators_values, mae_train_scores, marker='o', linestyle='-', label='MAE_Train')
    plt.plot(n_estimators_values, mae_test_scores, marker='o', linestyle='-', label='MAE_Test')
    plt.title('MAE vs. n_estimators for Random Forest')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_r2_vs_n_estimators(n_estimators_values, r2_train_scores, r2_test_scores):
    """
    Plots the R-squared (R²) versus the number of estimators (n_estimators) for a Random Forest model.

    Parameters:
        n_estimators_values (list): List of values for the number of estimators used for training the Random Forest
            model.
        r2_train_scores (list): List of R² scores on the training set for each number of estimators.
        r2_test_scores (list): List of R² scores on the test set for each number of estimators.

    Returns:
        None: The function generates and displays a plot depicting R² versus n_estimators for Random Forest.
    """
    plt.figure()
    plt.plot(n_estimators_values, r2_train_scores, marker='o', linestyle='-', label='R2_Train')
    plt.plot(n_estimators_values, r2_test_scores, marker='o', linestyle='-', label='R2_Test')
    plt.xlabel("n_estimators")
    plt.ylabel("R2 (R_Square)")
    plt.title("R2 vs. n_estimators for Random Forest")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_true_vs_predicted(y_true, y_predicted, model_name="Model"):
    """
    Plots a scatter plot of true values versus predicted values for a given model.

    Parameters:
        y_true (array-like): True values of the target variable.
        y_predicted (array-like): Predicted values of the target variable.
        model_name (str): Name of the model for labeling the plot (default is "Model").

    Returns:
        None: The function generates and displays a scatter plot of true versus predicted values.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_predicted)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs. Predicted Values ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

    
def scatter_plot_true_vs_predicted(y_true, y_predicted, model_name="Model"):
    """
    Generates a scatter plot comparing true values to predicted values for a given model.

    Parameters:
        y_true (array-like): True values of the target variable.
        y_predicted (array-like): Predicted values of the target variable.
        model_name (str): Name of the model for labeling the plot (default is "Model").

    Returns:
        None: The function generates and displays a scatter plot comparing true versus predicted values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_true, c='blue', label='Real Values', alpha=0.5)
    plt.scatter(y_true, y_predicted, c='red', label='Predicted Values', alpha=0.5)
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Real Values vs. Predicted Values ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.show()
