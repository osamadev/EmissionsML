import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import joblib
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
from IPython.display import display, HTML

def remove_all_zero_rows(df, subset=None, exclude=None):
    """
    Removes rows where all specified (or all numeric) columns have a value of zero,
    with optional exclusion of some columns.
    """
    if subset is None:
        subset = df.select_dtypes(include='number').columns.tolist()

    if exclude is not None:
        subset = [col for col in subset if col not in exclude]

    # Create a mask where all values in the row for the subset are zero
    mask_all_zeros = (df[subset] == 0).all(axis=1)

    print(f"Removed {mask_all_zeros.sum()} all-zero rows (based on {len(subset)} columns).")

    return df[~mask_all_zeros].copy()

def plot_correlation_matrix(correlations_df, original_df, top_n=40, correlation_cols=None, title="Correlation Matrix", figsize=(12, 10)):
    """
    Plots a heatmap of the correlation matrix for the top N correlated features.
    """
    if correlation_cols is None:
        correlation_cols = ['pearson_co2_avg', 'spearman_co2_avg', 'kendall_co2_avg']

    # Calculate a combined score to rank features
    correlations_df['sum'] = correlations_df[correlation_cols].sum(axis=1)

    # Select top N features by combined correlation strength
    top_features_df = correlations_df.nlargest(top_n, 'sum')
    selected_columns = top_features_df.index.tolist()

    # Extract from original dataset
    df_processed = original_df[selected_columns]

    # Generate and show correlation heatmap
    corr_matrix = df_processed.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
    plt.title(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

    return selected_columns

def plot_correlation_heatmap(df, figsize=(17, 10), title='Correlation between features'):
    """
    Plots a heatmap showing the correlation between variables in a DataFrame.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def scale_data(X, y=None, scaler_X=None, scaler_Y=None, scaler_type="standard", fit_scaler=True):
    """
    Scales input features and target values using StandardScaler or MinMaxScaler.
    
    Parameters:
    X (array-like): Input features to be scaled.
    y (array-like, optional): Target values to be scaled.
    scaler_X (object, optional): Pre-fitted scaler for X; if None, a new scaler is created.
    scaler_Y (object, optional): Pre-fitted scaler for y; if None, a new scaler is created.
    scaler_type (str): Type of scaler to use; either 'standard' (default) or 'minmax'.
    fit_scaler (bool): Whether to fit the scaler before transforming (default: True).
    
    Returns:
    tuple: Scaled X, scaled y (if provided), scaler_X, and scaler_Y (if applicable).
    """
    # Choose the scaler type
    if scaler_type == "minmax":
        scaler_X = scaler_X if scaler_X else MinMaxScaler()
        scaler_Y = scaler_Y if scaler_Y else MinMaxScaler() if y is not None else None
    else:
        scaler_X = scaler_X if scaler_X else StandardScaler()
        scaler_Y = scaler_Y if scaler_Y else StandardScaler() if y is not None else None

    # Scale input features
    if fit_scaler:
        X_scaled = scaler_X.fit_transform(X)
    else:
        X_scaled = scaler_X.transform(X)

    # Scale target values if provided
    if y is not None:
        if fit_scaler:
            y_scaled = scaler_Y.fit_transform(y)
        else:
            y_scaled = scaler_Y.transform(y)
        return X_scaled, y_scaled, scaler_X, scaler_Y
    else:
        return X_scaled, None, scaler_X, None

def compare_predictions(y_test, y_pred, sample_sizes=None, randomize=True, model_name="Model"):
    """
    Compares actual vs. predicted values for multiple target variables.
    
    Parameters:
    y_test (pd.DataFrame): Actual target values.
    y_pred (array-like): Predicted target values.
    sample_sizes (dict, optional): Dictionary specifying sample size per target variable.
    randomize (bool): Whether to randomly select samples (default: True).
    model_name (str): Name of the model for display purposes.
    
    Returns:
    pd.DataFrame: A table comparing actual and predicted values for selected samples.
    """
    # Convert predictions to DataFrame with correct column names
    y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)

    # Reset index for consistency
    y_test_reset = y_test.reset_index(drop=True)
    y_pred_df_reset = y_pred_df.reset_index(drop=True)

    # If no sample sizes provided, set default (5 per target or available count)
    if sample_sizes is None:
        sample_sizes = {col: min(5, len(y_test_reset)) for col in y_test.columns}

    # Store selected samples
    selected_rows = []

    for target in y_test.columns:
        total_samples = min(len(y_test_reset), len(y_pred_df_reset))
        sample_size = min(sample_sizes.get(target, 5), total_samples)  # Ensure valid size

        if randomize:
            indices = np.random.choice(total_samples, size=sample_size, replace=False)
        else:
            indices = np.arange(sample_size)

        # Store selected rows for the target
        selected_rows.append(pd.DataFrame({
            f"Actual_{target}": y_test_reset.iloc[indices][target].values,
            f"Predicted_{target}": y_pred_df_reset.iloc[indices][target].values,
        }))

    # Concatenate all selected samples side by side for tabular format
    df_compare = pd.concat(selected_rows, axis=1)

    # Display the comparison table
    print(f"\n=== {model_name} - Actual vs. Predicted Comparison ===")

    return df_compare

def evaluate_regressor(y_true, y_pred, model_name="Model"):
    """
    Evaluates a regression model's performance using multiple metrics.
    
    Parameters:
    y_true (pd.DataFrame): Actual target values.
    y_pred (array-like): Predicted target values.
    model_name (str): Name of the model for display purposes.
    
    Returns:
    pd.DataFrame: A table containing evaluation metrics (MAE, R², MSE, RMSE) for each target variable and overall.
    """
    # Calculate per-target metrics
    mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)

    # Overall MSE and RMSE
    overall_mse = mean_squared_error(y_true, y_pred)
    overall_rmse = np.sqrt(overall_mse)

    # Create a DataFrame for better visualization
    metrics_df = pd.DataFrame({
        "MAE": mae,
        "R² Score": r2,
        "MSE": mse,
        "RMSE": rmse
    }, index=y_true.columns)

    # Add overall metrics as a separate row
    overall_metrics = pd.DataFrame({
        "MAE": [np.mean(mae)],
        "R² Score": [np.mean(r2)],
        "MSE": [overall_mse],
        "RMSE": [overall_rmse]
    }, index=["Overall"])
    
    metrics_df = pd.concat([metrics_df, overall_metrics])

    # Print table of metrics
    print(f"\n=== {model_name} Evaluation Metrics ===")

    # Plot scatter plots for actual vs. predicted values
    num_targets = len(y_true.columns)
    num_cols = min(2, num_targets)  # Use at most 2 columns
    num_rows = (num_targets + 1) // 2  # Adjust dynamically

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    axes = axes.flatten() if num_targets > 1 else [axes]  # Handle single target case
    
    for i, target in enumerate(y_true.columns):
        ax = axes[i]
        sns.scatterplot(x=y_true[target], y=y_pred[:, i], alpha=0.5, ax=ax)
        sns.regplot(x=y_true[target], y=y_pred[:, i], scatter=False, color="red", ax=ax)  # Regression line
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{model_name} - Actual vs. Predicted ({target})")
        ax.axline((0, 0), slope=1, linestyle="--", color="gray", linewidth=1)  # Identity line (y=x)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return metrics_df

def plot_feature_importance(model_estimators, X, y):
    """
    Plots the feature importance for each target variable using trained estimators.
    
    Parameters:
    model_estimators (list): A list of trained models corresponding to each target variable.
    X (pd.DataFrame): Input feature set used for training.
    y (pd.DataFrame): Target variable set with multiple target columns.
    
    Returns:
    None: Displays bar plots showing the importance of each feature for each target variable.
    """
    # Extract feature importance for each target variable
    feature_importance = pd.DataFrame(
        {target: est.feature_importances_ for target, est in zip(y.columns, model_estimators)},
        index=X.columns
    )

    num_targets = len(y.columns)
    num_cols = min(2, num_targets)  # Use at most 2 columns
    num_rows = (num_targets + 1) // 2  # Adjust row count dynamically

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    axes = axes.flatten() if num_targets > 1 else [axes]  # Handle single target case

    for i, target in enumerate(y.columns):
        sorted_importance = feature_importance[target].sort_values(ascending=False)
        sorted_importance.plot(kind="bar", color="royalblue", ax=axes[i])
        axes[i].set_title(f"Feature Importance for {target}")
        axes[i].set_ylabel("Importance Score")
        axes[i].tick_params(axis='x', rotation=90)

    # Remove unused subplots if there are fewer than 4 targets
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlapping
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=1.9)
    plt.show()

def save_model(model, model_filename, scaler_X=None, scaler_Y=None, target_columns=None):
    """
    Saves a trained model and additional metadata (scalers, target columns) to a file.
    
    Parameters:
    model (object): Trained machine learning model.
    model_filename (str): Filename to save the model.
    scaler_X (object, optional): Scaler used for input features.
    scaler_Y (object, optional): Scaler used for target values.
    target_columns (list, optional): List of target column names.
    """
    model_data = {
        "model": model,
        "scaler_X": scaler_X,  # Save input scaler if exists
        "scaler_Y": scaler_Y,  # Save output scaler if exists
        "target_columns": target_columns  # Save target columns for reference
    }

    with open(model_filename, 'wb') as file:
        pickle.dump(model_data, file)

    print(f"Model and additional data saved to {model_filename}")

def load_model(model_filename):
    """
    Loads a saved model along with scalers and target column metadata.
    
    Parameters:
    model_filename (str): Filename from which to load the model.
    
    Returns:
    tuple: Loaded model, scaler_X, scaler_Y, and target_columns.
    """
    with open(model_filename, 'rb') as file:
        model_data = pickle.load(file)

    model = model_data.get("model")
    scaler_X = model_data.get("scaler_X")
    scaler_Y = model_data.get("scaler_Y")
    target_columns = model_data.get("target_columns")

    print(f"Model loaded from {model_filename}")
    return model, scaler_X, scaler_Y, target_columns

# load cleaned dataset from csv file, and split into features and target
def load_data(data_filename, target_columns: list):
    """
    Loads a cleaned dataset from a CSV file and splits it into features and target variables.
    
    Parameters:
    data_filename (str): Path to the CSV file.
    target_columns (list): List of column names to be used as target variables.
    
    Returns:
    tuple: Feature matrix (X) and target values (y).
    """
    df = pd.read_csv(data_filename)
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    return X, y

# save cleaned dataset as csv file
def save_data(X, y, data_filename):
    """
    Saves cleaned dataset as a CSV file.
    
    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.DataFrame): Target values.
    data_filename (str): Filename to save the dataset.
    """
    df = pd.concat([X, y], axis=1)
    df.to_csv(data_filename, index=False)
    print(f"Data saved as {data_filename}")

# implement a function to apply GridSearchCV to a model
def tune_model(model, param_grid, X_train, y_train):
    """
    Performs hyperparameter tuning using GridSearchCV.
    
    Parameters:
    model (object): The machine learning model to tune.
    param_grid (dict): Dictionary of hyperparameters to search.
    X_train (array-like): Training feature matrix.
    y_train (array-like): Training target values.
    
    Returns:
    object: The best estimator found by GridSearchCV.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def plot_ridge_feature_importance(ridge_model, X, target_columns):
    """
    Plots feature importance for Ridge regression model coefficients.
    
    Parameters:
    ridge_model (object): Trained Ridge regression model.
    X (pd.DataFrame): Feature matrix.
    target_columns (list): List of target variable names.
    """
    # Ensure the coefficients are a DataFrame with correct dimensions
    feature_importance = pd.DataFrame(
        ridge_model.coef_.T,  # Transpose to match (features, targets)
        index=X.columns,  # Feature names as index
        columns=target_columns  # Target names as columns
    )

    num_targets = len(target_columns)
    rows = math.ceil(num_targets / 2)  # Calculate rows needed for 2 columns
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))  # Adjust figure size

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, target in enumerate(target_columns):
        sorted_coeffs = feature_importance[target].sort_values(ascending=False)
        sorted_coeffs.plot(kind="bar", color="royalblue", ax=axes[i])
        axes[i].set_title(f"Feature Importance for {target}")
        axes[i].set_xlabel("Feature")
        axes[i].set_ylabel("Coefficient Value")
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    # Hide any unused subplots (if target_columns < 4)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_mlp_feature_importance(pipeline, X, y):
    """
    Extracts and plots feature importance from an MLPRegressor model.
    
    Parameters:
    - mlp_regressor: The trained MLPRegressor model (should be extracted from a pipeline if needed).
    - X: The feature matrix (DataFrame).
    - y: The target variable(s) (DataFrame with multiple columns for multi-output regression).
    """

    # Extract the actual MLPRegressor from the pipeline
    mlp_regressor = pipeline.named_steps["mlp"]  # Extract MLPRegressor

    # Create an empty DataFrame to store feature importance
    feature_importance = pd.DataFrame()

    # Extract input-to-hidden layer weights (feature importance)
    coefs = mlp_regressor.coefs_[0]  # First weight matrix (input to hidden layer)

    for i, target in enumerate(y.columns):
        feature_importance[target] = coefs[:, i]  # Extract weights for each target

    # Set feature names as index
    feature_importance.index = X.columns

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot feature importance for each target variable
    for i, target in enumerate(y.columns):
        sorted_coeffs = feature_importance[target].sort_values(ascending=False)
        sorted_coeffs.plot(kind="bar", color="royalblue", ax=axes[i])
        axes[i].set_title(f"Feature Importance for {target}")
        axes[i].set_ylabel("Coefficient Value")
        axes[i].tick_params(axis="x", rotation=90)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df, target_columns, plot=True, scatter=True, scatter_x=None):
    """
    Detects and visualizes outliers using the IQR method with compact grid plotting.
    """
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")

    outlier_masks = {}
    combined_mask = pd.Series([False] * len(df), index=df.index)

    # Detect outliers using IQR
    for col in target_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_masks[col] = mask
        combined_mask |= mask
        print(f"{col}: {mask.sum()} outliers detected")

    outliers_df = df[combined_mask]
    print(f"\nTotal outlier rows: {len(outliers_df)} out of {len(df)} ({len(outliers_df) * 100 / len(df):.2f}%)")

    # Compact grid plots
    n = len(target_columns)

    if plot:
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        for i, col in enumerate(target_columns):
            sns.boxplot(x=df[col], ax=axes[i], color=palette[i % len(palette)], linewidth=1.5, fliersize=4)
            axes[i].set_title(f"Boxplot: {col}", fontsize=11)
            axes[i].set_xlabel(col)
            axes[i].grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()

    if scatter and scatter_x and scatter_x in df.columns:
        fig, axes = plt.subplots(len(target_columns), 1, figsize=(7, 3 * len(target_columns)))

        if len(target_columns) == 1:
            axes = [axes]

        for i, col in enumerate(target_columns):
            sns.scatterplot(data=df, x=scatter_x, y=col, ax=axes[i], color=palette[i % len(palette)], s=40)
            sns.scatterplot(data=outliers_df, x=scatter_x, y=col, ax=axes[i], color='red', marker='X', s=50, label='Outliers')
            axes[i].set_title(f"{col} vs {scatter_x}", fontsize=11)
            axes[i].grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()

    return outliers_df

def analyze_correlations(df, target_column="co2_avg"):
    """
    Computes correlation metrics and feature importance scores with respect to a target emission column.
    """
    df_correlations = pd.DataFrame(index=df.columns)

    # Pearson, Spearman, Kendall
    df_correlations[f"pearson_{target_column}"] = df.corrwith(df[target_column])
    df_correlations[f"spearman_{target_column}"] = df.corrwith(df[target_column], method="spearman")
    df_correlations[f"kendall_{target_column}"] = df.corrwith(df[target_column], method="kendall")

    # Select numeric float columns only
    float_columns = [col for col in df.columns if df[col].dtype == float and col != target_column]

    X = df[float_columns].copy()
    y = df[target_column].copy()

    X.fillna(0, inplace=True)
    y.fillna(0, inplace=True)

    # F-statistics
    fs = feature_selection.f_regression(X, y, center=True)
    f_statistic = pd.Series([round(f, 10) for f in fs[0]], index=float_columns)
    df_correlations[f"f_statistic_{target_column}"] = f_statistic

    # Mutual Information
    mi_r = feature_selection.mutual_info_regression(X, y)
    mi_r_series = pd.Series(mi_r, index=float_columns)
    df_correlations[f"mi_r_{target_column}"] = mi_r_series

    # Drop the row for the target itself to avoid self-correlation clutter
    df_correlations = df_correlations.drop(index=target_column)

    # Sort by absolute Pearson correlation (you can change to another metric if preferred)
    df_correlations["abs_pearson"] = df_correlations[f"pearson_{target_column}"].abs()
    df_sorted = df_correlations.sort_values("abs_pearson", ascending=False).drop(columns=["abs_pearson"])

    return df_sorted

def correlate_targets(df, target_columns, feature_columns, method='pearson', plot=True):
    """
    Calculates and visualizes correlation between emissions target variables and selected feature(s).
    """
    # Combine and compute correlation
    selected_cols = feature_columns + target_columns
    corr_matrix = df[selected_cols].corr(method=method)

    # Slice correlation matrix to show only target-feature correlation
    corr_targets = corr_matrix.loc[feature_columns, target_columns]

    if plot:
        plt.figure(figsize=(1.2 * len(target_columns), 0.6 * len(feature_columns) + 1))
        sns.heatmap(corr_targets, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt=".2f")
        plt.title(f"{method.capitalize()} Correlation: Features vs Targets", fontsize=12, weight='bold')
        plt.tight_layout()
        plt.show()

    return corr_targets

def remove_outliers_iqr(df, target_columns):
    """
    Removes outliers from the specified columns in a dataset using the Interquartile Range (IQR) method.

    Parameters:
    - data_filename (str): Path to the CSV file containing the dataset.
    - target_columns (list of str): Names of the columns for which outliers should be removed.

    Returns:
    - pd.DataFrame: A new DataFrame with outliers removed.
    """
    
    print("Shape before removing outliers:", df.shape)
    
    # Store initial count
    initial_target_count = len(df)
    
    # Compute IQR for selected columns
    Q1 = df[target_columns].quantile(0.25)
    Q3 = df[target_columns].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Keep only rows where all target columns are within bounds
    mask = (df[target_columns] >= lower_bound) & (df[target_columns] <= upper_bound)
    
    # Filter data: keep rows where ALL selected columns are within bounds
    data_filtered = df[mask.all(axis=1)]

    removed_count = initial_target_count - len(data_filtered)
    remove_count_pourcentage = removed_count*100/initial_target_count
    print(f"Removed {removed_count} ({remove_count_pourcentage:.2f} %) outliers from {target_columns}")
    print("Shape after removing outliers:", data_filtered.shape)
    
    return data_filtered

def smart_grid_search(model, model_name, param_grid, X_train, y_train, X_test, y_test, apply_target_scaling=False, cv_splits=5, scoring="r2", save_model=True):
    """
    Performs Grid Search with cross-validation to find the best hyperparameters for a given model.
    Optionally applies target scaling for MLPRegressor models.
    
    Parameters:
    - model: The machine learning model to tune.
    - model_name: Name of the model (used for saving and logging).
    - param_grid: Dictionary of hyperparameters to search over.
    - X_train, y_train: Training data.
    - X_test, y_test: Test data for evaluation.
    - apply_target_scaling: Whether to apply target scaling (for MLPRegressor models).
    - cv_splits: Number of cross-validation folds.
    - scoring: Scoring metric for model evaluation.
    - save_model: Whether to save the best model.
    
    Returns:
    - Dictionary containing the best model, best parameters, and evaluation metrics.
    """
    print(f"\n🔍 Running Grid Search for {model_name}...\n")

    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # If model is MLPRegressor, apply target scaling
    if apply_target_scaling:
        print("⚡ Applying target scaling for MLPRegressor...")
        pipeline = Pipeline([
            ("mlp", model)  # Model inside the pipeline
        ])

        # Wrap in TransformedTargetRegressor for target scaling
        model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
    
    
    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Evaluate model on test data
    y_pred = grid_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results = {
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "mae": mae,
        "r2": r2,
        "mse": mse,
        "rmse": rmse
    }

    # Calculate per-target metrics
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)

    # Overall MSE and RMSE
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_rmse = np.sqrt(overall_mse)

    # Create a DataFrame for better visualization
    metrics_df = pd.DataFrame({
        "MAE": mae,
        "R² Score": r2,
        "MSE": mse,
        "RMSE": rmse
    }, index=y_test.columns)

    # Add overall metrics as a separate row
    overall_metrics = pd.DataFrame({
        "MAE": [np.mean(mae)],
        "R² Score": [np.mean(r2)],
        "MSE": [overall_mse],
        "RMSE": [overall_rmse]
    }, index=["Overall"])
    
    metrics_df = pd.concat([metrics_df, overall_metrics])


    print(f"\n✅ Best Parameters for {model_name}: {grid_search.best_params_}\n")
    print(f"📊 Metrics evaluations of the Best Parameters:\n")
    print(metrics_df)

    # Save the model
    if save_model:
        filename = f"./checkpoints/{model_name}_best_model.pkl"
        joblib.dump(grid_search.best_estimator_, filename)
        print(f"\n💾 Model saved: {filename}")

    return results,metrics_df


def smart_grid_search_advanced(
    model,
    model_name,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    apply_target_scaling=False,
    use_random_search=False,
    n_iter=10,
    cv_splits=5,
    scoring="r2",
    save_model=True,
):
    """
    Performs either Grid Search or Randomized Search for hyperparameter tuning with cross-validation.
    Optionally applies target scaling and supports MultiOutputRegressor models.
    
    Parameters:
    - model: The machine learning model to tune.
    - model_name: Name of the model (used for saving and logging).
    - param_grid: Dictionary of hyperparameters to search over.
    - X_train, y_train: Training data.
    - X_test, y_test: Test data for evaluation.
    - apply_target_scaling: Whether to apply target scaling (for MLPRegressor models).
    - use_random_search: Whether to use RandomizedSearchCV instead of GridSearchCV.
    - n_iter: Number of iterations for RandomizedSearchCV.
    - cv_splits: Number of cross-validation folds.
    - scoring: Scoring metric for model evaluation.
    - save_model: Whether to save the best model.
    
    Returns:
    - Dictionary containing the best model, best parameters, and evaluation metrics.
    """
    print(f"\n🔍 Running {'Randomized' if use_random_search else 'Grid'} Search for {model_name}...\n")

    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # If model is MLPRegressor, apply target scaling
    if apply_target_scaling:
        print("⚡ Applying target scaling for MLPRegressor...")
        pipeline = Pipeline([
            ("mlp", model)  # Model inside the pipeline
        ])
        # Wrap in TransformedTargetRegressor for target scaling
        model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

    # If model is wrapped inside MultiOutputRegressor (e.g., XGBoost)
    if isinstance(model, MultiOutputRegressor):
        param_grid = {f"estimator__{key}": value for key, value in param_grid.items()}

    # Choose between Grid Search or Randomized Search
    search_class = RandomizedSearchCV if use_random_search else GridSearchCV
    search = search_class(
        model,
        param_distributions=param_grid if use_random_search else param_grid,  # param_distributions for RandomizedSearchCV
        cv=kfold,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        n_iter=n_iter if use_random_search else None  # n_iter only applies to RandomizedSearchCV
    )

    # Fit search
    search.fit(X_train, y_train)

    # Evaluate model on test data
    y_pred = search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results = {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "mae": mae,
        "r2": r2,
        "mse": mse,
        "rmse": rmse
    }
    
    # Calculate per-target metrics
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)

    # Overall MSE and RMSE
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_rmse = np.sqrt(overall_mse)

    # Create a DataFrame for better visualization
    metrics_df = pd.DataFrame({
        "MAE": mae,
        "R² Score": r2,
        "MSE": mse,
        "RMSE": rmse
    }, index=y_test.columns)

    # Add overall metrics as a separate row
    overall_metrics = pd.DataFrame({
        "MAE": [np.mean(mae)],
        "R² Score": [np.mean(r2)],
        "MSE": [overall_mse],
        "RMSE": [overall_rmse]
    }, index=["Overall"])
    
    metrics_df = pd.concat([metrics_df, overall_metrics])
    
    print(f"\n✅ Best Parameters for {model_name}: {search.best_params_}\n")
    #print(f"📊 Test MAE: {mae:.4f}, Test R²: {r2:.4f}, Test MSE: {mse:.4f}, Test RMSE: {rmse:.4f}\n")
    print(f"📊 Metrics evaluations of the Best Parameters:\n")
    print(metrics_df)

    # Save the model
    if save_model:
        filename = f"./checkpoints/{model_name}_best_model.pkl"
        joblib.dump(search.best_estimator_, filename)
        print(f"\n💾 Model saved: {filename}")

    return results, metrics_df

def display_side_by_side(df1, df2, df1_title="Evaluation of initial modelling", df2_title="Evaluation of improved modelling"):
    """
    Displays two pandas DataFrames side by side in an HTML format for easy comparison.

    Parameters:
    - df1: First DataFrame to display (e.g., initial model evaluation results).
    - df2: Second DataFrame to display (e.g., improved model evaluation results).
    - df1_title: Title for the first DataFrame (default: "Evaluation of initial modelling").
    - df2_title: Title for the second DataFrame (default: "Evaluation of improved modelling").
    """
    display(HTML(
        f"""
        <div style="display: flex; justify-content: space-around;">
            <div>
                <h4>{df1_title}</h4>
                {df1.to_html()}
            </div>
            <div>
                <h4>{df2_title}</h4>
                {df2.to_html()}
            </div>
        </div>
        """
    ))