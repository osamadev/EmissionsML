import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
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


def scale_data(X, y=None, scaler_X=None, scaler_Y=None, scaler_type="standard", fit_scaler=True):
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
    Load cleaned dataset from a csv file, and split into features and target.
    """
    df = pd.read_csv(data_filename)
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    return X, y

# save cleaned dataset as csv file
def save_data(X, y, data_filename):
    """
    Save cleaned dataset as a csv file.
    """
    df = pd.concat([X, y], axis=1)
    df.to_csv(data_filename, index=False)
    print(f"Data saved as {data_filename}")

# implement a function to apply GridSearchCV to a model
def tune_model(model, param_grid, X_train, y_train):
    """
    Implement a function to apply GridSearchCV to a model.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def plot_ridge_feature_importance(ridge_model, X, target_columns):
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

def smart_grid_search(model, model_name, param_grid, X_train, y_train, X_test, y_test, apply_target_scaling=False, cv_splits=5, scoring="r2", save_model=True):
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
    grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring=scoring, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    # Evaluate model on test data
    y_pred = grid_search.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

    print(f"\n✅ Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"📊 Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R²: {r2:.4f}\n")

    # Save the model
    if save_model:
        filename = f"./checkpoints/{model_name}_best_model.pkl"
        joblib.dump(grid_search.best_estimator_, filename)
        print(f"💾 Model saved: {filename}")

    return results

def plot_mlp_feature_importance(mlp_model, X_test_scaled, y_test_scaled, feature_names, target_names, num_repeats=10):
    print("🔍 Computing feature importance for each target variable separately...")

    num_targets = y_test_scaled.shape[1]  # Number of target variables
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Arrange in 2x2 grid
    axes = axes.flatten()  # Flatten axes for easy iteration

    for i in range(num_targets):
        print(f"➡ Computing feature importance for target: {target_names[i]}")

        # Define a function to predict only for the i-th target
        model_for_target = lambda X: mlp_model.predict(X)[:, i]

        # Compute permutation importance separately for each target variable
        result = permutation_importance(
            model_for_target, X_test_scaled, y_test_scaled[:, i], n_repeats=num_repeats, random_state=42, scoring="r2"
        )

        importance_scores = result.importances_mean  # Average importance per feature
        sorted_indices = np.argsort(importance_scores)[::-1]  # Sort descending
        sorted_importance = importance_scores[sorted_indices]
        sorted_features = np.array(feature_names)[sorted_indices]

        # Plot for each target
        axes[i].barh(sorted_features, sorted_importance, color="royalblue")
        axes[i].set_xlabel("Importance Score")
        axes[i].set_ylabel("Feature")
        axes[i].set_title(f"Feature Importance for {target_names[i]}")
        axes[i].invert_yaxis()  # Most important feature on top
        axes[i].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


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
        verbose=3,
        n_iter=n_iter if use_random_search else None  # n_iter only applies to RandomizedSearchCV
    )

    # Fit search
    search.fit(X_train, y_train)

    # Evaluate model on test data
    y_pred = search.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

    print(f"\n✅ Best Parameters for {model_name}: {search.best_params_}")
    print(f"📊 Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R²: {r2:.4f}\n")

    # Save the model
    if save_model:
        filename = f"./checkpoints/{model_name}_best_model.pkl"
        joblib.dump(search.best_estimator_, filename)
        print(f"💾 Model saved: {filename}")

    return results

