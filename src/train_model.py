import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI issues
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from src.data_preprocessing import clean_target_and_duplicates
from src.feature_engineering import build_preprocessor
from src.utils import get_project_root, save_pipeline, setup_logging

def train_and_evaluate():
    logger = setup_logging()
    project_root = get_project_root()
    
    # Define paths
    data_path = project_root / "data" / "IMDb Movies India.csv"
    model_path = project_root / "models" / "movie_rating_model.pkl"
    results_path = project_root / "reports" / "model_results.csv"
    figures_dir = project_root / "reports" / "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    logger.info(f"Loading dataset from {data_path}...")
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}!")
        return
        
    df = pd.read_csv(data_path, encoding='latin1')
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Pre-cleaning: drop target nulls and duplicate rows
    df_clean = clean_target_and_duplicates(df)
    logger.info(f"Dataset shape after removing target nulls and duplicates: {df_clean.shape}")
    
    # Separate features and target
    X = df_clean.drop(columns=['Rating'])
    y = df_clean['Rating']
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training set records: {X_train.shape[0]}")
    logger.info(f"Testing set records: {X_test.shape[0]}")
    
    # Define candidate models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Hist Gradient Boosting': HistGradientBoostingRegressor(random_state=42)
    }
    
    # Try adding XGBoost if available
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(random_state=42, n_jobs=-1)
        logger.info("XGBoost is available and added to candidate models.")
    except ImportError:
        logger.info("XGBoost not available. Skipping.")
        
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results_summary = []
    
    # Evaluate models using Cross-Validation
    logger.info("Starting Cross-Validation comparisons...")
    for name, model in models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # We evaluate using MAE, RMSE, and R2
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={
                'neg_mean_absolute_error': 'neg_mean_absolute_error',
                'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
                'r2': 'r2'
            },
            n_jobs=-1,
            error_score='raise'
        )
        
        cv_mae = -scores['test_neg_mean_absolute_error'].mean()
        cv_rmse = -scores['test_neg_root_mean_squared_error'].mean()
        cv_r2 = scores['test_r2'].mean()
        
        logger.info(f"{name} - CV MAE: {cv_mae:.3f} | CV RMSE: {cv_rmse:.3f} | CV R2: {cv_r2:.3f}")
        cv_results_summary.append({
            'Model': name,
            'CV_MAE': cv_mae,
            'CV_RMSE': cv_rmse,
            'CV_R2': cv_r2
        })
        
    cv_df = pd.DataFrame(cv_results_summary)
    
    # Select best model based on CV RMSE
    best_model_name = cv_df.sort_values(by='CV_RMSE').iloc[0]['Model']
    logger.info(f"Best model based on Cross-Validation RMSE: {best_model_name}")
    
    # Hyperparameter Tuning for the Best Model
    logger.info(f"Tuning hyperparameters for {best_model_name}...")
    best_base_model = models[best_model_name]
    
    # We will build parameter grids for the tuning process
    param_grids = {
        'Linear Regression': {},
        'Ridge Regression': {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]},
        'Random Forest': {
            'regressor__n_estimators': [50, 100, 150],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'regressor__n_estimators': [50, 100, 150],
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__max_depth': [3, 5]
        },
        'Hist Gradient Boosting': {
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__max_iter': [100, 150],
            'regressor__max_depth': [5, 10, None]
        },
        'XGBoost': {
            'regressor__n_estimators': [50, 100, 150],
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7]
        }
    }
    
    param_grid = param_grids.get(best_model_name, {})
    
    # Build complete tuning pipeline
    tune_preprocessor = build_preprocessor()
    tune_pipeline = Pipeline(steps=[
        ('preprocessor', tune_preprocessor),
        ('regressor', best_base_model)
    ])
    
    if param_grid:
        grid_search = GridSearchCV(
            tune_pipeline, param_grid, cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
    else:
        logger.info("No parameters to tune for this model. Fitting default parameters.")
        tune_pipeline.fit(X_train, y_train)
        best_pipeline = tune_pipeline
        
    # Evaluate all models on Test Set
    logger.info("Evaluating all candidate models on Test Set...")
    test_results = []
    
    for name, model in models.items():
        # Fit base model on full training set
        model_preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[
            ('preprocessor', model_preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        test_results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        
    # Add tuned model results to comparison
    y_pred_tuned = best_pipeline.predict(X_test)
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    r2_tuned = r2_score(y_test, y_pred_tuned)
    
    test_results.append({
        'Model': f"{best_model_name} (Tuned)",
        'MAE': mae_tuned,
        'RMSE': rmse_tuned,
        'R2': r2_tuned
    })
    
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved model comparison results to {results_path}")
    print("\n--- Model Comparison Table ---")
    print(results_df.to_string(index=False))
    
    # Save the best tuned pipeline
    logger.info(f"Saving best tuned pipeline model...")
    save_pipeline(best_pipeline, model_path)
    
    # Output final metrics report format
    print(f"\nBest Model: {best_model_name} (Tuned)")
    print(f"MAE: {mae_tuned:.2f}")
    print(f"RMSE: {rmse_tuned:.2f}")
    print(f"R²: {r2_tuned:.2f}")
    
    # --- Generate Plots ---
    logger.info("Generating evaluation charts...")
    sns.set_theme(style="whitegrid")
    
    # 1. Model Comparison Chart
    plt.figure(figsize=(10, 6))
    base_models_res = results_df[~results_df['Model'].str.contains("Tuned")]
    ax = sns.barplot(x='Model', y='RMSE', data=base_models_res, palette='viridis')
    plt.title('Model Comparison (RMSE on Test Set)', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE (Lower is Better)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=300)
    plt.close()
    
    # 2. Actual vs Predicted plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_tuned, alpha=0.4, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted Rating ({best_model_name} Tuned)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual IMDb Rating', fontsize=12)
    plt.ylabel('Predicted IMDb Rating', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted.png", dpi=300)
    plt.close()
    
    # 3. Residual plot
    residuals = y_test - y_pred_tuned
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_tuned, residuals, alpha=0.4, color='darkorange')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.title('Residuals vs Predicted Rating', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Rating', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "residuals_plot.png", dpi=300)
    plt.close()
    
    # 4. Prediction error distribution (Residual Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='crimson', bins=30)
    plt.title('Distribution of Prediction Errors (Residuals)', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "prediction_error_distribution.png", dpi=300)
    plt.close()
    
    # --- Feature Importance / Model Interpretability ---
    logger.info("Calculating feature importance using Permutation Importance...")
    # Permutation importance calculates importance by shuffling features on the test set
    result_importances = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    importances_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': result_importances.importances_mean,
        'Std': result_importances.importances_std
    }).sort_values(by='Importance', ascending=False)
    
    logger.info("Permutation Feature Importance (on Test Set):")
    print(importances_df.to_string(index=False))
    print("\n*Note: Feature importance shows model association, not causation.*")
    
    # Save Feature Importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances_df, palette='magma')
    plt.title('Permutation Feature Importance (on Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Mean Importance Score (drop in R² when shuffled)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=300)
    plt.close()
    
    logger.info("All evaluation charts saved to reports/figures/")

if __name__ == '__main__':
    train_and_evaluate()
