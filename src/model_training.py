#!/usr/bin/env python3
"""
HPLC Retention Time Prediction Model Training Script
====================================================

This script trains machine learning models to predict retention times (RT) 
using the selected features from the preprocessing pipeline.

Features used:
1. VC.3
2. xlogp3
3. nHBDon
4. topoShape
5. XLogP
6. BCUTp.1h

Author: AI Assistant
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Selected features from preprocessing
SELECTED_FEATURES = ['VC.3', 'xlogp3', 'nHBDon', 'topoShape', 'XLogP', 'BCUTp.1h']

def load_training_data(file_path="data/processed/selected_features_final.csv"):
    """Load and prepare training data."""
    print("Loading training data...")
    df = pd.read_csv(file_path)
    
    # Filter for training data only
    train_df = df[df['split'] == 'train'].copy()
    
    # Extract features and target
    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df['RT']
    
    print(f"Training data loaded: {len(train_df)} samples")
    print(f"Features: {SELECTED_FEATURES}")
    print(f"Target: RT (Retention Time)")
    
    return X_train, y_train, train_df

def evaluate_models(X_train, y_train, cv_folds=5):
    """Evaluate multiple regression models using cross-validation."""
    print(f"\nEvaluating models with {cv_folds}-fold cross-validation...")
    print("="*60)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # Store results
    results = {}
    
    # Perform cross-validation for each model
    for name, model in models.items():
        # Use negative MSE for cross-validation (sklearn convention)
        cv_scores = cross_val_score(model, X_train, y_train, 
                                  cv=cv_folds, 
                                  scoring='neg_mean_squared_error')
        
        # Convert to positive RMSE
        rmse_scores = np.sqrt(-cv_scores)
        
        # Calculate R² scores
        r2_scores = cross_val_score(model, X_train, y_train, 
                                  cv=cv_folds, 
                                  scoring='r2')
        
        # Calculate MAE scores
        mae_scores = cross_val_score(model, X_train, y_train, 
                                   cv=cv_folds, 
                                   scoring='neg_mean_absolute_error')
        mae_scores = -mae_scores
        
        results[name] = {
            'model': model,
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std()
        }
        
        print(f"{name:25} RMSE: {rmse_scores.mean():.3f}+/-{rmse_scores.std():.3f} "
              f"R²: {r2_scores.mean():.3f}+/-{r2_scores.std():.3f} "
              f"MAE: {mae_scores.mean():.3f}+/-{mae_scores.std():.3f}")
    
    return results

def hyperparameter_tuning(X_train, y_train, model_type='RandomForest', cv_folds=5):
    """Perform hyperparameter tuning for the specified model."""
    print(f"\nPerforming hyperparameter tuning for {model_type}...")
    print("="*60)
    
    if model_type == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_type == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif model_type == 'Ridge':
        model = Ridge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
    elif model_type == 'Lasso':
        model = Lasso()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
    else:
        print(f"Hyperparameter tuning not implemented for {model_type}")
        return None
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=cv_folds, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.3f}")
    
    return grid_search.best_estimator_

def train_final_model(X_train, y_train, model_type='LinearRegression'):
    """Train the final model with best hyperparameters."""
    print(f"\nTraining final {model_type} model...")
    print("="*60)
    
    if model_type == 'LinearRegression':
        model = LinearRegression()
    elif model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Training MAE: {train_mae:.3f}")
    
    return model

def feature_importance_analysis(model, feature_names):
    """Analyze feature importance for tree-based models."""
    print(f"\nFeature Importance Analysis:")
    print("="*40)
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top features by importance:")
        for idx, row in importance_df.iterrows():
            print(f"{row['feature']:12} {row['importance']:.3f}")
        
        return importance_df
    else:
        print("Feature importance not available for this model type.")
        return None

def make_predictions(model, X_test, feature_names):
    """Make predictions and display results."""
    predictions = model.predict(X_test)
    
    print(f"\nPrediction Summary:")
    print("="*30)
    print(f"Mean predicted RT: {predictions.mean():.3f}")
    print(f"Std predicted RT: {predictions.std():.3f}")
    print(f"Min predicted RT: {predictions.min():.3f}")
    print(f"Max predicted RT: {predictions.max():.3f}")
    
    return predictions

def main():
    """Main training pipeline."""
    print("HPLC Retention Time Prediction Model Training")
    print("="*60)
    
    # Load training data
    X_train, y_train, train_df = load_training_data()
    
    # Basic statistics
    print(f"\nTraining Data Statistics:")
    print(f"="*30)
    print(f"Number of samples: {len(X_train)}")
    print(f"Number of features: {len(SELECTED_FEATURES)}")
    print(f"RT range: {y_train.min():.3f} - {y_train.max():.3f}")
    print(f"RT mean: {y_train.mean():.3f}")
    print(f"RT std: {y_train.std():.3f}")
    
    # Check for missing values
    print(f"\nData Quality Check:")
    print(f"="*25)
    print(f"Missing values in features: {X_train.isnull().sum().sum()}")
    print(f"Missing values in target: {y_train.isnull().sum()}")
    
    # Evaluate multiple models
    cv_results = evaluate_models(X_train, y_train)
    
    # Find best model based on RMSE
    best_model_name = min(cv_results.keys(), key=lambda x: cv_results[x]['rmse_mean'])
    print(f"\nBest model based on CV RMSE: {best_model_name}")
    
    # Determine model type for final training
    if 'Random Forest' in best_model_name:
        final_model_type = 'RandomForest'
    elif 'Gradient Boosting' in best_model_name:
        final_model_type = 'GradientBoosting'
    elif 'Ridge' in best_model_name:
        final_model_type = 'Ridge'
    elif 'Lasso' in best_model_name:
        final_model_type = 'Lasso'
    else:
        final_model_type = 'LinearRegression'
    
    # Perform hyperparameter tuning for the best model
    if final_model_type != 'LinearRegression':
        best_model = hyperparameter_tuning(X_train, y_train, final_model_type)
    else:
        best_model = train_final_model(X_train, y_train, final_model_type)
    
    # If we did hyperparameter tuning, train final model with best params
    if hasattr(best_model, 'predict'):
        final_model = best_model
    else:
        final_model = train_final_model(X_train, y_train, final_model_type)
    
    # Feature importance analysis
    importance_df = feature_importance_analysis(final_model, SELECTED_FEATURES)
    
    # Final model summary
    print(f"\n" + "="*60)
    print("FINAL MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"Model type: {final_model_type}")
    print(f"Training samples: {len(X_train)}")
    print(f"Features used: {len(SELECTED_FEATURES)}")
    print(f"Feature names: {', '.join(SELECTED_FEATURES)}")
    
    if importance_df is not None:
        print(f"\nTop 3 most important features:")
        for idx, row in importance_df.head(3).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    
    print(f"\nModel training completed successfully!")
    print("="*60)
    
    return final_model, SELECTED_FEATURES

if __name__ == "__main__":
    model, features = main()