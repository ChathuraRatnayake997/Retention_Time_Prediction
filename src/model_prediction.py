#!/usr/bin/env python3
"""
HPLC Retention Time Prediction Script
=====================================

This script uses the trained model to make predictions on validation data
and evaluates the model performance.

Author: AI Assistant
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Selected features from preprocessing
SELECTED_FEATURES = ['VC.3', 'xlogp3', 'nHBDon', 'topoShape', 'XLogP', 'BCUTp.1h']

def load_data_and_train_model():
    """Load data and train the Ridge regression model."""
    print("Loading data and training Ridge regression model...")
    
    # Load the processed data
    df = pd.read_csv("data/processed/selected_features_final.csv")
    
    # Split into train and validation sets
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    
    # Prepare training data
    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df['RT']
    
    # Prepare validation data
    X_val = val_df[SELECTED_FEATURES]
    y_val = val_df['RT']
    
    # Train Ridge regression model with best hyperparameters
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Model: Ridge Regression (alpha=10.0)")
    
    return model, X_train, y_train, X_val, y_val, train_df, val_df

def evaluate_model_performance(model, X_train, y_train, X_val, y_val):
    """Evaluate model performance on training and validation sets."""
    print(f"\nModel Performance Evaluation:")
    print("="*50)
    
    # Training set performance
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    print(f"Training Set Performance:")
    print(f"  RMSE: {train_rmse:.3f}")
    print(f"  R²:   {train_r2:.3f}")
    print(f"  MAE:  {train_mae:.3f}")
    
    # Validation set performance
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    print(f"\nValidation Set Performance:")
    print(f"  RMSE: {val_rmse:.3f}")
    print(f"  R²:   {val_r2:.3f}")
    print(f"  MAE:  {val_mae:.3f}")
    
    # Model comparison
    print(f"\nModel Analysis:")
    print(f"  Overfitting check (RMSE difference): {abs(val_rmse - train_rmse):.3f}")
    if abs(val_rmse - train_rmse) < 0.5:
        print(f"  Good generalization - minimal overfitting")
    elif abs(val_rmse - train_rmse) < 1.0:
        print(f"  WARNING: Moderate overfitting detected")
    else:
        print(f"  ERROR: High overfitting - consider regularization")
    
    return {
        'train_rmse': train_rmse, 'train_r2': train_r2, 'train_mae': train_mae,
        'val_rmse': val_rmse, 'val_r2': val_r2, 'val_mae': val_mae,
        'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred
    }

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance for linear models."""
    print(f"\nFeature Importance Analysis (Coefficients):")
    print("="*55)
    
    coefficients = model.coef_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Features ranked by importance (absolute coefficient value):")
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:12} {row['coefficient']:8.3f} (|{row['abs_coefficient']:.3f}|)")
    
    return importance_df

def detailed_predictions_analysis(y_val, y_val_pred, val_df):
    """Detailed analysis of predictions."""
    print(f"\nDetailed Predictions Analysis:")
    print("="*35)
    
    # Prediction statistics
    print(f"Actual RT range: {y_val.min():.3f} - {y_val.max():.3f}")
    print(f"Predicted RT range: {y_val_pred.min():.3f} - {y_val_pred.max():.3f}")
    print(f"Mean absolute error per sample:")
    
    # Individual prediction errors
    val_errors = np.abs(y_val - y_val_pred)
    
    # Find best and worst predictions
    best_idx = np.argmin(val_errors)
    worst_idx = np.argmax(val_errors)
    
    print(f"  Best prediction (MAE={val_errors.iloc[best_idx]:.3f}):")
    print(f"    Actual: {y_val.iloc[best_idx]:.3f}, Predicted: {y_val_pred[best_idx]:.3f}")
    print(f"    Compound: {val_df.iloc[best_idx]['compound']}")
    
    print(f"  Worst prediction (MAE={val_errors.iloc[worst_idx]:.3f}):")
    print(f"    Actual: {y_val.iloc[worst_idx]:.3f}, Predicted: {y_val_pred[worst_idx]:.3f}")
    print(f"    Compound: {val_df.iloc[worst_idx]['compound']}")
    
    # Error distribution
    print(f"\nError Distribution:")
    print(f"  Samples with MAE < 1.0: {(val_errors < 1.0).sum()}/{len(val_errors)} ({(val_errors < 1.0).mean()*100:.1f}%)")
    print(f"  Samples with MAE < 2.0: {(val_errors < 2.0).sum()}/{len(val_errors)} ({(val_errors < 2.0).mean()*100:.1f}%)")
    print(f"  Samples with MAE > 3.0: {(val_errors > 3.0).sum()}/{len(val_errors)} ({(val_errors > 3.0).mean()*100:.1f}%)")

def save_predictions(model, X_val, val_df, output_file="data/processed/predictions.csv"):
    """Save predictions to CSV file."""
    # Make predictions
    y_val_pred = model.predict(X_val)
    
    # Create predictions dataframe
    predictions_df = val_df.copy()
    predictions_df['predicted_RT'] = y_val_pred
    predictions_df['absolute_error'] = np.abs(predictions_df['RT'] - predictions_df['predicted_RT'])
    predictions_df['relative_error'] = np.abs(predictions_df['RT'] - predictions_df['predicted_RT']) / predictions_df['RT'] * 100
    
    # Save to file
    predictions_df.to_csv(output_file, index=False)
    
    print(f"\nPredictions saved to: {output_file}")
    
    return predictions_df

def main():
    """Main prediction pipeline."""
    print("HPLC Retention Time Prediction and Evaluation")
    print("="*60)
    
    # Load data and train model
    model, X_train, y_train, X_val, y_val, train_df, val_df = load_data_and_train_model()
    
    # Evaluate model performance
    performance = evaluate_model_performance(model, X_train, y_train, X_val, y_val)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(model, SELECTED_FEATURES)
    
    # Detailed predictions analysis
    detailed_predictions_analysis(y_val, performance['y_val_pred'], val_df)
    
    # Save predictions
    predictions_df = save_predictions(model, X_val, val_df, "data/processed/predictions.csv")
    
    # Final summary
    print(f"\n" + "="*60)
    print("PREDICTION PIPELINE SUMMARY")
    print("="*60)
    print(f"Model: Ridge Regression")
    print(f"Features used: {len(SELECTED_FEATURES)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Best validation performance:")
    print(f"  RMSE: {performance['val_rmse']:.3f}")
    print(f"  R²:   {performance['val_r2']:.3f}")
    print(f"  MAE:  {performance['val_mae']:.3f}")
    print(f"\nTop 3 most important features:")
    for idx, row in importance_df.head(3).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.3f}")
    print(f"\nPrediction completed successfully!")
    print("="*60)
    
    return model, predictions_df

if __name__ == "__main__":
    model, predictions = main()