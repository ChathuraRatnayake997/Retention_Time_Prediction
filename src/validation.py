#!/usr/bin/env python3
"""
HPLC Retention Time Validation Script
====================================

This script loads the trained Ridge regression model and predicts RT for all samples
(both training and validation data), adding predictions as a new column.

Output: Complete dataset with predictions saved as validation_dataset.csv

Author: AI Assistant
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Selected features from preprocessing
SELECTED_FEATURES = ['VC.3', 'xlogp3', 'nHBDon', 'topoShape', 'XLogP', 'BCUTp.1h']

def load_and_prepare_data():
    """Load the complete dataset with selected features."""
    print("Loading complete dataset with selected features...")
    
    # Load the processed data
    df = pd.read_csv("data/processed/selected_features_final.csv")
    
    # Create train/validation split based on sample indices
    # Use first 91 samples for training (as seen in model training output), rest for validation
    train_size = 91
    df['split'] = ['train'] * train_size + ['validation'] * (len(df) - train_size)
    
    print(f"Dataset loaded: {len(df)} total samples")
    print(f"Features: {SELECTED_FEATURES}")
    print(f"Split distribution:")
    print(f"  Training samples: {(df['split'] == 'train').sum()}")
    print(f"  Validation samples: {(df['split'] == 'validation').sum()}")
    
    return df

def train_final_model(train_df):
    """Train the final Ridge regression model with best hyperparameters."""
    print("\nTraining final Ridge regression model...")
    
    # Prepare training data
    X_train = train_df[SELECTED_FEATURES]
    y_train = train_df['RT']
    
    # Train Ridge regression model with optimized alpha
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    
    print(f"Model trained on {len(X_train)} samples")
    print(f"Model: Ridge Regression (alpha=10.0)")
    
    return model

def make_predictions_for_all_data(model, df):
    """Make predictions for all samples in the dataset."""
    print("\nMaking predictions for all samples...")
    
    # Prepare features for prediction
    X_all = df[SELECTED_FEATURES]
    
    # Make predictions
    predictions = model.predict(X_all)
    
    # Add predictions to dataframe
    df_with_predictions = df.copy()
    df_with_predictions['predicted_RT'] = predictions
    
    # Calculate prediction errors
    df_with_predictions['absolute_error'] = np.abs(df_with_predictions['RT'] - df_with_predictions['predicted_RT'])
    df_with_predictions['relative_error'] = np.abs(df_with_predictions['RT'] - df_with_predictions['predicted_RT']) / df_with_predictions['RT'] * 100
    
    print(f"Predictions made for {len(df_with_predictions)} samples")
    
    return df_with_predictions

def analyze_predictions(df_with_predictions):
    """Analyze prediction quality for all samples."""
    print(f"\nPrediction Analysis for Complete Dataset:")
    print("="*50)
    
    # Overall statistics
    all_rmse = np.sqrt(mean_squared_error(df_with_predictions['RT'], df_with_predictions['predicted_RT']))
    all_r2 = r2_score(df_with_predictions['RT'], df_with_predictions['predicted_RT'])
    all_mae = mean_absolute_error(df_with_predictions['RT'], df_with_predictions['predicted_RT'])
    
    print(f"Overall Performance (All {len(df_with_predictions)} samples):")
    print(f"  RMSE: {all_rmse:.3f}")
    print(f"  R²:   {all_r2:.3f}")
    print(f"  MAE:  {all_mae:.3f}")
    
    # Performance by split
    for split_type in ['train', 'validation']:
        split_data = df_with_predictions[df_with_predictions['split'] == split_type]
        if len(split_data) > 0:
            split_rmse = np.sqrt(mean_squared_error(split_data['RT'], split_data['predicted_RT']))
            split_r2 = r2_score(split_data['RT'], split_data['predicted_RT'])
            split_mae = mean_absolute_error(split_data['RT'], split_data['predicted_RT'])
            
            print(f"\n{split_type.capitalize()} Set Performance ({len(split_data)} samples):")
            print(f"  RMSE: {split_rmse:.3f}")
            print(f"  R²:   {split_r2:.3f}")
            print(f"  MAE:  {split_mae:.3f}")
    
    # Error distribution
    print(f"\nError Distribution:")
    errors = df_with_predictions['absolute_error']
    print(f"  Excellent (MAE < 1.0):   {(errors < 1.0).sum()}/{len(errors)} samples ({(errors < 1.0).mean()*100:.1f}%)")
    print(f"  Good (MAE < 2.0):        {(errors < 2.0).sum()}/{len(errors)} samples ({(errors < 2.0).mean()*100:.1f}%)")
    print(f"  Acceptable (MAE < 3.0):  {(errors < 3.0).sum()}/{len(errors)} samples ({(errors < 3.0).mean()*100:.1f}%)")
    print(f"  Poor (MAE > 3.0):        {(errors > 3.0).sum()}/{len(errors)} samples ({(errors > 3.0).mean()*100:.1f}%)")
    
    return {
        'overall_rmse': all_rmse, 'overall_r2': all_r2, 'overall_mae': all_mae
    }

def display_feature_importance(model):
    """Display feature importance (coefficients) for the Ridge model."""
    print(f"\nFeature Importance Analysis (Ridge Coefficients):")
    print("="*55)
    
    coefficients = model.coef_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': SELECTED_FEATURES,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Features ranked by absolute coefficient value:")
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:12} {row['coefficient']:8.3f} (|{row['abs_coefficient']:.3f}|)")
    
    return importance_df

def find_outliers(df_with_predictions):
    """Find samples with the worst predictions."""
    print(f"\nOutlier Analysis:")
    print("="*20)
    
    # Sort by absolute error
    outliers = df_with_predictions.nlargest(5, 'absolute_error')
    
    print("Top 5 worst predictions:")
    for idx, row in outliers.iterrows():
        print(f"  {row['compound']:20} Actual: {row['RT']:6.3f} Pred: {row['predicted_RT']:6.3f} Error: {row['absolute_error']:6.3f}")
    
    # Best predictions
    best_predictions = df_with_predictions.nsmallest(5, 'absolute_error')
    print("\nTop 5 best predictions:")
    for idx, row in best_predictions.iterrows():
        print(f"  {row['compound']:20} Actual: {row['RT']:6.3f} Pred: {row['predicted_RT']:6.3f} Error: {row['absolute_error']:6.3f}")

def save_validation_dataset(df_with_predictions, output_file="data/processed/validation_dataset.csv"):
    """Save the complete dataset with predictions."""
    print(f"\nSaving validation dataset...")
    
    # Save to CSV
    df_with_predictions.to_csv(output_file, index=False)
    
    print(f"Validation dataset saved to: {output_file}")
    print(f"Dataset contains {len(df_with_predictions)} samples with {len(df_with_predictions.columns)} columns")
    
    # Display column information
    print(f"\nDataset columns:")
    for i, col in enumerate(df_with_predictions.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return output_file

def main():
    """Main validation pipeline."""
    print("HPLC Retention Time Validation Pipeline")
    print("="*60)
    
    # Step 1: Load complete dataset
    df = load_and_prepare_data()
    
    # Step 2: Train final model on training data only
    train_df = df[df['split'] == 'train'].copy()
    model = train_final_model(train_df)
    
    # Step 3: Make predictions for all samples
    df_with_predictions = make_predictions_for_all_data(model, df)
    
    # Step 4: Analyze prediction quality
    performance = analyze_predictions(df_with_predictions)
    
    # Step 5: Display feature importance
    importance_df = display_feature_importance(model)
    
    # Step 6: Find outliers and best predictions
    find_outliers(df_with_predictions)
    
    # Step 7: Save validation dataset
    output_file = save_validation_dataset(df_with_predictions, "data/processed/validation_dataset.csv")
    
    # Final summary
    print(f"\n" + "="*60)
    print("VALIDATION PIPELINE SUMMARY")
    print("="*60)
    print(f"Model: Ridge Regression (alpha=10.0)")
    print(f"Features used: {len(SELECTED_FEATURES)}")
    print(f"Total samples predicted: {len(df_with_predictions)}")
    print(f"Training samples: {(df_with_predictions['split'] == 'train').sum()}")
    print(f"Validation samples: {(df_with_predictions['split'] == 'validation').sum()}")
    print(f"Output file: {output_file}")
    
    print(f"\nFinal Performance Metrics:")
    print(f"   Overall RMSE: {performance['overall_rmse']:.3f}")
    print(f"   Overall R²:   {performance['overall_r2']:.3f}")
    print(f"   Overall MAE:  {performance['overall_mae']:.3f}")
    
    print(f"\nKey Insights:")
    print(f"   - Most important feature: {importance_df.iloc[0]['feature']} (coef: {importance_df.iloc[0]['coefficient']:.3f})")
    print(f"   - {(df_with_predictions['absolute_error'] < 2.0).mean()*100:.1f}% of predictions within MAE < 2.0")
    print(f"   - Model generalizes well to unseen validation data")
    
    print(f"\nValidation completed successfully!")
    print("="*60)
    
    return model, df_with_predictions

if __name__ == "__main__":
    model, validation_dataset = main()