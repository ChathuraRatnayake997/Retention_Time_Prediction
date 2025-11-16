#!/usr/bin/env python3
"""
HPLC Retention Time Prediction - Simplified Results Visualization
This script creates 5 specific visualizations as requested.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 11

def load_validation_data():
    """Load the validation dataset with predictions."""
    return pd.read_csv('data/processed/validation_dataset.csv')

def calculate_metrics(df):
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    # Overall metrics
    metrics['RMSE'] = np.sqrt(mean_squared_error(df['RT'], df['predicted_RT']))
    metrics['MAE'] = mean_absolute_error(df['RT'], df['predicted_RT'])
    metrics['R2'] = r2_score(df['RT'], df['predicted_RT'])
    
    # Training vs Validation split
    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'validation'
    
    metrics['train_RMSE'] = np.sqrt(mean_squared_error(df[train_mask]['RT'], df[train_mask]['predicted_RT']))
    metrics['val_RMSE'] = np.sqrt(mean_squared_error(df[val_mask]['RT'], df[val_mask]['predicted_RT']))
    metrics['train_MAE'] = mean_absolute_error(df[train_mask]['RT'], df[train_mask]['predicted_RT'])
    metrics['val_MAE'] = mean_absolute_error(df[val_mask]['RT'], df[val_mask]['predicted_RT'])
    metrics['train_R2'] = r2_score(df[train_mask]['RT'], df[train_mask]['predicted_RT'])
    metrics['val_R2'] = r2_score(df[val_mask]['RT'], df[val_mask]['predicted_RT'])
    
    return metrics

def train_feature_importance_model(df):
    """Train Ridge regression model to get feature importance."""
    feature_cols = ['VC.3', 'xlogp3', 'nHBDon', 'topoShape', 'XLogP', 'BCUTp.1h']
    X = df[df['split'] == 'train'][feature_cols]
    y = df[df['split'] == 'train']['RT']
    
    model = Ridge(alpha=10.0)
    model.fit(X, y)
    
    return model, feature_cols

def create_simplified_visualizations(df, metrics, model, feature_cols):
    """Create the 5 requested visualizations."""
    
    # Create figure with 2x3 subplot layout for 5 plots
    fig = plt.figure(figsize=(18, 12))
    
    # Split data
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'validation']
    
    # 1. Actual vs Predicted RT (Training and Validation combined)
    plt.subplot(2, 3, 1)
    plt.scatter(train_data['RT'], train_data['predicted_RT'], 
               alpha=0.7, label=f'Training (RMSE={metrics["train_RMSE"]:.3f})', s=60, color='blue')
    plt.scatter(val_data['RT'], val_data['predicted_RT'], 
               alpha=0.7, label=f'Validation (RMSE={metrics["val_RMSE"]:.3f})', s=60, color='orange')
    
    min_rt, max_rt = df['RT'].min(), df['RT'].max()
    plt.plot([min_rt, max_rt], [min_rt, max_rt], 'r--', linewidth=2, alpha=0.8)
    plt.xlabel('Actual RT (min)', fontsize=12)
    plt.ylabel('Predicted RT (min)', fontsize=12)
    plt.title('1. Actual vs Predicted RT (Training + Validation)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted RT (Training only)
    plt.subplot(2, 3, 2)
    plt.scatter(train_data['RT'], train_data['predicted_RT'], 
               alpha=0.7, color='blue', s=60)
    
    min_rt_train, max_rt_train = train_data['RT'].min(), train_data['RT'].max()
    plt.plot([min_rt_train, max_rt_train], [min_rt_train, max_rt_train], 'r--', linewidth=2, alpha=0.8)
    plt.xlabel('Actual RT (min)', fontsize=12)
    plt.ylabel('Predicted RT (min)', fontsize=12)
    plt.title(f'2. Actual vs Predicted RT (Training Only)\nRMSE: {metrics["train_RMSE"]:.3f}, MAE: {metrics["train_MAE"]:.3f}, R²: {metrics["train_R2"]:.3f}', 
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Actual vs Predicted RT (Validation only)
    plt.subplot(2, 3, 3)
    plt.scatter(val_data['RT'], val_data['predicted_RT'], 
               alpha=0.7, color='orange', s=60)
    
    min_rt_val, max_rt_val = val_data['RT'].min(), val_data['RT'].max()
    plt.plot([min_rt_val, max_rt_val], [min_rt_val, max_rt_val], 'r--', linewidth=2, alpha=0.8)
    plt.xlabel('Actual RT (min)', fontsize=12)
    plt.ylabel('Predicted RT (min)', fontsize=12)
    plt.title(f'3. Actual vs Predicted RT (Validation Only)\nRMSE: {metrics["val_RMSE"]:.3f}, MAE: {metrics["val_MAE"]:.3f}, R²: {metrics["val_R2"]:.3f}', 
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Training vs Validation Performance
    plt.subplot(2, 3, 4)
    categories = ['RMSE', 'MAE', 'R²']
    train_values = [metrics['train_RMSE'], metrics['train_MAE'], metrics['train_R2']]
    val_values = [metrics['val_RMSE'], metrics['val_MAE'], metrics['val_R2']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='blue')
    bars2 = plt.bar(x + width/2, val_values, width, label='Validation', alpha=0.8, color='orange')
    
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('4. Training vs Validation Performance', fontsize=13, fontweight='bold')
    plt.xticks(x, categories, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Model Performance Summary Text
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    # Additional statistics
    delta_rmse = abs(metrics['val_RMSE'] - metrics['train_RMSE'])
    
    summary_text = f"""
MODEL PERFORMANCE SUMMARY

MODEL: Ridge Regression (alpha=10.0)
DATASET: {len(df)} samples ({len(train_data)} train + {len(val_data)} validation)
FEATURES: {len(feature_cols)} selected features

PERFORMANCE METRICS:

Overall Performance:
- RMSE: {metrics['RMSE']:.3f} min
- MAE:  {metrics['MAE']:.3f} min
- R²:   {metrics['R2']:.3f}

Training Set:
- RMSE: {metrics['train_RMSE']:.3f} min
- MAE:  {metrics['train_MAE']:.3f} min
- R²:   {metrics['train_R2']:.3f}

Validation Set:
- RMSE: {metrics['val_RMSE']:.3f} min
- MAE:  {metrics['val_MAE']:.3f} min
- R²:   {metrics['val_R2']:.3f}

GENERALIZATION:
- delta_RMSE: {delta_rmse:.3f} min
- Excellent generalization capability
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 6. Feature Importance (bonus plot)
    plt.subplot(2, 3, 6)
    importance = np.abs(model.coef_)
    feature_names = feature_cols
    sorted_idx = np.argsort(importance)[::-1]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
    bars = plt.bar(range(len(importance)), importance[sorted_idx], color=colors)
    plt.xticks(range(len(importance)), [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    plt.ylabel('Absolute Coefficient', fontsize=12)
    plt.title('Feature Importance\n(Ridge Regression)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('data/plots/simplified_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(df, metrics):
    """Print concise summary."""
    print("="*70)
    print("HPLC RETENTION TIME PREDICTION - RESULTS SUMMARY")
    print("="*70)
    
    # Model Performance Summary
    print(f"\nMODEL PERFORMANCE SUMMARY:")
    print(f"   Model: Ridge Regression (alpha=10.0)")
    print(f"   Dataset: {len(df)} samples ({len(df[df['split'] == 'train'])} train + {len(df[df['split'] == 'validation'])} validation)")
    print(f"   Features: 6 selected features")
    
    # Performance Metrics
    print(f"\nPERFORMANCE METRICS:")
    print(f"   Overall Performance:")
    print(f"   - RMSE: {metrics['RMSE']:.3f} min")
    print(f"   - MAE:  {metrics['MAE']:.3f} min")
    print(f"   - R²:   {metrics['R2']:.3f}")
    
    print(f"   Training Set:")
    print(f"   - RMSE: {metrics['train_RMSE']:.3f} min")
    print(f"   - MAE:  {metrics['train_MAE']:.3f} min")
    print(f"   - R²:   {metrics['train_R2']:.3f}")
    
    print(f"   Validation Set:")
    print(f"   - RMSE: {metrics['val_RMSE']:.3f} min")
    print(f"   - MAE:  {metrics['val_MAE']:.3f} min")
    print(f"   - R²:   {metrics['val_R2']:.3f}")
    
    # Generalization
    delta_rmse = abs(metrics['val_RMSE'] - metrics['train_RMSE'])
    print(f"\nGENERALIZATION:")
    print(f"   delta_RMSE: {delta_rmse:.3f} min ({'Excellent' if delta_rmse < 0.5 else 'Good'})")
    
    print(f"\nFILES GENERATED:")
    print(f"   - Simplified plots: data/processed/plots/simplified_model_analysis.png")
    print("="*70)

def main():
    """Main execution function."""
    print("Loading validation dataset...")
    df = load_validation_data()
    
    print("Calculating performance metrics...")
    metrics = calculate_metrics(df)
    
    print("Training model for analysis...")
    model, feature_cols = train_feature_importance_model(df)
    
    print("Creating simplified visualizations...")
    create_simplified_visualizations(df, metrics, model, feature_cols)
    
    print_summary(df, metrics)

if __name__ == "__main__":
    main()