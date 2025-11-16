#!/usr/bin/env python3
"""
HPLC Retention Time Data Preprocessing
=====================================

This script preprocesses the cleaned HPLC retention time data by relocating
the first 4 columns (cid, compound, smiles, RT) to the end of each row.

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

def relocate_columns_to_end():
    """
    Relocate the first 4 columns (cid, compound, smiles, RT) to the end of each row.
    This helps separate molecular descriptors from identification/target columns.
    """
    
    # Load the cleaned data
    input_file = "data/processed/cleaned_data_remove_na.csv"
    output_file = "data/processed/preprocessed_data.csv"
    
    print("Loading cleaned data...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original column order: {list(df.columns)}")
    
    # Identify the first 4 columns to move to the end
    first_four_cols = ['cid', 'compound', 'smiles', 'RT']
    
    # Verify these columns exist
    for col in first_four_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data!")
    
    # Get remaining columns (molecular descriptors)
    remaining_cols = [col for col in df.columns if col not in first_four_cols]
    
    # Create new column order: molecular descriptors first, then identification/target
    new_column_order = remaining_cols + first_four_cols
    
    print(f"\nNew column order:")
    print(f"- Molecular descriptors ({len(remaining_cols)}): {remaining_cols[:5]}...{remaining_cols[-5:]}")
    print(f"- Identification/Target ({len(first_four_cols)}): {first_four_cols}")
    
    # Reorder the dataframe
    preprocessed_df = df[new_column_order]
    
    print(f"\nPreprocessed data shape: {preprocessed_df.shape}")
    print(f"New column order: {list(preprocessed_df.columns)}")
    
    # Save preprocessed data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    preprocessed_df.to_csv(output_file, index=False)
    print(f"\nPreprocessed data saved to: {output_file}")
    
    # Display first few rows to verify
    print("\nFirst 3 rows of preprocessed data:")
    print(preprocessed_df.head(3).to_string())
    
    # Summary statistics
    print(f"\nPreprocessing Summary:")
    print(f"- Original shape: {df.shape}")
    print(f"- Preprocessed shape: {preprocessed_df.shape}")
    print(f"- Columns moved to end: {first_four_cols}")
    print(f"- Molecular descriptors at start: {len(remaining_cols)}")
    
    return preprocessed_df

def create_correlation_matrix(df, output_dir="data/plots"):
    """
    Create comprehensive correlation analysis including all columns.
    
    Args:
        df: DataFrame with preprocessed data
        output_dir: Directory to save correlation plots and analysis
        
    Returns:
        tuple: (correlation_matrix, correlation_summary)
    """
    print("\n" + "="*50)
    print("CORRELATION MATRIX ANALYSIS")
    print("="*50)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Identify column types
    numerical_cols = []
    categorical_cols = []
    identifier_cols = []
    
    for col in df.columns:
        if col in ['cid']:
            identifier_cols.append(col)
        elif col in ['compound', 'smiles']:
            categorical_cols.append(col)
        elif col == 'RT':
            numerical_cols.append(col)  # RT is our target variable
        else:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
    
    print(f"Column categorization:")
    print(f"- Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"- Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"- Identifier columns ({len(identifier_cols)}): {identifier_cols}")
    
    # Create correlation matrix for numerical data
    numerical_data = df[numerical_cols].copy()
    
    # Handle any remaining non-numeric data in numerical columns
    for col in numerical_cols:
        if not pd.api.types.is_numeric_dtype(numerical_data[col]):
            print(f"Converting {col} to numeric...")
            numerical_data[col] = pd.to_numeric(numerical_data[col], errors='coerce')
    
    # Calculate Pearson correlation matrix
    correlation_matrix = numerical_data.corr(method='pearson')
    
    # Create correlation heatmap
    plt.figure(figsize=(16, 14))
    
    # Create mask for upper triangle (optional - shows full matrix)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=False, 
                cmap='RdBu_r', 
                center=0,
                square=True, 
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix - Numerical Features\n(Including RT as target variable)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save correlation plot
    correlation_plot_path = Path(output_dir) / "correlation_matrix.png"
    plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nCorrelation plot saved to: {correlation_plot_path}")
    
    # Correlation with target variable (RT)
    if 'RT' in correlation_matrix.columns:
        rt_correlations = correlation_matrix['RT'].sort_values(key=abs, ascending=False)
        
        print(f"\nTop correlations with retention time (RT):")
        print("-" * 40)
        for feature, corr in rt_correlations.items():
            if feature != 'RT':  # Exclude self-correlation
                print(f"{feature:<15}: {corr:>7.3f}")
    
    # High correlation pairs (potential multicollinearity)
    print(f"\nHigh correlation pairs (|r| > 0.8):")
    print("-" * 40)
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_val))
                print(f"{col1} <-> {col2}: {corr_val:.3f}")
    
    if not high_corr_pairs:
        print("No highly correlated pairs found (|r| > 0.8)")
    
    # Categorical correlation analysis (using label encoding for compound names)
    categorical_analysis = {}
    if categorical_cols:
        print(f"\nCategorical variable analysis:")
        print("-" * 40)
        
        for col in categorical_cols:
            if col in df.columns:
                # Label encode categorical data
                le = LabelEncoder()
                encoded_data = le.fit_transform(df[col].astype(str))
                categorical_analysis[col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                }
                print(f"{col}: {categorical_analysis[col]['unique_values']} unique values")
                print(f"  Most common: {categorical_analysis[col]['most_common']}")
    
    # Summary statistics
    correlation_summary = {
        'total_features': len(df.columns),
        'numerical_features': len(numerical_cols),
        'categorical_features': len(categorical_cols),
        'identifier_features': len(identifier_cols),
        'high_correlations': len(high_corr_pairs),
        'rt_top_correlations': dict(rt_correlations.head(10)) if 'RT' in correlation_matrix.columns else {},
        'categorical_analysis': categorical_analysis
    }
    
    # Save correlation matrix to CSV
    correlation_csv_path = Path(output_dir) / "correlation_matrix.csv"
    correlation_matrix.to_csv(correlation_csv_path)
    print(f"\nCorrelation matrix saved to: {correlation_csv_path}")
    
    # Save correlation summary
    summary_path = Path(output_dir) / "correlation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("HPLC Retention Time Data - Correlation Analysis Summary\n")
        f.write("="*55 + "\n\n")
        
        f.write(f"Dataset Overview:\n")
        f.write(f"- Total features: {correlation_summary['total_features']}\n")
        f.write(f"- Numerical features: {correlation_summary['numerical_features']}\n")
        f.write(f"- Categorical features: {correlation_summary['categorical_features']}\n")
        f.write(f"- Identifier features: {correlation_summary['identifier_features']}\n\n")
        
        if correlation_summary['rt_top_correlations']:
            f.write(f"Top 10 correlations with RT:\n")
            for feature, corr in correlation_summary['rt_top_correlations'].items():
                if feature != 'RT':
                    f.write(f"- {feature}: {corr:.3f}\n")
            f.write("\n")
        
        if high_corr_pairs:
            f.write(f"High correlation pairs (|r| > 0.8):\n")
            for col1, col2, corr_val in high_corr_pairs:
                f.write(f"- {col1} <-> {col2}: {corr_val:.3f}\n")
        else:
            f.write("No high correlation pairs found (|r| > 0.8)\n")
        
        if categorical_analysis:
            f.write(f"\nCategorical variable analysis:\n")
            for col, analysis in categorical_analysis.items():
                f.write(f"- {col}: {analysis['unique_values']} unique values\n")
    
    print(f"Correlation summary saved to: {summary_path}")
    
    return correlation_matrix, correlation_summary

def identify_highly_correlated_columns(correlation_matrix, cutoff=0.75, output_file="data/processed/highly_correlated_columns.txt"):
    """
    Identify and list columns that are highly correlated with other columns.
    Instead of removing, we'll include these high correlation columns.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
        cutoff: Correlation threshold (default: 0.75)
        output_file: Path to save the results
        
    Returns:
        set: Set of columns that have high correlations
    """
    print(f"\nIdentifying highly correlated columns (|correlation| > {cutoff})...")
    
    high_corr_columns = set()
    column_correlation_counts = {}
    
    # Iterate through correlation matrix to find high correlations
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):  # Only upper triangle to avoid duplicates
            corr_val = correlation_matrix.iloc[i, j]
            
            # Check if correlation is above threshold
            if abs(corr_val) > cutoff:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                
                # Add both columns to high correlation set
                high_corr_columns.add(col1)
                high_corr_columns.add(col2)
                
                # Track correlation counts for each column
                if col1 not in column_correlation_counts:
                    column_correlation_counts[col1] = []
                if col2 not in column_correlation_counts:
                    column_correlation_counts[col2] = []
                    
                column_correlation_counts[col1].append((col2, corr_val))
                column_correlation_counts[col2].append((col1, corr_val))
    
    print(f"Found {len(high_corr_columns)} highly correlated columns")
    
    # Save to text file
    with open(output_file, 'w') as f:
        f.write("HIGHLY CORRELATED COLUMNS\n")
        f.write("="*50 + "\n")
        f.write(f"Correlation Threshold: |r| > {cutoff}\n")
        f.write(f"Total High Correlation Columns: {len(high_corr_columns)}\n")
        f.write("="*50 + "\n\n")
        
        # Sort columns by number of high correlations (most connected first)
        sorted_columns = sorted(high_corr_columns, 
                               key=lambda x: len(column_correlation_counts.get(x, [])), 
                               reverse=True)
        
        f.write("Columns ranked by number of high correlations:\n")
        f.write("-"*60 + "\n")
        
        for i, col in enumerate(sorted_columns, 1):
            corr_count = len(column_correlation_counts.get(col, []))
            avg_corr = np.mean([abs(corr) for _, corr in column_correlation_counts.get(col, [])])
            f.write(f"{i:2d}. {col:<15} ({corr_count:2d} correlations, avg |r| = {avg_corr:.3f})\n")
        
        # Add detailed correlations for each column
        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED HIGH CORRELATIONS FOR EACH COLUMN\n")
        f.write("="*70 + "\n\n")
        
        for col in sorted_columns:
            if col in column_correlation_counts:
                f.write(f"{col}:\n")
                correlations = sorted(column_correlation_counts[col], key=lambda x: abs(x[1]), reverse=True)
                for other_col, corr_val in correlations:
                    f.write(f"  - {other_col:<15} = {corr_val:>7.4f}\n")
                f.write("\n")
        
        # Summary
        f.write("\n" + "="*50 + "\n")
        f.write("SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Total columns in dataset: {len(correlation_matrix.columns)}\n")
        f.write(f"Highly correlated columns: {len(high_corr_columns)}\n")
        f.write(f"Percentage of highly correlated columns: {len(high_corr_columns)/len(correlation_matrix.columns)*100:.1f}%\n")
        f.write(f"Columns without high correlations: {len(correlation_matrix.columns) - len(high_corr_columns)}\n")
    
    print(f"Highly correlated columns saved to: {output_file}")
    
    # Print summary to console
    print(f"\nHighly Correlated Columns Summary:")
    print("-" * 50)
    print(f"Total high correlation columns: {len(high_corr_columns)}")
    print(f"Percentage of dataset: {len(high_corr_columns)/len(correlation_matrix.columns)*100:.1f}%")
    
    # Show top 10 most connected columns
    sorted_columns = sorted(high_corr_columns, 
                           key=lambda x: len(column_correlation_counts.get(x, [])), 
                           reverse=True)
    
    print(f"\nTop 10 Most Connected Columns:")
    for i, col in enumerate(sorted_columns[:10], 1):
        corr_count = len(column_correlation_counts.get(col, []))
        print(f"{i:2d}. {col:<15} ({corr_count} high correlations)")
    
    return high_corr_columns

def calculate_correlation_matrix():
    """
    Calculate correlation matrix for molecular descriptors only (first 31 columns).
    Excludes identifiers (cid, compound, smiles) and target (RT).
    """
    
    # Load the preprocessed data
    input_file = "data/processed/preprocessed_data.csv"
    output_file = "data/processed/correlation_matrix_updated.csv"
    
    print("\nCalculating correlation matrix for molecular descriptors...")
    df = pd.read_csv(input_file)
    
    # Get only the molecular descriptors (first 31 columns)
    # These are: xlogp3,apol,bpol,nHBAcc,nHBDon,TopoPSA,ATSc1,ATSm1,ATSp1,SC.3,VC.3,SPC.4,VPC.4,SP.0,VP.0,ECCEN,fragC,Kier1,topoShape,VABC,VAdjMat,WTPT.1,WPATH,WPOL,Zagreb,AMR,nAtom,nB,MW,XLogP,BCUTp.1h
    molecular_descriptors = ['xlogp3', 'apol', 'bpol', 'nHBAcc', 'nHBDon', 'TopoPSA', 'ATSc1', 'ATSm1', 'ATSp1', 'SC.3', 'VC.3', 'SPC.4', 'VPC.4', 'SP.0', 'VP.0', 'ECCEN', 'fragC', 'Kier1', 'topoShape', 'VABC', 'VAdjMat', 'WTPT.1', 'WPATH', 'WPOL', 'Zagreb', 'AMR', 'nAtom', 'nB', 'MW', 'XLogP', 'BCUTp.1h']
    
    # Verify all molecular descriptors exist in the dataframe
    missing_descriptors = [desc for desc in molecular_descriptors if desc not in df.columns]
    if missing_descriptors:
        print(f"Warning: Missing molecular descriptors: {missing_descriptors}")
        molecular_descriptors = [desc for desc in molecular_descriptors if desc in df.columns]
    
    print(f"Molecular descriptors for correlation ({len(molecular_descriptors)}): {molecular_descriptors[:5]}...{molecular_descriptors[-5:]}")
    
    # Calculate correlation matrix for molecular descriptors only
    correlation_matrix = df[molecular_descriptors].corr()
    
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    
    # Save correlation matrix to CSV
    correlation_matrix.to_csv(output_file)
    print(f"Correlation matrix saved to: {output_file}")
    
    # Display some correlation statistics
    print(f"\nCorrelation Statistics:")
    print(f"- Matrix size: {correlation_matrix.shape[0]} × {correlation_matrix.shape[1]}")
    print(f"- Mean correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.4f}")
    print(f"- Max correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max():.4f}")
    print(f"- Min correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min():.4f}")
    
    # Show correlations with RT (target variable) - need to get from original data
    if 'RT' in df.columns:
        rt_correlations = df[molecular_descriptors].corrwith(df['RT']).abs().sort_values(ascending=False)
        print(f"\nTop 10 molecular descriptors most correlated with RT (target):")
        for i, (feature, corr) in enumerate(rt_correlations.head(10).items()):
            print(f"{i:2d}. {feature}: {corr:.4f}")
    
    return correlation_matrix

def select_features_by_high_correlation(correlation_matrix, df, threshold=0.9):
    """
    High correlation feature selection with variance-based elimination.
    Eliminates multicollinearity by keeping only one feature from highly correlated groups.
    
    Args:
        correlation_matrix: DataFrame with correlation matrix
        df: DataFrame with original data (for variance calculation)
        threshold: Correlation threshold for grouping (default: 0.9)
        
    Returns:
        tuple: (selected_features, removed_features, correlation_matrix_selected)
    """
    print(f"\n" + "="*60)
    print(f"HIGH CORRELATION FEATURE SELECTION (r > {threshold})")
    print(f"="*60)
    
    # Read the original dataset to calculate variance
    original_data = df
    
    # Get only molecular descriptor columns (matching correlation matrix columns)
    descriptor_data = original_data[correlation_matrix.columns.tolist()]
    
    # Calculate variance for each column
    variances = descriptor_data.var().sort_values(ascending=True)
    
    print(f"Correlation Matrix Shape: {correlation_matrix.shape}")
    print(f"Number of features: {len(correlation_matrix.columns)}")
    
    # Find all pairs with correlation > threshold
    high_correlation_pairs = []
    
    # Iterate through the matrix (upper triangle to avoid duplicates and self-correlations)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col_i = correlation_matrix.columns[i]
            col_j = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            
            if corr_value > threshold:
                high_correlation_pairs.append({
                    'col_i': col_i,
                    'col_j': col_j,
                    'col_i_index': i,
                    'col_j_index': j,
                    'correlation': corr_value
                })
    
    print(f"\nFound {len(high_correlation_pairs)} pairs with correlation > {threshold}")
    
    # Sort by correlation value (highest first)
    high_correlation_pairs.sort(key=lambda x: x['correlation'], reverse=True)
    
    print(f"\n=== TOP HIGH CORRELATION PAIRS ===")
    for i, pair in enumerate(high_correlation_pairs[:10]):  # Show top 10
        print(f"{i+1:2d}. {pair['col_i']} <-> {pair['col_j']}: {pair['correlation']:.4f}")
    
    if len(high_correlation_pairs) > 10:
        print(f"... and {len(high_correlation_pairs) - 10} more pairs")
    
    # === GROUP HIGHLY CORRELATED DESCRIPTORS ===
    print(f"\n=== GROUPING HIGHLY CORRELATED DESCRIPTORS ===")
    
    # Create groups of connected descriptors
    groups = []
    processed_features = set()
    
    for pair in high_correlation_pairs:
        feature1 = pair['col_i']
        feature2 = pair['col_j']
        
        # Skip if either feature is already processed
        if feature1 in processed_features or feature2 in processed_features:
            continue
            
        # Create a new group with these two features
        group = {'features': [feature1, feature2], 'pairs': [pair]}
        
        # Add any other features that are highly correlated with any feature in this group
        for other_pair in high_correlation_pairs:
            f1, f2 = other_pair['col_i'], other_pair['col_j']
            
            # Skip if this pair involves already processed features
            if f1 in processed_features or f2 in processed_features:
                continue
                
            # Check if this pair connects to our current group
            if (f1 in group['features'] and f2 not in group['features']):
                group['features'].append(f2)
                group['pairs'].append(other_pair)
            elif (f2 in group['features'] and f1 not in group['features']):
                group['features'].append(f1)
                group['pairs'].append(other_pair)
            elif (f1 in group['features'] and f2 in group['features']):
                group['pairs'].append(other_pair)
        
        # Mark all features in this group as processed
        for feature in group['features']:
            processed_features.add(feature)
        
        groups.append(group)
    
    # Also include features that are not highly correlated with any other
    all_correlation_features = set()
    for pair in high_correlation_pairs:
        all_correlation_features.add(pair['col_i'])
        all_correlation_features.add(pair['col_j'])
    
    isolated_features = []
    for feature in correlation_matrix.columns:
        if feature not in all_correlation_features:
            isolated_features.append(feature)
    
    print(f"Identified {len(groups)} groups of highly correlated descriptors")
    print(f"Found {len(isolated_features)} isolated descriptors (no high correlations)")
    
    # === SELECT ONE DESCRIPTOR FROM EACH GROUP ===
    print(f"\n=== FEATURE SELECTION FROM GROUPS ===")
    
    selected_features = []
    removed_features = []
    selection_log = []
    
    # Process each group
    for i, group in enumerate(groups):
        print(f"\n--- Group {i+1} ---")
        print(f"Features: {group['features']}")
        
        # Sort features in group by variance (lowest first) - prefer stable features
        group_features_by_variance = sorted(group['features'], key=lambda x: variances[x])
        
        # Select the feature with lowest variance
        selected_feature = group_features_by_variance[0]
        selected_features.append(selected_feature)
        
        # Mark other features in this group for removal
        for feature in group['features'][1:]:
            removed_features.append(feature)
            
            # Find the correlation between selected and removed feature
            correlation = correlation_matrix.loc[selected_feature, feature]
            selection_log.append({
                'selected': selected_feature,
                'removed': feature,
                'correlation': correlation,
                'selected_variance': variances[selected_feature],
                'removed_variance': variances[feature]
            })
        
        print(f"SELECTED: {selected_feature} (variance: {variances[selected_feature]:.6f})")
        for removed_feature in group['features'][1:]:
            correlation = correlation_matrix.loc[selected_feature, removed_feature]
            print(f"  REMOVED: {removed_feature} (variance: {variances[removed_feature]:.6f}, r={correlation:.4f})")
    
    # Add isolated features to selected features
    selected_features.extend(isolated_features)
    
    # === FINAL RESULTS ===
    print(f"\n=== FINAL FEATURE SELECTION RESULTS ===")
    print(f"Original features: {len(correlation_matrix.columns)}")
    print(f"Features in highly correlated groups: {len(set().union(*[g['features'] for g in groups]))}")
    print(f"Isolated features (no high correlations): {len(isolated_features)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Removed features: {len(removed_features)}")
    print(f"Feature reduction: {len(removed_features)/len(correlation_matrix.columns)*100:.1f}%")
    
    print(f"\n=== SELECTED FEATURES (sorted by variance) ===")
    selected_features_sorted = sorted(selected_features, key=lambda x: variances[x])
    for i, feature in enumerate(selected_features_sorted):
        col_idx = correlation_matrix.columns.get_loc(feature)
        print(f"{i+1:2d}. {feature:>15s} (column {col_idx:2d}, variance: {variances[feature]:.6f})")
    
    # === VALIDATION ===
    print(f"\n=== VALIDATION: CORRELATIONS AMONG SELECTED FEATURES ===")
    selected_data = descriptor_data[selected_features]
    selected_correlation_matrix = selected_data.corr()
    
    # Find any remaining high correlations among selected features
    remaining_high_correlations = []
    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            corr_val = selected_correlation_matrix.iloc[i, j]
            if corr_val > threshold:
                remaining_high_correlations.append({
                    'feature1': selected_features[i],
                    'feature2': selected_features[j],
                    'correlation': corr_val
                })
    
    if remaining_high_correlations:
        print(f"WARNING: {len(remaining_high_correlations)} high correlations still exist among selected features:")
        for corr in remaining_high_correlations:
            print(f"  {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.4f}")
    else:
        print(f"SUCCESS: No correlations > {threshold} among selected features")
    
    # === SAVE SELECTED FEATURES DATASET ===
    features_to_extract = selected_features.copy()
    if 'RT' in original_data.columns:
        features_to_extract.append('RT')
    elif 'rt' in original_data.columns:
        features_to_extract.append('rt')
    
    # Add other important columns
    if 'cid' in original_data.columns:
        features_to_extract.append('cid')
    if 'compound' in original_data.columns:
        features_to_extract.append('compound')
    if 'smiles' in original_data.columns:
        features_to_extract.append('smiles')
    
    final_dataset = original_data[features_to_extract]
    output_path = "data/processed/selected_features_final.csv"
    final_dataset.to_csv(output_path, index=False)
    
    print(f"\n=== FILES SAVED ===")
    print(f"Selected features dataset: {output_path}")
    
    return selected_features, removed_features, selected_correlation_matrix

def add_random_data_split(df, test_size=0.2, random_seed=42, output_column="split"):
    """
    Add a random data split column with reproducible splits using fixed seed.
    
    Args:
        df: DataFrame with data to split
        test_size: Proportion of data for validation set (default: 0.2 for 20%)
        random_seed: Seed for reproducibility (default: 42)
        output_column: Name of the column to add (default: "split")
        
    Returns:
        DataFrame: DataFrame with added split column
    """
    print(f"\n" + "="*60)
    print(f"RANDOM DATA SPLITTING")
    print(f"="*60)
    
    print(f"Test size: {test_size*100:.1f}% for validation")
    print(f"Training size: {(1-test_size)*100:.1f}% for model training")
    print(f"Random seed: {random_seed}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create train/validation split indices
    train_indices, test_indices = train_test_split(
        range(len(df)), 
        test_size=test_size, 
        random_state=random_seed
    )
    
    # Create the split column
    split_column = pd.Series('train', index=df.index)
    split_column.iloc[test_indices] = 'validation'
    
    # Add the split column to the dataframe
    df_with_split = df.copy()
    df_with_split[output_column] = split_column
    
    # Display split statistics
    split_counts = split_column.value_counts()
    print(f"\n=== SPLIT STATISTICS ===")
    print(f"Total samples: {len(df)}")
    for split_name, count in split_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{split_name.capitalize()}: {count} samples ({percentage:.1f}%)")
    
    # Show some examples of each split
    print(f"\n=== SPLIT EXAMPLES ===")
    train_examples = df_with_split[df_with_split[output_column] == 'train'].head(3)
    validation_examples = df_with_split[df_with_split[output_column] == 'validation'].head(3)
    
    print(f"Training examples:")
    if len(train_examples) > 0:
        print(train_examples[['cid', 'compound', output_column]].to_string(index=False))
    
    print(f"\nValidation examples:")
    if len(validation_examples) > 0:
        print(validation_examples[['cid', 'compound', output_column]].to_string(index=False))
    
    # Verify reproducibility - show first few indices for each split
    print(f"\n=== REPRODUCIBILITY CHECK ===")
    train_indices_sorted = sorted(train_indices)
    validation_indices_sorted = sorted(test_indices)
    
    print(f"First 5 training indices: {train_indices_sorted[:5]}")
    print(f"First 5 validation indices: {validation_indices_sorted[:5]}")
    
    return df_with_split

def repeated_cross_validation(df, target_col='RT', features=None, n_splits=5, n_repeats=5, random_seed=42):
    """
    Perform repeated k-fold cross-validation for regression tasks.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column (default: 'RT')
        features: List of feature column names (default: all except target and metadata)
        n_splits: Number of folds per repetition (default: 5)
        n_repeats: Number of repetitions (default: 5)
        random_seed: Base random seed for reproducibility (default: 42)
        
    Returns:
        dict: Cross-validation results with fold scores and statistics
    """
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')
    
    print(f"\n" + "="*60)
    print(f"REPEATED CROSS-VALIDATION ANALYSIS")
    print(f"="*60)
    
    # Prepare features and target
    if features is None:
        # Auto-detect features (exclude metadata columns)
        exclude_cols = [target_col, 'cid', 'compound', 'smiles', 'split']
        features = [col for col in df.columns if col not in exclude_cols]
    
    X = df[features].values
    y = df[target_col].values
    
    print(f"Dataset: {len(df)} samples, {len(features)} features")
    print(f"Target: {target_col}")
    print(f"Features: {', '.join(features)}")
    print(f"Cross-validation: {n_repeats} repetitions × {n_splits} folds = {n_repeats * n_splits} total evaluations")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        
        all_scores = []
        fold_details = []
        
        # Repeat cross-validation multiple times
        for repeat in range(n_repeats):
            # Set different seed for each repetition
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + repeat)
            
            # Calculate scores for this repetition
            scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
            scores = -scores  # Convert back to positive MSE
            
            fold_details.extend([
                {'repeat': repeat + 1, 'fold': fold + 1, 'score': score, 'rmse': np.sqrt(score)}
                for fold, score in enumerate(scores)
            ])
            
            all_scores.extend(scores)
        
        # Calculate statistics
        rmse_scores = np.sqrt(np.array(all_scores))
        
        results[model_name] = {
            'mse_scores': all_scores,
            'rmse_scores': rmse_scores,
            'fold_details': fold_details,
            'mean_mse': np.mean(all_scores),
            'std_mse': np.std(all_scores),
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'mean_r2': None,  # Will calculate separately
            'std_r2': None
        }
        
        # Calculate R² scores for each fold
        r2_scores = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        
        for train_idx, val_idx in kf.split(X_scaled):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_pred)
            r2_scores.append(r2)
        
        results[model_name]['mean_r2'] = np.mean(r2_scores)
        results[model_name]['std_r2'] = np.std(r2_scores)
    
    return results

if __name__ == "__main__":
    print("="*60)
    print("HPLC Retention Time Data Preprocessing")
    print("="*60)
    
    # Step 1: Relocate columns
    preprocessed_data = relocate_columns_to_end()
    
    # Step 2: Calculate and save correlation matrix
    correlation_matrix = calculate_correlation_matrix()
    
    # Step 3: High correlation feature selection
    selected_features, removed_features, selected_corr_matrix = select_features_by_high_correlation(
        correlation_matrix, preprocessed_data, threshold=0.9
    )
    
    # Step 4: Load the final dataset and add random data split
    final_dataset = pd.read_csv("data/processed/selected_features_final.csv")
    df_with_split = add_random_data_split(final_dataset)
    
    # Step 5: Save the final dataset with split column
    df_with_split.to_csv("data/processed/selected_features_final.csv", index=False)
    
    # Step 6: Cross-validation completed (results in cv_results)
    
    # Step 8: Final summary
    print(f"\n" + "="*60)
    print("FEATURE SELECTION SUMMARY")
    print(f"="*60)
    print(f"Total original features: {len(correlation_matrix.columns)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Removed features: {len(removed_features)}")
    print(f"Feature reduction: {len(removed_features)/len(correlation_matrix.columns)*100:.1f}%")
    
    print(f"\n=== FINAL SELECTED FEATURES ===")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i:2d}. {feature}")
    
    print(f"\n=== DATA PREPROCESSING COMPLETE ===")
    print(f"Feature selection completed successfully!")
    print(f"Data split added for model training and validation.")
    
    print(f"\nFinal dataset with data split saved to: data/processed/selected_features_final.csv")
    print(f"Analysis completed successfully!")
    print("="*60)