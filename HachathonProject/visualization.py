# 📌 Visualization
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
# Suppress warnings about non-interactive backend
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap
import io
import base64
import os
from datetime import datetime


def plot_target_distribution(y, title="Target Distribution", return_image=False, save_path=None):
    """Plot the distribution of the target variable."""
    plt.figure(figsize=(8, 6))
    # Sort by class index (0, 1) to ensure correct label assignment
    # Class 0 = "No" (No Depression), Class 1 = "Yes" (Depression)
    counts = y.value_counts().sort_index()
    
    # Assign labels and colors based on class value, not count order
    labels = []
    colors = []
    for class_val in counts.index:
        if class_val == 0:
            labels.append('No Depression (No)')
            colors.append('#2ecc71')  # Green for no depression
        elif class_val == 1:
            labels.append('Depression (Yes)')
            colors.append('#e74c3c')  # Red for depression
        else:
            labels.append(f'Class {class_val}')
            colors.append('#95a5a6')
    
    plt.pie(counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if return_image:
        return _fig_to_base64(save_path=save_path, filename="target_distribution.png")
    # Close figure instead of showing when using non-interactive backend
    if matplotlib.get_backend().lower() == 'agg':
        plt.close()
    else:
        try:
            plt.show()
        except Exception:
            plt.close()  # Fallback to close if show fails


def plot_feature_distributions(df, cat_cols, target, n_cols=3, return_image=False, save_path=None):
    """Plot distributions of categorical features by target."""
    # Exclude Name and ID from feature distributions (Name is for display only, ID is for merging only)
    excluded_cols = ['Name', 'ID']
    filtered_cat_cols = [col for col in cat_cols if col not in excluded_cols]
    
    # Additional safety check: ensure Name and ID are not in the dataframe columns we're using
    # Also verify the column exists in df before plotting
    valid_cat_cols = []
    for col in filtered_cat_cols:
        if col in df.columns and col not in excluded_cols:
            valid_cat_cols.append(col)
        elif col in excluded_cols:
            # Skip if somehow excluded columns made it through
            continue
    
    n_features = len(valid_cat_cols)
    if n_features == 0:
        print("Warning: No valid categorical features to plot after filtering out Name and ID")
        # Create an empty plot with a message
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No categorical features available\n(Name and ID excluded)', 
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.axis('off')
        plt.title('Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if return_image:
            return _fig_to_base64(save_path=save_path, filename="feature_distributions.png")
        if matplotlib.get_backend().lower() == 'agg':
            plt.close()
        else:
            try:
                plt.show()
            except Exception:
                plt.close()
        return
    
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Translate column names to English
    try:
        from gradio_helpers import translate_feature_name
    except ImportError:
        translate_feature_name = lambda x: x  # Fallback if translation not available
    
    for idx, col in enumerate(valid_cat_cols[:n_features]):
        ax = axes[idx]
        # Double-check that Name and ID are not being plotted
        if col in excluded_cols:
            continue
        crosstab = pd.crosstab(df[col], df[target])
        crosstab.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        # Translate column name for display
        translated_col = translate_feature_name(col)
        ax.set_title(f'{translated_col}', fontweight='bold')
        ax.set_xlabel(translated_col)
        ax.set_ylabel('Count')
        ax.legend(['No', 'Yes'])
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if return_image:
        return _fig_to_base64(save_path=save_path, filename="feature_distributions.png")
    # Close figure instead of showing when using non-interactive backend
    if matplotlib.get_backend().lower() == 'agg':
        plt.close()
    else:
        try:
            plt.show()
        except Exception:
            plt.close()  # Fallback to close if show fails


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix", return_image=False, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Depression', 'Depression'],
                yticklabels=['No Depression', 'Depression'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if return_image:
        return _fig_to_base64(save_path=save_path, filename="confusion_matrix.png")
    # Close figure instead of showing when using non-interactive backend
    if matplotlib.get_backend().lower() == 'agg':
        plt.close()
    else:
        try:
            plt.show()
        except Exception:
            plt.close()  # Fallback to close if show fails


def plot_roc_curve(y_test, y_proba, roc_auc, title="ROC Curve", return_image=False, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3498db', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if return_image:
        return _fig_to_base64(save_path=save_path, filename="roc_curve.png")
    # Close figure instead of showing when using non-interactive backend
    if matplotlib.get_backend().lower() == 'agg':
        plt.close()
    else:
        try:
            plt.show()
        except Exception:
            plt.close()  # Fallback to close if show fails


def plot_shap_summary(pipeline, X_test, cat_cols, max_display=10, save_path=None):
    """Plot SHAP summary for feature importance for both classes."""
    try:
        # Get the preprocessed features
        preprocessor = pipeline.named_steps['preprocess']
        X_test_processed = preprocessor.transform(X_test[:100])  # Use subset for speed
        
        # Get feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()
        
        # Get the model
        model = pipeline.named_steps['model']
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for both classes
        # For binary classification with XGBoost, shap_values() can return:
        # 1. A list with 2 arrays [class_0_shap, class_1_shap]
        # 2. A single array (typically for the positive class/class 1)
        shap_values_output = explainer.shap_values(X_test_processed)
        
        # Handle different return types
        if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
            # List format with both classes: [shap_values_class_0, shap_values_class_1]
            shap_values_class_0 = shap_values_output[0]
            shap_values_class_1 = shap_values_output[1]
        elif isinstance(shap_values_output, list) and len(shap_values_output) == 1:
            # Single element list - use it as class 1, calculate class 0
            shap_values_class_1 = shap_values_output[0]
            # For binary classification: SHAP values for class 0 = -SHAP values for class 1
            # (since probabilities sum to 1, their SHAP values are negated)
            shap_values_class_0 = -shap_values_class_1
        else:
            # Single array format: typically for positive class (class 1)
            # For binary classification, class 0 SHAP = -class 1 SHAP
            shap_values_class_1 = shap_values_output
            shap_values_class_0 = -shap_values_class_1
        
        # Determine number of features
        n_features = shap_values_class_1.shape[1]
        feature_names_to_use = feature_names[:n_features]
        
        # Clean feature names: remove "cat__" prefix and other preprocessing prefixes
        # Remove "cat__" prefix (can appear at start or after "__")
        feature_names_to_use = [name.replace('cat__', '').replace('num__', '') if 'cat__' in name or 'num__' in name else name for name in feature_names_to_use]
        # Also clean up any remaining "__" separators that might be left
        feature_names_to_use = [name.split('__')[-1] if '__' in name else name for name in feature_names_to_use]
        
        print(f"\n   Plotting SHAP values for Class 0 (No depression)...")
        # Plot summary for Class 0 (No depression)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_class_0, X_test_processed, 
                         feature_names=feature_names_to_use,
                         max_display=max_display, 
                         show=False)
        plt.title("SHAP Summary Plot - Class 0 (No Depression)", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        if save_path:
            _fig_to_base64(save_path=save_path, filename="shap_summary_class0.png")
        if matplotlib.get_backend().lower() == 'agg':
            plt.close()
        else:
            try:
                plt.show()
            except Exception:
                plt.close()
        
        # Plot bar plot for Class 0
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class_0, X_test_processed, 
                         feature_names=feature_names_to_use,
                         plot_type="bar", 
                         max_display=max_display,
                         show=False)
        plt.title("SHAP Feature Importance - Class 0 (No Depression)", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        if save_path:
            _fig_to_base64(save_path=save_path, filename="shap_bar_class0.png")
        if matplotlib.get_backend().lower() == 'agg':
            plt.close()
        else:
            try:
                plt.show()
            except Exception:
                plt.close()
        
        print(f"   Plotting SHAP values for Class 1 (Yes depression)...")
        # Plot summary for Class 1 (Yes depression)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_class_1, X_test_processed, 
                         feature_names=feature_names_to_use,
                         max_display=max_display,
                         show=False)
        plt.title("SHAP Summary Plot - Class 1 (Yes Depression)", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        if save_path:
            _fig_to_base64(save_path=save_path, filename="shap_summary_class1.png")
        if matplotlib.get_backend().lower() == 'agg':
            plt.close()
        else:
            try:
                plt.show()
            except Exception:
                plt.close()
        
        # Plot bar plot for Class 1
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class_1, X_test_processed, 
                         feature_names=feature_names_to_use,
                         plot_type="bar", 
                         max_display=max_display,
                         show=False)
        plt.title("SHAP Feature Importance - Class 1 (Yes Depression)", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        if save_path:
            _fig_to_base64(save_path=save_path, filename="shap_bar_class1.png")
        if matplotlib.get_backend().lower() == 'agg':
            plt.close()
        else:
            try:
                plt.show()
            except Exception:
                plt.close()
        
    except Exception as e:
        print(f"SHAP visualization error: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping SHAP plots...")


def plot_correlation_heatmap(df, target, return_image=False, save_path=None):
    """Plot correlation heatmap for features (handles both numerical and categorical)."""
    try:
        # Create a copy of dataframe for encoding
        df_encoded = df.copy()
        
        # Exclude ID and Name from correlation heatmap (ID is for merging only, Name is for display only)
        excluded_cols = ['ID', 'Name']
        
        # Get all feature columns (exclude target, ID, and Name)
        feature_cols = [col for col in df_encoded.columns if col != target and col not in excluded_cols]
        
        if len(feature_cols) == 0:
            # No features to plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No features available\nfor correlation analysis', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.axis('off')
            plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if return_image:
                return _fig_to_base64(save_path=save_path, filename="correlation_heatmap.png")
            if matplotlib.get_backend().lower() == 'agg':
                plt.close()
            else:
                try:
                    plt.show()
                except Exception:
                    plt.close()
            return
        
        # Encode categorical columns to numerical for correlation calculation
        # IMPORTANT: Create a new LabelEncoder for each column to avoid conflicts
        from sklearn.preprocessing import LabelEncoder
        
        # Encode categorical columns (each column gets its own encoder)
        for col in feature_cols:
            if df_encoded[col].dtype == 'object':
                try:
                    le = LabelEncoder()  # New encoder for each column
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str).fillna(''))
                except Exception as e:
                    # If encoding fails, try to convert to numeric directly
                    try:
                        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                    except Exception:
                        # If all else fails, drop the column
                        print(f"Warning: Could not encode column {col} for correlation heatmap: {e}")
                        if col in df_encoded.columns:
                            df_encoded = df_encoded.drop(columns=[col])
        
        # Select columns that are now numerical (after encoding)
        numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target, ID, and Name from numerical_cols if they're there
        for col_to_remove in [target] + excluded_cols:
            if col_to_remove in numerical_cols:
                numerical_cols.remove(col_to_remove)
        
        if len(numerical_cols) > 1:
            # Translate column names to English and truncate if too long
            translated_cols = []
            try:
                from gradio_helpers import translate_feature_name
                for col in numerical_cols:
                    try:
                        translated_name = translate_feature_name(col)
                        # Truncate very long names to prevent display issues
                        if len(translated_name) > 30:
                            translated_name = translated_name[:27] + '...'
                        translated_cols.append(translated_name)
                    except Exception:
                        # If translation fails for a specific column, use original (truncated if needed)
                        display_name = col[:30] + '...' if len(col) > 30 else col
                        translated_cols.append(display_name)
            except ImportError:
                # Fallback if translation not available
                translated_cols = [col[:30] + '...' if len(col) > 30 else col for col in numerical_cols]
            
            # Calculate dynamic figure size based on number of features
            n_features = len(numerical_cols)
            fig_size = max(14, min(24, n_features * 0.9)), max(12, min(24, n_features * 0.9))
            
            fig, ax = plt.subplots(figsize=fig_size)
            corr_matrix = df_encoded[numerical_cols].corr()
            
            # Create heatmap with translated column names
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                       xticklabels=translated_cols,
                       yticklabels=translated_cols,
                       ax=ax,
                       annot_kws={'size': 8})  # Smaller annotation font for readability
            
            # Set title
            ax.set_title('Correlation Heatmap (Categorical Features Encoded)', fontsize=14, fontweight='bold', pad=20)
            
            # Rotate and align labels properly to prevent overlap
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
            plt.setp(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)
            
            # Adjust layout to ensure labels are fully visible
            plt.tight_layout(pad=3.0)
            
            if return_image:
                result = _fig_to_base64(save_path=save_path, filename="correlation_heatmap.png")
                return result
            # Close figure instead of showing when using non-interactive backend
            if matplotlib.get_backend().lower() == 'agg':
                plt.close()
            else:
                try:
                    plt.show()
                except Exception:
                    plt.close()  # Fallback to close if show fails
        else:
            # If no numerical columns, create a message plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f'Insufficient numerical features for correlation analysis\n(Found {len(numerical_cols)} numerical columns, need at least 2)', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.axis('off')
            plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if return_image:
                result = _fig_to_base64(save_path=save_path, filename="correlation_heatmap.png")
                return result
            # Close figure instead of showing when using non-interactive backend
            if matplotlib.get_backend().lower() == 'agg':
                plt.close()
            else:
                try:
                    plt.show()
                except Exception:
                    plt.close()  # Fallback to close if show fails
    except Exception as e:
        # Error handling: create error message plot
        import traceback
        error_msg = f"Error generating correlation heatmap: {str(e)}"
        print(f"ERROR in plot_correlation_heatmap: {error_msg}")
        print(traceback.format_exc())
        
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error generating correlation heatmap:\n{str(e)[:100]}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        plt.axis('off')
        plt.title('Correlation Heatmap - Error', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if return_image:
            result = _fig_to_base64(save_path=save_path, filename="correlation_heatmap.png")
            return result
        if matplotlib.get_backend().lower() == 'agg':
            plt.close()
        else:
            try:
                plt.show()
            except Exception:
                plt.close()


def plot_prediction_distribution(y_proba, title="Prediction Probability Distribution", return_image=False, save_path=None):
    """Plot distribution of prediction probabilities."""
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='#e74c3c', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if return_image:
        return _fig_to_base64(save_path=save_path, filename="prediction_distribution.png")
    # Close figure instead of showing when using non-interactive backend
    if matplotlib.get_backend().lower() == 'agg':
        plt.close()
    else:
        try:
            plt.show()
        except Exception:
            plt.close()  # Fallback to close if show fails


def create_all_visualizations(df, X_test, y_test, y_pred, y_proba, roc_auc, 
                            pipeline, cat_cols, target, algorithm_name="XGBoost", save_plots=True):
    """
    Create all visualizations for the model.
    
    Args:
        df: DataFrame with data
        X_test: Test features
        y_test: Test labels
        y_pred: Predictions
        y_proba: Prediction probabilities
        roc_auc: ROC AUC score
        pipeline: Trained pipeline
        cat_cols: Categorical columns
        target: Target column name
        algorithm_name: Name of the algorithm (e.g., "XGBoost", "RandomForest") for folder organization
        save_plots: Whether to save plots to files (default: True)
    """
    print("\n" + "="*50)
    print("Creating Visualizations...")
    print("="*50)
    
    # Create save path based on algorithm name
    save_path = None
    if save_plots:
        save_path = os.path.join("output", "plots", algorithm_name)
        os.makedirs(save_path, exist_ok=True)
        print(f"\nSaving plots to: {save_path}/")
    
    # 1. Target distribution
    print("\n1. Plotting target distribution...")
    plot_target_distribution(df[target], save_path=save_path)
    
    # 2. Feature distributions
    if len(cat_cols) > 0:
        print("\n2. Plotting feature distributions...")
        plot_feature_distributions(df, cat_cols, target, save_path=save_path)
    
    # 3. Confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, save_path=save_path)
    
    # 4. ROC curve
    print("\n4. Plotting ROC curve...")
    plot_roc_curve(y_test, y_proba, roc_auc, save_path=save_path)
    
    # 5. Prediction distribution
    print("\n5. Plotting prediction probability distribution...")
    plot_prediction_distribution(y_proba, save_path=save_path)
    
    # 6. Correlation heatmap
    print("\n6. Plotting correlation heatmap...")
    plot_correlation_heatmap(df, target, save_path=save_path)
    
    # 7. SHAP plots
    print("\n7. Creating SHAP visualizations...")
    plot_shap_summary(pipeline, X_test, cat_cols, save_path=save_path)
    
    print("\nAll visualizations completed!")
    if save_path:
        print(f"All plots saved to: {save_path}/")


def _fig_to_base64(save_path=None, filename=None):
    """
    Convert current matplotlib figure to base64 encoded image.
    Optionally save to file if save_path is provided.
    
    Args:
        save_path: Directory path to save the file (optional)
        filename: Filename to save (optional, will use timestamp if not provided)
    
    Returns:
        Base64 encoded HTML image string
    """
    # Save to file if path provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if filename is None:
            filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
        print(f"   Saved: {filepath}")
    
    # Convert to base64 for HTML
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'