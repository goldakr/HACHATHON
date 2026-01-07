"""
Visualization functions for the Gradio interface with type hints and improved error handling.
"""
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
import base64
import os
import shap
from sklearn.pipeline import Pipeline

from exceptions import VisualizationError, SHAPExplanationError, PipelineError
from gradio_helpers import translate_feature_name


def create_shap_summary_plot(
    top_features: List[Tuple[str, float]],
    _shap_values_single: np.ndarray,  # Unused but kept for API compatibility
    _feature_names: np.ndarray,  # Unused but kept for API compatibility
    save_path: Optional[str] = None
) -> str:
    """
    Create a user-friendly SHAP summary plot as HTML.
    
    Args:
        top_features: List of tuples (feature_name, shap_value)
        shap_values_single: Array of SHAP values (unused, kept for API compatibility)
        feature_names: Array of feature names (unused, kept for API compatibility)
        save_path: Optional directory path to save the plot
    
    Returns:
        HTML string with embedded image
    
    Raises:
        VisualizationError: If plot generation fails
    """
    try:
        # Extract feature names and values
        feat_names = [feat.split('__')[-1] if '__' in feat else feat for feat, _ in top_features]
        # Translate Hebrew feature names to English
        feat_names = [translate_feature_name(name) for name in feat_names]
        shap_vals = [val for _, val in top_features]
        
        # Create horizontal bar plot - use consistent, moderate size
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ['#e74c3c' if val > 0 else '#2ecc71' for val in shap_vals]
        y_pos = np.arange(len(feat_names))
        
        bars = ax.barh(y_pos, shap_vals, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_names, fontsize=12)
        ax.set_xlabel('SHAP Value (Impact on Risk)', fontsize=14, fontweight='bold')
        ax.set_title('Top 5 Feature Contributions to PPD Risk', fontsize=16, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, shap_vals)):
            width = bar.get_width()
            label_x = width + (0.01 if width > 0 else -0.01)
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left' if width > 0 else 'right', 
                   va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Save to file if path provided
        if save_path is not None:
            try:
                os.makedirs(save_path, exist_ok=True)
                filepath = os.path.join(save_path, "shap_summary_bar.png")
                plt.savefig(filepath, format='png', dpi=200, bbox_inches='tight')
                print(f"   💾 Saved: {filepath}")
            except OSError as e:
                print(f"Warning: Could not save plot to {save_path}: {e}")
        
        # Convert to base64 HTML with higher DPI and better styling
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=250, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Improved HTML styling for consistent sizing across XGBoost and RF
        # Use moderate, consistent width that works well for both models
        return f'<div style="width:100%; text-align:center; overflow-x:auto; padding:10px;"><img src="data:image/png;base64,{img_base64}" style="width:900px; max-width:100%; height:auto; display:block; margin:0 auto; border:1px solid #ddd;"></div>'
    except Exception as e:
        plt.close()
        raise VisualizationError(f"Failed to create SHAP summary plot: {str(e)}") from e


def create_enhanced_shap_plot(
    top_features: List[Tuple[str, float]],
    shap_values_single: np.ndarray,
    feature_names: np.ndarray,
    base_value: float = 0.5,
    save_path: Optional[str] = None
) -> str:
    """
    Create an enhanced SHAP plot with waterfall-style visualization showing cumulative effect.
    
    Args:
        top_features: List of tuples (feature_name, shap_value)
        shap_values_single: Array of SHAP values
        feature_names: Array of feature names
        base_value: Base prediction value (default: 0.5)
        save_path: Optional directory path to save the plot
    
    Returns:
        HTML string with embedded image
    
    Raises:
        VisualizationError: If plot generation fails
    """
    try:
        # Extract feature names and values
        feat_names = [feat.split('__')[-1] if '__' in feat else feat for feat, _ in top_features]
        # Translate Hebrew feature names to English
        feat_names = [translate_feature_name(name) for name in feat_names]
        shap_vals = [val for _, val in top_features]
        
        # Sort by absolute value for better visualization
        sorted_data = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
        feat_names = [x[0] for x in sorted_data]
        shap_vals = [x[1] for x in sorted_data]
        
        # Create figure with two subplots: bar plot and waterfall
        # Use consistent, moderate size that works well for both XGBoost and RF
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.3)
        
        # Subplot 1: Horizontal bar plot
        ax1 = fig.add_subplot(gs[0])
        colors = ['#e74c3c' if val > 0 else '#2ecc71' for val in shap_vals]
        y_pos = np.arange(len(feat_names))
        
        bars = ax1.barh(y_pos, shap_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feat_names, fontsize=12)
        ax1.set_xlabel('SHAP Value (Impact on Risk)', fontsize=13, fontweight='bold')
        ax1.set_title('Feature Contributions to PPD Risk Prediction', fontsize=15, fontweight='bold', pad=15)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(axis='x', alpha=0.3, linestyle=':')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, shap_vals)):
            width = bar.get_width()
            label_x = width + (0.02 if width > 0 else -0.02)
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{val:+.3f}', ha='left' if width > 0 else 'right', 
                    va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Subplot 2: Waterfall-style cumulative plot
        ax2 = fig.add_subplot(gs[1])
        
        # Calculate cumulative values
        cumulative = [base_value]
        for val in shap_vals:
            cumulative.append(cumulative[-1] + val)
        
        # Create waterfall bars
        for i in range(len(feat_names)):
            bar_height = shap_vals[i]
            bar_bottom = cumulative[i]
            color = '#e74c3c' if bar_height > 0 else '#2ecc71'
            
            ax2.bar(i + 1, bar_height, bottom=bar_bottom, color=color, 
                   alpha=0.7, edgecolor='black', linewidth=1)
            
            label_y = bar_bottom + bar_height/2
            ax2.text(i + 1, label_y, f'{bar_height:+.3f}', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if abs(bar_height) > 0.05 else 'black')
        
        # Show base value
        ax2.bar(0, base_value, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
        ax2.text(0, base_value/2, f'Base\n{base_value:.2f}', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Show final prediction
        final_pred = cumulative[-1]
        ax2.bar(len(feat_names) + 1, final_pred, color='#9b59b6', alpha=0.7, 
               edgecolor='black', linewidth=2)
        ax2.text(len(feat_names) + 1, final_pred/2, f'Final\n{final_pred:.2f}', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Set labels
        ax2.set_xticks(range(len(feat_names) + 2))
        ax2.set_xticklabels(['Base'] + feat_names + ['Final'], rotation=45, ha='right', fontsize=11)
        ax2.set_ylabel('Cumulative Prediction Value', fontsize=13, fontweight='bold')
        ax2.set_title('Waterfall Plot: How Features Build Up to Final Prediction', 
                     fontsize=15, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle=':')
        ax2.set_ylim([0, max(cumulative) * 1.1])
        
        # Add legend
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.7, label='Increases Risk'),
            Patch(facecolor='#2ecc71', alpha=0.7, label='Decreases Risk'),
            Patch(facecolor='#3498db', alpha=0.7, label='Base Value'),
            Patch(facecolor='#9b59b6', alpha=0.7, label='Final Prediction')
        ]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=11)
        
        # Use subplots_adjust() instead of tight_layout() for GridSpec compatibility
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
        
        # Save to file if path provided
        if save_path is not None:
            try:
                os.makedirs(save_path, exist_ok=True)
                filepath = os.path.join(save_path, "shap_enhanced_waterfall.png")
                plt.savefig(filepath, format='png', dpi=200, bbox_inches='tight')
                print(f"   💾 Saved: {filepath}")
            except OSError as e:
                print(f"Warning: Could not save plot to {save_path}: {e}")
        
        # Convert to base64 HTML with higher DPI and better styling
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=250, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Improved HTML styling for consistent sizing across XGBoost and RF
        # Use moderate, consistent width that works well for both models
        return f'<div style="width:100%; text-align:center; overflow-x:auto; padding:10px;"><img src="data:image/png;base64,{img_base64}" style="width:1000px; max-width:100%; height:auto; display:block; margin:0 auto; border:1px solid #ddd;"></div>'
    except Exception as e:
        plt.close()
        # Fallback to simple plot
        try:
            return create_shap_summary_plot(top_features, shap_values_single, feature_names, save_path)
        except Exception:
            raise VisualizationError(f"Failed to create enhanced SHAP plot: {str(e)}") from e


def create_shap_summary_plot_class1(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    max_display: int = 15,
    return_image: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Optional[str]:
    """
    Create a SHAP summary plot for class 1 (Yes Depression) using test data.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features (DataFrame)
        max_display: Maximum number of features to display
        return_image: If True, return base64 HTML image, else show plot
        title: Optional custom title for the plot
        save_path: Optional directory path to save the plot
        
    Returns:
        HTML string with embedded image or None
    
    Raises:
        SHAPExplanationError: If SHAP plot generation fails
        PipelineError: If pipeline operations fail
    """
    try:
        # Get the preprocessed features
        if 'preprocess' not in pipeline.named_steps:
            raise PipelineError("Pipeline missing 'preprocess' step")
        
        preprocessor = pipeline.named_steps['preprocess']
        
        # Use a sample of test data for SHAP calculation (faster)
        X_test_sample = X_test.sample(min(100, len(X_test)), random_state=42) if len(X_test) > 100 else X_test
        X_test_processed = preprocessor.transform(X_test_sample)
        
        # Convert to numpy array if it's a sparse matrix
        if hasattr(X_test_processed, 'toarray'):
            X_test_processed = X_test_processed.toarray()
        X_test_processed = np.array(X_test_processed)
        
        # Get feature names after one-hot encoding
        feature_names = preprocessor.get_feature_names_out()
        
        # Get the model
        if 'model' not in pipeline.named_steps:
            raise PipelineError("Pipeline missing 'model' step")
        
        model = pipeline.named_steps['model']
        
        # Create SHAP explainer
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            raise SHAPExplanationError(f"Failed to create SHAP explainer: {str(e)}") from e
        
        # Calculate SHAP values
        try:
            shap_values_output = explainer.shap_values(X_test_processed)
        except Exception as e:
            raise SHAPExplanationError(f"Failed to calculate SHAP values: {str(e)}") from e
        
        # Handle different return types
        if isinstance(shap_values_output, list):
            if len(shap_values_output) == 2:
                shap_values_class_1 = np.array(shap_values_output[1])
            elif len(shap_values_output) == 1:
                shap_values_class_1 = np.array(shap_values_output[0])
            else:
                shap_values_class_1 = np.array(shap_values_output[-1])
        else:
            shap_values_class_1 = np.array(shap_values_output)
        
        # Ensure shap_values_class_1 is 2D
        if len(shap_values_class_1.shape) == 1:
            shap_values_class_1 = shap_values_class_1.reshape(1, -1)
        elif len(shap_values_class_1.shape) == 3:
            shap_values_class_1 = shap_values_class_1[:, :, -1] if shap_values_class_1.shape[2] > 1 else shap_values_class_1[:, :, 0]
        
        # Ensure matching shapes
        if X_test_processed.shape[0] != shap_values_class_1.shape[0]:
            min_samples = min(X_test_processed.shape[0], shap_values_class_1.shape[0])
            X_test_processed = X_test_processed[:min_samples]
            shap_values_class_1 = shap_values_class_1[:min_samples]
        
        # Determine number of features
        n_features = shap_values_class_1.shape[1]
        
        # Ensure feature names match
        if len(feature_names) > n_features:
            feature_names_to_use = feature_names[:n_features]
        elif len(feature_names) < n_features:
            feature_names_to_use = [f"Feature_{i}" for i in range(n_features)]
        else:
            feature_names_to_use = feature_names
        
        # Clean feature names
        feature_names_to_use = [
            name.replace('cat__', '').replace('num__', '') if 'cat__' in name or 'num__' in name else name
            for name in feature_names_to_use
        ]
        feature_names_to_use = [
            name.split('__')[-1] if '__' in name else name for name in feature_names_to_use
        ]
        # Translate Hebrew feature names to English
        feature_names_to_use = [translate_feature_name(name) for name in feature_names_to_use]
        
        # Ensure X_test_processed has matching features
        if X_test_processed.shape[1] != n_features:
            min_features = min(X_test_processed.shape[1], n_features)
            X_test_processed = X_test_processed[:, :min_features]
            shap_values_class_1 = shap_values_class_1[:, :min_features]
            feature_names_to_use = feature_names_to_use[:min_features]
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(shap_values_class_1, X_test_processed, 
                             feature_names=feature_names_to_use,
                             max_display=max_display,
                             show=False)
        except Exception:
            # Fallback: try without feature names
            try:
                shap.summary_plot(shap_values_class_1, X_test_processed, 
                                 max_display=max_display,
                                 show=False)
            except Exception:
                # Final fallback: use bar plot
                shap.summary_plot(shap_values_class_1, X_test_processed, 
                                 plot_type="bar",
                                 max_display=max_display,
                                 show=False)
        
        plot_title = title if title is not None else "SHAP Summary Plot - Class 1 (Yes Depression)"
        plt.title(plot_title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save to file if path provided
        if save_path is not None:
            try:
                os.makedirs(save_path, exist_ok=True)
                filepath = os.path.join(save_path, "shap_summary_class1.png")
                plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
                print(f"   💾 Saved: {filepath}")
            except OSError as e:
                print(f"Warning: Could not save plot to {save_path}: {e}")
        
        if return_image:
            # Convert to base64 HTML
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'
        else:
            try:
                plt.show()
            except Exception:
                pass
            plt.close()
            return None
            
    except (SHAPExplanationError, PipelineError):
        raise
    except Exception as e:
        plt.close()
        raise SHAPExplanationError(f"Failed to generate SHAP summary plot: {str(e)}") from e


def generate_detailed_shap_explanation(
    feature_importance: List[dict],
    risk_percentage: float
) -> str:
    """
    Generate a detailed SHAP explanation markdown.
    
    Args:
        feature_importance: List of dicts with 'feature' and 'shap_value' keys
        risk_percentage: Risk percentage (0-100)
    
    Returns:
        Markdown string with detailed SHAP explanation
    """
    try:
        explanation_lines = [
            "## SHAP (SHapley Additive exPlanations) Analysis",
            "",
            f"**Overall Risk Score: {risk_percentage:.2f}%**",
            "",
            "SHAP values explain how each feature contributes to the final prediction.",
            "- **Positive SHAP values** push the prediction toward higher PPD risk",
            "- **Negative SHAP values** push the prediction toward lower PPD risk",
            "- **Magnitude** indicates the strength of the contribution",
            "",
            "### Feature Contributions:",
            ""
        ]
        
        for i, feature in enumerate(feature_importance[:5], 1):
            feat_name = feature.get('feature', 'Unknown').split('__')[-1]
            # Translate Hebrew feature names to English
            feat_name = translate_feature_name(feat_name)
            shap_val = feature.get('shap_value', 0.0)
            abs_val = abs(shap_val)
            impact = "high" if abs_val > 0.1 else "moderate" if abs_val > 0.05 else "low"
            direction = "increases" if shap_val > 0 else "decreases"
            
            explanation_lines.extend([
                f"**{i}. {feat_name}**",
                f"  - SHAP value: {shap_val:+.4f}",
                f"  - Impact: {impact.upper()} ({direction} risk by {abs_val:.4f})",
                f"  - {'Positive SHAP value means this feature pushes the prediction toward higher risk.' if shap_val > 0 else 'Negative SHAP value means this feature pushes the prediction toward lower risk.'}",
                ""
            ])
        
        explanation_lines.extend([
            "### How to Interpret:",
            "- **High impact** (|SHAP| > 0.1): This feature significantly influences the prediction",
            "- **Moderate impact** (0.05 < |SHAP| ≤ 0.1): This feature moderately influences the prediction",
            "- **Low impact** (|SHAP| ≤ 0.05): This feature has a minor influence on the prediction",
            "",
            "The sum of all SHAP values equals the difference between the model's prediction and the baseline (average prediction)."
        ])
        
        return "\n".join(explanation_lines)
    except Exception as e:
        return f"## SHAP Explanation Unavailable\n\nError generating explanation: {str(e)}"



