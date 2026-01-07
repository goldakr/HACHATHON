"""
Prediction functions for the Gradio interface with type hints and improved error handling.
"""
from typing import Tuple, List, Dict, Optional, Any
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import shap
import sys

from exceptions import PredictionError, SHAPExplanationError, DataValidationError


def predict_depression(
    pipeline: Pipeline,
    explainer: Optional[shap.TreeExplainer],
    feature_columns: List[str],
    feature_dtypes: Dict[str, str],
    **kwargs
) -> Tuple[str, str, str, str, str]:
    """
    Predict postpartum depression risk based on input features.

    Args:
        pipeline: Trained sklearn pipeline
        explainer: SHAP explainer object (optional)
        feature_columns: List of column names in the order expected by the pipeline
        feature_dtypes: Dict of dtypes from training data
        **kwargs: Feature values as keyword arguments matching feature column names

    Returns:
        tuple: (risk_score, feature_importance, personalized_explanation, shap_explanation, plot_html)

    Raises:
        DataValidationError: If input data validation fails
        PredictionError: If prediction fails
        SHAPExplanationError: If SHAP explanation generation fails
    """
    try:
        # Filter out composite target column if present
        feature_columns_filtered = [col for col in feature_columns if col != "PPD_Composite"]
        
        # Create input row matching the exact structure used during training
        # Map kwargs to feature columns dynamically
        row_dict: Dict[str, str] = {}
        for col in feature_columns_filtered:
            # Check if this column is in kwargs (exact match or with spaces converted to underscores)
            col_key = col.replace(" ", "_")
            if col in kwargs:
                value = kwargs[col]
                row_dict[col] = "" if value is None else str(value).strip()
            elif col_key in kwargs:
                value = kwargs[col_key]
                row_dict[col] = "" if value is None else str(value).strip()
            else:
                # Default to empty string if not provided
                row_dict[col] = ""

        # Validate that all required columns are present
        missing_cols = set(feature_columns_filtered) - set(row_dict.keys())
        if missing_cols:
            raise DataValidationError(
                f"Missing required feature columns: {missing_cols}. "
                "This indicates a bug in the feature mapping logic."
            )

        # Create DataFrame ensuring exact column order and dtypes match training data
        row = pd.DataFrame([row_dict], columns=feature_columns_filtered)

        # Validate for NaN values before type conversion
        if row.isna().any().any():
            nan_cols = row.columns[row.isna().any()].tolist()
            raise DataValidationError(
                f"Found NaN values in feature columns: {nan_cols}. "
                "All feature values must be provided."
            )

        # Convert dtypes to match training data exactly
        for col in row.columns:
            if col in feature_dtypes:
                target_dtype = feature_dtypes[col]
                if target_dtype == "object":
                    row[col] = row[col].fillna("").astype(str)
                    row[col] = row[col].replace("nan", "", regex=False)

        # Debug output
        print("DEBUG - Input row values:", file=sys.stderr)
        for col in row.columns:
            print(f"  {col}: {row[col].values[0]}", file=sys.stderr)

        # Get prediction probability
        try:
            proba_result = pipeline.predict_proba(row)
        except Exception as e:
            raise PredictionError(f"Failed to generate prediction probabilities: {str(e)}") from e

        # Verify the model classes and get probability
        model = pipeline.named_steps["model"]
        if hasattr(model, "classes_"):
            prob_class_0 = proba_result[0, 0]
            prob_class_1 = proba_result[0, 1]
            
            if len(model.classes_) == 2:
                if model.classes_[0] == 0 and model.classes_[1] == 1:
                    proba = prob_class_1
                elif model.classes_[0] == 1 and model.classes_[1] == 0:
                    proba = prob_class_0
                else:
                    proba = prob_class_1
            else:
                proba = proba_result[0, 1]
        else:
            proba = proba_result[0, 1]

        risk_score = f"PPD Risk Score: {proba:.2%}"

        # Get prediction class
        try:
            pred_class = pipeline.predict(row)[0]
        except Exception as e:
            raise PredictionError(f"Failed to generate prediction class: {str(e)}") from e

        # SHAP explanation
        try:
            if explainer is None:
                raise SHAPExplanationError(
                    "SHAP explainer not initialized. Please train the model first."
                )
            
            # Get preprocessed features
            preprocessor = pipeline.named_steps["preprocess"]
            row_processed = preprocessor.transform(row)
            feature_names = preprocessor.get_feature_names_out()

            # Calculate SHAP values
            shap_values = explainer.shap_values(row_processed)

            # Handle SHAP values format
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]

            shap_values = np.array(shap_values)
            
            if len(shap_values.shape) > 1:
                shap_values_single = shap_values[0]
            else:
                shap_values_single = shap_values
            
            shap_values_single = np.array(shap_values_single).flatten()
            
            # Convert row_processed to dense array if sparse, then flatten
            try:
                if hasattr(row_processed, 'toarray'):
                    row_processed_flat = row_processed.toarray().flatten()
                else:
                    row_processed_flat = np.array(row_processed).flatten()
            except Exception:
                # Fallback: if we can't get processed values, don't filter
                row_processed_flat = None

            # 🔧 FIX: Filter out inactive one-hot encoded features
            # For one-hot encoded features, only show the active one (value = 1)
            # This prevents showing both "GDM_No" and "GDM_Yes" when only one is active
            # One-hot encoded features have value 1.0 for the active category and 0.0 for others
            
            # Create list of (feature_name, shap_value) tuples, filtering inactive one-hot features
            feat_imp_candidates = []
            for idx, (feat_name, shap_val) in enumerate(zip(feature_names, shap_values_single)):
                # Ensure shap_val is numeric
                try:
                    shap_val = float(shap_val)
                except (ValueError, TypeError):
                    continue  # Skip if not numeric
                
                # Check if this is an active one-hot encoded feature
                if row_processed_flat is not None and idx < len(row_processed_flat):
                    try:
                        # Convert to float safely
                        if hasattr(row_processed_flat[idx], 'item'):
                            feature_value = float(row_processed_flat[idx].item())
                        else:
                            feature_value = float(row_processed_flat[idx])
                        
                        # For one-hot encoded features, value should be 1.0 for active, 0.0 for inactive
                        # We only want to show features that are actually active (value ≈ 1.0)
                        # OR numeric features (which can have any value)
                        
                        # Detect if this is likely a one-hot encoded feature
                        # One-hot features typically have values of exactly 0.0 or 1.0
                        is_onehot_feature = abs(feature_value - 1.0) < 0.01 or abs(feature_value - 0.0) < 0.01
                        
                        # If it's a one-hot feature and not active (value ≈ 0), skip it
                        if is_onehot_feature and abs(feature_value - 1.0) > 0.01:
                            continue  # Skip inactive one-hot encoded features
                    except (IndexError, ValueError, TypeError):
                        # If we can't check the value, include the feature anyway
                        pass
                
                feat_imp_candidates.append((feat_name, shap_val))
            
            # Get top 5 most important features (from active features only)
            # If filtering removed all features, fall back to showing all features
            if len(feat_imp_candidates) == 0:
                # Fallback: show top features without filtering
                feat_imp = sorted(
                    list(zip(feature_names, shap_values_single)), key=lambda x: -abs(x[1])
                )[:5]
            else:
                feat_imp = sorted(
                    feat_imp_candidates, key=lambda x: -abs(x[1])
                )[:5]

            # Format feature importance
            feat_imp_lines = []
            if len(feat_imp) > 0:
                for i, (feat, val) in enumerate(feat_imp, 1):
                    # Ensure val is numeric
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        continue  # Skip if not numeric
                    
                    clean_feat = feat.split('__')[-1] if '__' in feat else feat
                    # Translate Hebrew feature names to English
                    from gradio_helpers import translate_feature_name
                    clean_feat = translate_feature_name(clean_feat)
                    # SHAP value interpretation:
                    # Positive SHAP = increases risk (pushes prediction toward class 1)
                    # Negative SHAP = decreases risk (pushes prediction toward class 0)
                    # For SES: SES_Low should increase risk (positive), SES_High should decrease risk (negative)
                    direction = "increases" if val > 0 else "decreases"
                    impact = "high" if abs(val) > 0.1 else "moderate" if abs(val) > 0.05 else "low"
                    feat_imp_lines.append(
                        f"{i}. {clean_feat}\n   Impact: {impact} ({direction} risk by {abs(val):.3f})"
                    )
            feat_imp_str = "\n\n".join(feat_imp_lines) if feat_imp_lines else "Feature importance not available"

            # Generate personalized explanation
            personalized_explanation = generate_personalized_explanation(
                proba, pred_class, feat_imp, row, feature_names, shap_values_single
            )

            # Create SHAP visualization (imported from gradio_visualizations)
            # Use create_enhanced_shap_plot for consistency with agent path
            from gradio_visualizations import create_enhanced_shap_plot
            import os
            
            save_path_shap = None
            try:
                model_type = type(pipeline.named_steps['model']).__name__
                algo_name = "XGBoost" if "XGB" in model_type.upper() else "RandomForest"
                save_path_shap = os.path.join("output", "plots", algo_name)
            except Exception:
                pass
            
            # Convert feat_imp to the format expected by create_enhanced_shap_plot
            # feat_imp is list of (feature_name, shap_value) tuples
            feat_names_list = [feat[0] for feat in feat_imp]
            shap_vals_array = np.array([feat[1] for feat in feat_imp])
            
            plot_html = create_enhanced_shap_plot(
                feat_imp,  # top_features: list of tuples
                shap_vals_array,  # shap_values_single
                np.array(feat_names_list),  # feature_names
                base_value=0.5,  # base value for waterfall
                save_path=save_path_shap
            )
            
            # Create detailed SHAP explanation
            shap_explanation_lines = []
            for i, (feat, val) in enumerate(feat_imp, 1):
                clean_feat = feat.split('__')[-1] if '__' in feat else feat
                abs_val = abs(val)
                impact = "high" if abs_val > 0.1 else "moderate" if abs_val > 0.05 else "low"
                direction = "increases" if val > 0 else "decreases"
                shap_explanation_lines.append(
                    f"**{clean_feat}**: SHAP value = {val:.4f}\n"
                    f"  • This feature {direction} the PPD risk prediction by {abs_val:.4f}\n"
                    f"  • Impact level: {impact.upper()}\n"
                    f"  • {'Positive SHAP value means this feature pushes the prediction toward higher risk.' if val > 0 else 'Negative SHAP value means this feature pushes the prediction toward lower risk.'}"
                )
            
            shap_explanation = f"""## SHAP (SHapley Additive exPlanations) Analysis

SHAP values explain how each feature contributes to the final prediction.
- **Positive SHAP values** push the prediction toward higher PPD risk
- **Negative SHAP values** push the prediction toward lower PPD risk
- **Magnitude** indicates the strength of the contribution

### Feature Contributions:

{chr(10).join(shap_explanation_lines) if shap_explanation_lines else 'No SHAP values available'}

### How to Interpret:
- **High impact** (|SHAP| > 0.1): This feature significantly influences the prediction
- **Moderate impact** (0.05 < |SHAP| ≤ 0.1): This feature moderately influences the prediction
- **Low impact** (|SHAP| ≤ 0.05): This feature has a minor influence on the prediction

The sum of all SHAP values equals the difference between the model's prediction and the baseline (average prediction).
"""

        except SHAPExplanationError:
            raise
        except Exception as e:
            raise SHAPExplanationError(
                f"Failed to generate SHAP explanation: {str(e)}"
            ) from e

        return risk_score, feat_imp_str, personalized_explanation, shap_explanation, plot_html

    except (DataValidationError, PredictionError, SHAPExplanationError):
        raise
    except Exception as e:
        raise PredictionError(f"Unexpected error during prediction: {str(e)}") from e


def generate_personalized_explanation(
    proba: float,
    pred_class: int,
    top_features: List[Tuple[str, float]],
    row: pd.DataFrame,
    feature_names: np.ndarray,
    shap_values_single: np.ndarray,
) -> str:
    """
    Generate a personalized explanation based on prediction results.

    Args:
        proba: Prediction probability
        pred_class: Predicted class (0 or 1)
        top_features: List of tuples (feature_name, shap_value) for top features
        row: Input row DataFrame
        feature_names: Array of feature names
        shap_values_single: Array of SHAP values

    Returns:
        Personalized explanation string
    """
    risk_level = "high" if proba >= 0.7 else "moderate" if proba >= 0.4 else "low"
    prediction = "Yes (Depression)" if pred_class == 1 else "No (No Depression)"
    
    explanation = f"""Based on the provided information, the model predicts: **{prediction}**

**Risk Level: {risk_level.upper()}** (Probability: {proba:.1%})

### Key Factors Influencing This Prediction:
"""
    
    for feat, val in top_features[:3]:  # Top 3 features
        # Clean feature name properly from one-hot encoded format
        try:
            from gradio_helpers import clean_feature_name, translate_feature_name
            clean_feat = clean_feature_name(feat)
            # Extract base feature name from one-hot encoded format
            # One-hot encoded features have format: "FeatureName_CategoryValue"
            if '_' in clean_feat:
                # Extract base feature (everything before last underscore)
                base_feat = clean_feat.rsplit('_', 1)[0]
                clean_feat = base_feat
            # Translate to English
            clean_feat = translate_feature_name(clean_feat)
        except ImportError:
            # Fallback: simple cleaning
            clean_feat = feat.split('__')[-1] if '__' in feat else feat
            if '_' in clean_feat:
                clean_feat = clean_feat.rsplit('_', 1)[0]
        
        direction = "increases" if val > 0 else "decreases"
        explanation += f"\n- **{clean_feat}**: {direction.capitalize()} risk (SHAP: {val:+.3f})"
    
    if proba >= 0.7:
        explanation += "\n\n⚠️ **High Risk**: Consider consulting with a healthcare professional."
    elif proba >= 0.4:
        explanation += "\n\n⚠️ **Moderate Risk**: Monitor symptoms and consider professional advice."
    else:
        explanation += "\n\n✅ **Low Risk**: Continue monitoring and maintain healthy practices."
    
    return explanation



