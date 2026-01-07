"""
Helper functions for Gradio interface to reduce code duplication and improve maintainability.
"""
from typing import Optional, List, Dict, Any
from sklearn.pipeline import Pipeline
import numpy as np


def get_algorithm_name(pipeline: Pipeline) -> str:
    """
    Get the algorithm name from a pipeline.
    
    Args:
        pipeline: Trained sklearn pipeline
    
    Returns:
        Algorithm name: "XGBoost" or "RandomForest"
    """
    try:
        model_type = type(pipeline.named_steps['model']).__name__
        return "XGBoost" if "XGB" in model_type.upper() else "RandomForest"
    except (KeyError, AttributeError):
        return "Unknown"


# Translation dictionary for Hebrew feature names to English
FEATURE_NAME_TRANSLATIONS = {
    "מחשבות פגיעה עצמית": "Self-harm thoughts",
    # Add more translations as needed
}


def translate_feature_name(feature_name: str) -> str:
    """
    Translate Hebrew feature names to English for visualization.
    
    Args:
        feature_name: Feature name (may be Hebrew)
        
    Returns:
        Translated feature name in English, or original if no translation exists
    """
    # Check for exact match
    if feature_name in FEATURE_NAME_TRANSLATIONS:
        return FEATURE_NAME_TRANSLATIONS[feature_name]
    
    # Check if the Hebrew text appears in the feature name (may have prefixes/suffixes)
    for hebrew, english in FEATURE_NAME_TRANSLATIONS.items():
        if hebrew in feature_name:
            return feature_name.replace(hebrew, english)
    
    return feature_name


def clean_feature_name(feature_name: str) -> str:
    """
    Clean feature name by removing preprocessing prefixes and translate Hebrew to English.
    
    Args:
        feature_name: Raw feature name (may contain prefixes like 'cat__', 'num__', etc.)
    
    Returns:
        Cleaned and translated feature name
    """
    # First clean the preprocessing prefixes
    if '__' in feature_name:
        cleaned = feature_name.split('__')[-1]
    else:
        cleaned = feature_name
    
    # Then translate Hebrew to English
    return translate_feature_name(cleaned)


def calculate_impact_level(shap_value: float) -> str:
    """
    Calculate impact level based on SHAP value magnitude.
    
    Args:
        shap_value: SHAP value (can be positive or negative)
    
    Returns:
        Impact level: "high", "moderate", or "low"
    """
    abs_val = abs(shap_value)
    if abs_val > 0.1:
        return "high"
    elif abs_val > 0.05:
        return "moderate"
    else:
        return "low"


def format_feature_importance_line(
    feature_name: str,
    shap_value: float,
    index: int
) -> str:
    """
    Format a single feature importance line.
    
    Args:
        feature_name: Feature name (will be cleaned)
        shap_value: SHAP value
        index: Feature index (1-based)
    
    Returns:
        Formatted string
    """
    clean_name = clean_feature_name(feature_name)
    abs_val = abs(shap_value)
    impact = calculate_impact_level(shap_value)
    direction = "increases" if shap_value > 0 else "decreases"
    
    return f"{index}. {clean_name}\n   Impact: {impact} ({direction} risk by {abs_val:.3f})"


def generate_shap_explanation_markdown(
    feature_importance: List[Dict[str, Any]],
    risk_percentage: Optional[float] = None
) -> str:
    """
    Generate detailed SHAP explanation markdown.
    
    Args:
        feature_importance: List of dicts with 'feature' and 'shap_value' keys
        risk_percentage: Optional risk percentage to include
    
    Returns:
        Markdown string with SHAP explanation
    """
    lines = [
        "## SHAP (SHapley Additive exPlanations) Analysis",
        ""
    ]
    
    if risk_percentage is not None:
        lines.append(f"**Overall Risk Score: {risk_percentage:.2f}%**")
        lines.append("")
    
    lines.extend([
        "SHAP values explain how each feature contributes to the final prediction.",
        "- **Positive SHAP values** push the prediction toward higher PPD risk",
        "- **Negative SHAP values** push the prediction toward lower PPD risk",
        "- **Magnitude** indicates the strength of the contribution",
        "",
        "### Feature Contributions:",
        ""
    ])
    
    for i, feature in enumerate(feature_importance[:5], 1):
        feat_name = clean_feature_name(feature.get('feature', 'Unknown'))
        shap_val = feature.get('shap_value', 0.0)
        abs_val = abs(shap_val)
        impact = calculate_impact_level(shap_val)
        direction = "increases" if shap_val > 0 else "decreases"
        
        lines.extend([
            f"**{i}. {feat_name}**",
            f"  - SHAP value: {shap_val:+.4f}",
            f"  - Impact: {impact.upper()} ({direction} risk by {abs_val:.4f})",
            f"  - {'Positive SHAP value means this feature pushes the prediction toward higher risk.' if shap_val > 0 else 'Negative SHAP value means this feature pushes the prediction toward lower risk.'}",
            ""
        ])
    
    lines.extend([
        "### How to Interpret:",
        "- **High impact** (|SHAP| > 0.1): This feature significantly influences the prediction",
        "- **Moderate impact** (0.05 < |SHAP| ≤ 0.1): This feature moderately influences the prediction",
        "- **Low impact** (|SHAP| ≤ 0.05): This feature has a minor influence on the prediction",
        "",
        "The sum of all SHAP values equals the difference between the model's prediction and the baseline (average prediction)."
    ])
    
    return "\n".join(lines)


def get_save_path_for_algorithm(pipeline: Optional[Pipeline] = None, algorithm_name: Optional[str] = None) -> Optional[str]:
    """
    Get save path for plots based on algorithm.
    Returns path relative to the Hackathon directory (script directory).
    
    Args:
        pipeline: Optional pipeline to detect algorithm from
        algorithm_name: Optional algorithm name (if pipeline not provided)
    
    Returns:
        Save path string or None
    """
    import os
    from pathlib import Path
    try:
        if algorithm_name is None and pipeline is not None:
            algorithm_name = get_algorithm_name(pipeline)
        elif algorithm_name is None:
            return None
        
        # Get the Hackathon directory (where the script is located)
        script_dir = Path(__file__).parent
        save_path = script_dir / "output" / "plots" / algorithm_name
        return str(save_path)
    except Exception:
        return None

