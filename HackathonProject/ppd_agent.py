"""
Postpartum Depression (PPD) Prediction Agent Tool

This module provides an agent interface for the PPD prediction system,
making it easy to integrate with agent frameworks, APIs, and other tools.
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from MLmodel import create_rf_pipeline, create_XGBoost_pipeline, train_and_evaluate, optimize_XGBoost_hyperparameters


class PPDAgent:
    """
    Agent class for Postpartum Depression risk prediction.
    
    This agent can be used as a standalone tool or integrated with
    agent frameworks like LangChain, AutoGPT, etc.
    """
    
    def __init__(self, pipeline, X_train, cat_cols, feature_columns=None):
        """
        Initialize the PPD Agent.
        
        Args:
            pipeline: Trained sklearn pipeline
            X_train: Training data (for SHAP explainer)
            cat_cols: List of categorical column names
            feature_columns: Optional list of feature column names in order
        """
        self.pipeline = pipeline
        self.X_train = X_train
        
        # Validate cat_cols - ensure all categorical columns exist in X_train
        X_train_cols = set(X_train.columns)
        valid_cat_cols = [col for col in cat_cols if col in X_train_cols]
        if len(valid_cat_cols) != len(cat_cols):
            missing = set(cat_cols) - set(valid_cat_cols)
            print(f"  ⚠️  Warning: Some categorical columns not in X_train: {missing}")
        self.cat_cols = valid_cat_cols
        
        self.feature_columns = feature_columns if feature_columns is not None else list(X_train.columns)
        
        # Filter out target column if present (support both PPD and PPD_Composite)
        self.feature_columns = [col for col in self.feature_columns if col not in ["PPD", "PPD_Composite"]]
        
        # Ensure ID and Name are never in feature columns (ID is for merging, Name is for display only)
        if 'ID' in self.feature_columns:
            print("  Warning: Removing 'ID' from feature_columns (ID is only for data merging)")
            self.feature_columns = [col for col in self.feature_columns if col != 'ID']
        if 'Name' in self.feature_columns:
            print("  Warning: Removing 'Name' from feature_columns (Name is only for display)")
            self.feature_columns = [col for col in self.feature_columns if col != 'Name']
        
        # Get feature dtypes from training data
        self.feature_dtypes = {col: str(X_train[col].dtype) for col in self.feature_columns}
        
        # Verify categorical columns match actual dtypes
        actual_cat_cols = [col for col in self.feature_columns if self.feature_dtypes.get(col) == "object"]
        if set(self.cat_cols) != set(actual_cat_cols):
            print(f"  ⚠️  Warning: cat_cols mismatch. Provided: {self.cat_cols}, Actual object columns: {actual_cat_cols}")
            # Update to match actual dtypes
            self.cat_cols = actual_cat_cols
        
        # Initialize SHAP explainer (only if model is trained)
        try:
            model = self.pipeline.named_steps.get("model")
            if model is not None:
                # Check if model is trained by looking for feature_importances_ or other training attributes
                # For tree-based models (RandomForest, XGBoost), feature_importances_ exists after training
                is_trained = (hasattr(model, 'feature_importances_') and model.feature_importances_ is not None) or \
                             (hasattr(model, 'estimators_') and model.estimators_ is not None) or \
                             (hasattr(model, 'get_booster') or hasattr(model, 'get_params'))
                
                if is_trained:
                    print("Initializing SHAP explainer...")
                    self.explainer = shap.TreeExplainer(model)
                    print("SHAP explainer ready!")
                else:
                    # Model not trained yet, explainer will be created after training
                    self.explainer = None
                    print("Model not trained yet. SHAP explainer will be created after training.")
            else:
                self.explainer = None
                print("No model found in pipeline. SHAP explainer will be created after training.")
        except Exception as e:
            # Model not trained or explainer creation failed
            self.explainer = None
            print(f"SHAP explainer not initialized (model may not be trained yet): {e}")
    
    def predict(self, **kwargs) -> Dict[str, Any]:
        """
        Predict PPD risk based on input features.
        
        This method accepts keyword arguments that match feature column names.
        For convenience, you can pass feature values directly as keyword arguments.
        
        Example:
            agent.predict(Age="30-35", "Marital status"="Married", SES="High", ...)
        
        Alternatively, use predict_from_dict() for better control.
        
        Args:
            **kwargs: Feature names and values matching self.feature_columns
        
        Returns:
            Dictionary with prediction results including:
            - risk_score: PPD risk probability (0-1)
            - risk_percentage: PPD risk as percentage
            - risk_level: 'Low', 'Moderate', 'High', or 'Very High'
            - prediction: Binary prediction (0 or 1)
            - feature_importance: Top 5 feature contributions
            - explanation: Personalized explanation
        """
        # Use predict_from_dict with the provided kwargs
        return self.predict_from_dict(kwargs)
    
    def _generate_explanation(self, risk_level: str, risk_percentage: float, 
                             feature_importance: List[Dict]) -> str:
        """Generate a personalized explanation based on risk and features."""
        explanation = f"The model identifies a {risk_level.lower()} risk ({risk_percentage:.2f}%)"
        
        if feature_importance:
            top_features = feature_importance[:3]
            # Clean feature names properly from one-hot encoded format
            feature_names = []
            try:
                from gradio_helpers import clean_feature_name, translate_feature_name
                
                # Get original feature columns for matching
                original_features = getattr(self, 'feature_columns', [])
                
                for f in top_features:
                    feat_name = f["feature"]
                    # Use clean_feature_name to remove preprocessing prefixes
                    clean_name = clean_feature_name(feat_name)
                    
                    # Extract base feature name from one-hot encoded format
                    # One-hot encoded features have format: "FeatureName_CategoryValue"
                    # We want to extract "FeatureName" (everything before the last underscore)
                    base_feature = clean_name
                    if '_' in clean_name:
                        # Try to match against original feature columns
                        parts = clean_name.rsplit('_', 1)  # Split from right, only once
                        if len(parts) == 2:
                            potential_base = parts[0]
                            # Check if this matches an original feature column
                            # Match case-insensitively and handle spaces
                            potential_base_normalized = potential_base.replace('_', ' ').strip()
                            found_match = False
                            for orig_feat in original_features:
                                if (potential_base_normalized.lower() == orig_feat.lower() or 
                                    potential_base.lower() == orig_feat.lower().replace(' ', '_')):
                                    base_feature = orig_feat
                                    found_match = True
                                    break
                            
                            # If no match found, use the part before last underscore
                            if not found_match:
                                base_feature = potential_base
                    
                    feature_names.append(base_feature)
                
                # Translate to English
                feature_names = [translate_feature_name(name) for name in feature_names]
            except (ImportError, AttributeError):
                # Fallback: simple cleaning
                for f in top_features:
                    feat_name = f["feature"]
                    # Remove preprocessing prefixes
                    if '__' in feat_name:
                        feat_name = feat_name.split('__')[-1]
                    # Extract base feature (everything before last underscore)
                    if '_' in feat_name:
                        feat_name = feat_name.rsplit('_', 1)[0]
                    feature_names.append(feat_name)
            
            impacts = [f["impact"] for f in top_features]
            
            if len(top_features) > 0:
                explanation += ", mainly due to "
                if len(top_features) == 1:
                    explanation += f"{feature_names[0]} which {impacts[0]} the risk"
                elif len(top_features) == 2:
                    explanation += f"the combination between {feature_names[0]} and {feature_names[1]}"
                else:
                    explanation += f"the combination between {', '.join(feature_names[:-1])}, and {feature_names[-1]}"
        
        explanation += "."
        return explanation
    
    def _apply_domain_knowledge_corrections(self, row_dict: Dict[str, Any], base_probability: float) -> tuple:
        """
        Apply domain knowledge corrections to prediction probability.
        
        This enforces correct relationships when data contradicts domain knowledge.
        
        Args:
            row_dict: Dictionary with feature values
            base_probability: Base prediction probability from model
            
        Returns:
            Tuple of (corrected_probability, adjustments_list)
        """
        corrected_prob = base_probability
        adjustments = []
        
        # Domain knowledge adjustment factors (based on medical literature)
        # These enforce correct relationships regardless of what model learned
        
        # 1. SES adjustments: Low SES should increase risk
        if 'SES' in row_dict:
            ses = str(row_dict['SES']).strip()
            if ses in ['Very Low', 'Low']:
                adjustments.append(('SES_Low', +0.10))  # Increase by 10%
            elif ses == 'High':
                adjustments.append(('SES_High', -0.05))  # Decrease by 5%
        
        # 2. First birth: Yes should increase risk
        if 'First birth' in row_dict:
            first_birth = str(row_dict['First birth']).strip()
            if first_birth == 'Yes':
                adjustments.append(('FirstBirth_Yes', +0.08))  # Increase by 8%
            elif first_birth == 'No':
                adjustments.append(('FirstBirth_No', -0.03))  # Decrease by 3%
        
        # 3. Family/Social support: Low should increase risk
        if 'Family or social support' in row_dict:
            support = str(row_dict['Family or social support']).strip()
            if support == 'Low':
                adjustments.append(('FamilySupport_Low', +0.12))  # Increase by 12%
            elif support == 'High':
                adjustments.append(('FamilySupport_High', -0.05))  # Decrease by 5%
        
        # 4. Partner support: Low/Interrupted should increase risk
        if 'Partner support' in row_dict:
            partner_support = str(row_dict['Partner support']).strip()
            if partner_support in ['Low', 'Interrupted']:
                adjustments.append(('PartnerSupport_Low', +0.10))  # Increase by 10%
            elif partner_support == 'High':
                adjustments.append(('PartnerSupport_High', -0.05))  # Decrease by 5%
        
        # 5. Depression/Anxiety history: Documented should increase risk
        if 'Depression History' in row_dict:
            if str(row_dict['Depression History']).strip() == 'Documented':
                adjustments.append(('DepressionHistory', +0.15))  # Increase by 15%
        
        if 'Anxiety History' in row_dict:
            if str(row_dict['Anxiety History']).strip() == 'Documented':
                adjustments.append(('AnxietyHistory', +0.12))  # Increase by 12%
        
        if 'Depression or anxiety during pregnancy' in row_dict:
            if str(row_dict['Depression or anxiety during pregnancy']).strip() == 'Yes':
                adjustments.append(('DepressionAnxietyPregnancy', +0.15))  # Increase by 15%
        
        # 6. Domestic violence: All types should increase risk
        if 'Domestic violence' in row_dict:
            violence = str(row_dict['Domestic violence']).strip()
            if violence == 'Physical':
                adjustments.append(('DomesticViolence_Physical', +0.20))  # Increase by 20%
            elif violence == 'Sexual':
                adjustments.append(('DomesticViolence_Sexual', +0.20))  # Increase by 20%
            elif violence == 'Verbal':
                adjustments.append(('DomesticViolence_Verbal', +0.10))  # Increase by 10%
            elif violence == 'Economic':
                adjustments.append(('DomesticViolence_Economic', +0.08))  # Increase by 8%
        
        # 7. Sleep quality: Poor sleep should increase risk
        if 'Sleep quality' in row_dict:
            sleep = str(row_dict['Sleep quality']).strip()
            if sleep in ['Disordered Breathing', 'RLS', 'Insomnia']:
                adjustments.append(('SleepQuality_Poor', +0.08))  # Increase by 8%
        
        # 8. Fatigue: Yes should increase risk
        if 'Fatigue' in row_dict:
            if str(row_dict['Fatigue']).strip() == 'Yes':
                adjustments.append(('Fatigue_Yes', +0.10))  # Increase by 10%
        
        # Apply adjustments (cap at 0 and 1)
        total_adjustment = sum(adj[1] for adj in adjustments)
        corrected_prob = base_probability + total_adjustment
        corrected_prob = max(0.0, min(1.0, corrected_prob))  # Clip to [0, 1]
        
        return corrected_prob, adjustments
    
    def predict_from_dict(self, input_dict: Dict[str, Any], apply_domain_knowledge: bool = True) -> Dict[str, Any]:
        """
        Predict from a dictionary of inputs. This is the core prediction method
        that works with any feature structure dynamically.
        
        Args:
            input_dict: Dictionary with feature names as keys (must match feature_columns)
            apply_domain_knowledge: If True, apply domain knowledge corrections to enforce
                                   correct relationships (default: True)
        
        Returns:
            Prediction results dictionary
        """
        # Create input row with all feature columns, using defaults for missing values
        row_dict = {}
        for col in self.feature_columns:
            if col in input_dict:
                value = input_dict[col]
                # Convert to string if not already
                if value is None:
                    row_dict[col] = ""
                else:
                    row_dict[col] = str(value).strip()
            else:
                # Use empty string as default for missing features
                row_dict[col] = ""
        
        # Create DataFrame with correct column order
        row = pd.DataFrame([row_dict], columns=self.feature_columns)
        
        # Validate column match
        missing_cols = set(self.feature_columns) - set(row.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input: {missing_cols}. Expected: {self.feature_columns}")
        
        extra_cols = set(row.columns) - set(self.feature_columns)
        if extra_cols:
            # Remove extra columns
            row = row[self.feature_columns]
        
        # Convert dtypes to match training data
        for col in row.columns:
            if col in self.feature_dtypes:
                target_dtype = self.feature_dtypes[col]
                if target_dtype == "object":
                    row[col] = row[col].fillna("").astype(str)
                    row[col] = row[col].replace("nan", "", regex=False)
                elif target_dtype in ["int64", "float64"]:
                    # Convert numeric columns
                    try:
                        row[col] = pd.to_numeric(row[col], errors='coerce')
                        if target_dtype == "int64":
                            row[col] = row[col].fillna(0).astype(int)
                        else:
                            row[col] = row[col].fillna(0.0).astype(float)
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to {target_dtype}: {e}")
                        # Fill with 0 for numeric columns if conversion fails
                        if target_dtype == "int64":
                            row[col] = 0
                        else:
                            row[col] = 0.0
        
        # Get base prediction from model
        try:
            proba_result = self.pipeline.predict_proba(row)
            # Handle different model output formats
            if proba_result.shape[1] == 2:
                prob_class_0_base = float(proba_result[0][0])
                prob_class_1_base = float(proba_result[0][1])
            elif proba_result.shape[1] == 1:
                # Single class output (unlikely but handle it)
                prob_class_1_base = float(proba_result[0][0])
                prob_class_0_base = 1.0 - prob_class_1_base
            else:
                # Multi-class or unexpected format
                model = self.pipeline.named_steps.get("model")
                if hasattr(model, "classes_"):
                    # Find index of class 1
                    class_1_idx = None
                    for idx, cls in enumerate(model.classes_):
                        if cls == 1:
                            class_1_idx = idx
                            break
                    if class_1_idx is not None:
                        prob_class_1_base = float(proba_result[0][class_1_idx])
                        prob_class_0_base = 1.0 - prob_class_1_base
                    else:
                        # Fallback: use last column
                        prob_class_1_base = float(proba_result[0][-1])
                        prob_class_0_base = float(proba_result[0][0])
                else:
                    # Fallback: assume binary classification
                    prob_class_1_base = float(proba_result[0][-1])
                    prob_class_0_base = float(proba_result[0][0])
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"  Row shape: {row.shape}")
            print(f"  Row columns: {list(row.columns)}")
            print(f"  Expected feature columns: {self.feature_columns}")
            print(f"  Row dtypes:\n{row.dtypes}")
            raise ValueError(error_msg) from e
        
        # Apply domain knowledge corrections if requested
        if apply_domain_knowledge:
            prob_class_1_corrected, adjustments = self._apply_domain_knowledge_corrections(
                input_dict, prob_class_1_base
            )
            # Use corrected probability
            prob_class_1 = prob_class_1_corrected
            prob_class_0 = 1.0 - prob_class_1
            domain_knowledge_applied = True
            domain_adjustments = {adj[0]: adj[1] for adj in adjustments}
        else:
            prob_class_1 = prob_class_1_base
            prob_class_0 = prob_class_0_base
            domain_knowledge_applied = False
            domain_adjustments = {}
        
        # Use prob_class_1 as risk score
        risk_score = prob_class_1
        risk_percentage = risk_score * 100
        prediction = int(prob_class_1 > 0.5)
        
        # Determine risk level
        if risk_percentage < 25:
            risk_level = "Low"
        elif risk_percentage < 50:
            risk_level = "Moderate"
        elif risk_percentage < 75:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Get SHAP values
        try:
            # Check if explainer is available
            if self.explainer is None:
                # Try to initialize explainer if model is trained
                model = self.pipeline.named_steps.get("model")
                if model is not None and (hasattr(model, 'feature_importances_') or hasattr(model, 'estimators_')):
                    try:
                        print("Initializing SHAP explainer for prediction...")
                        self.explainer = shap.TreeExplainer(model)
                        print("SHAP explainer initialized successfully")
                    except Exception as e:
                        print(f"Warning: Could not initialize SHAP explainer: {e}")
                        feature_importance = []
                        explanation = f"Risk assessment: {risk_level} risk ({risk_percentage:.2f}%)"
                        # Continue without SHAP
                        result = {
                            "risk_score": risk_score,
                            "risk_percentage": round(risk_percentage, 2),
                            "risk_level": risk_level,
                            "prediction": prediction,
                            "feature_importance": feature_importance,
                            "explanation": explanation,
                            "probabilities": {
                                "no_depression": round(prob_class_0 * 100, 2),
                                "depression": round(prob_class_1 * 100, 2)
                            },
                            "domain_knowledge_applied": domain_knowledge_applied
                        }
                        if domain_knowledge_applied:
                            result["base_probability"] = round(prob_class_1_base * 100, 2)
                            result["corrected_probability"] = round(prob_class_1 * 100, 2)
                            result["domain_adjustments"] = domain_adjustments
                        return result
                else:
                    raise ValueError("SHAP explainer not initialized. Model may not be trained yet.")
            
            # Preprocess the row
            preprocessor = self.pipeline.named_steps.get("preprocess")
            if preprocessor is None:
                raise ValueError("Preprocessor not found in pipeline")
            
            row_processed = preprocessor.transform(row)
            
            # Get SHAP values
            try:
                shap_values = self.explainer.shap_values(row_processed)
            except Exception as e:
                # If SHAP fails, continue without it
                print(f"Warning: SHAP explanation failed: {e}")
                feature_importance = []
                explanation = f"Risk assessment: {risk_level} risk ({risk_percentage:.2f}%)"
                # Continue without SHAP
                result = {
                    "risk_score": risk_score,
                    "risk_percentage": round(risk_percentage, 2),
                    "risk_level": risk_level,
                    "prediction": prediction,
                    "feature_importance": feature_importance,
                    "explanation": explanation,
                    "probabilities": {
                        "no_depression": round(prob_class_0 * 100, 2),
                        "depression": round(prob_class_1 * 100, 2)
                    },
                    "domain_knowledge_applied": domain_knowledge_applied
                }
                if domain_knowledge_applied:
                    result["base_probability"] = round(prob_class_1_base * 100, 2)
                    result["corrected_probability"] = round(prob_class_1 * 100, 2)
                    result["domain_adjustments"] = domain_adjustments
                return result
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Convert to numpy array and ensure 2D shape
            shap_values = np.array(shap_values)
            
            # Handle different shapes - flatten to 2D (n_samples, n_features)
            if len(shap_values.shape) == 1:
                # If 1D, reshape to (1, n_features)
                shap_values = shap_values.reshape(1, -1)
            elif len(shap_values.shape) == 3:
                # If 3D (n_samples, n_features, n_classes), take last class
                shap_values = shap_values[:, :, -1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
            elif len(shap_values.shape) > 2:
                # Flatten any extra dimensions
                shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
            # Get feature names after preprocessing
            feature_names = preprocessor.get_feature_names_out(self.feature_columns)
            
            # Get top 5 features (use first row of shap_values, ensure it's 1D)
            shap_values_row = np.array(shap_values[0]).flatten()  # Ensure 1D array
            
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
            
            # Create list of (index, feature_name, shap_value) tuples, filtering inactive one-hot features
            feat_candidates = []
            for idx in range(len(feature_names)):
                feature_name = feature_names[idx]
                
                # Ensure shap_val is numeric
                try:
                    shap_val = float(shap_values_row[idx])
                except (ValueError, TypeError, IndexError):
                    # Skip if we can't convert to float
                    continue
                
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
                
                feat_candidates.append((idx, feature_name, shap_val))
            
            # Get top 5 features by absolute SHAP value (from active features only)
            # If filtering removed all features, fall back to showing all features
            if len(feat_candidates) == 0:
                # Fallback: show top features without filtering
                shap_abs = np.abs(shap_values_row)
                top_indices = np.argsort(shap_abs)[-5:][::-1]
                top_candidates = [(idx, feature_names[idx], shap_values_row[idx]) for idx in top_indices]
            else:
                top_candidates = sorted(feat_candidates, key=lambda x: -abs(x[2]))[:5]
            
            feature_importance = []
            for idx, feature_name, shap_val_scalar in top_candidates:
                # Ensure shap_value is numeric
                try:
                    shap_value = float(shap_val_scalar)
                except (ValueError, TypeError):
                    continue  # Skip if not numeric
                
                abs_val_scalar = abs(shap_value)
                # SHAP value interpretation:
                # Positive SHAP = increases risk (pushes prediction toward class 1)
                # Negative SHAP = decreases risk (pushes prediction toward class 0)
                impact = "increases" if shap_value > 0 else "decreases"
                feature_importance.append({
                    "feature": feature_name,
                    "shap_value": shap_value,
                    "impact": impact,
                    "abs_contribution": float(abs_val_scalar)
                })
            
            # Generate personalized explanation
            explanation = self._generate_explanation(risk_level, risk_percentage, feature_importance)
            
        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            feature_importance = []
            explanation = f"Risk assessment: {risk_level} risk ({risk_percentage:.2f}%)"
        
        result = {
            "risk_score": risk_score,
            "risk_percentage": round(risk_percentage, 2),
            "risk_level": risk_level,
            "prediction": prediction,
            "feature_importance": feature_importance,
            "explanation": explanation,
            "probabilities": {
                "no_depression": round(prob_class_0 * 100, 2),
                "depression": round(prob_class_1 * 100, 2)
            },
            "domain_knowledge_applied": domain_knowledge_applied
        }
        
        if domain_knowledge_applied:
            result["base_probability"] = round(prob_class_1_base * 100, 2)
            result["corrected_probability"] = round(prob_class_1 * 100, 2)
            result["domain_adjustments"] = domain_adjustments
            # Update explanation to mention corrections
            if domain_adjustments:
                adj_summary = ", ".join([f"{k}: {v:+.1%}" for k, v in list(domain_adjustments.items())[:3]])
                result["explanation"] += f"\n\nNote: Prediction adjusted based on domain knowledge ({adj_summary}...)"
        
        return result
    
    def batch_predict(self, input_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Predict for multiple inputs at once.
        
        Args:
            input_list: List of dictionaries with feature names as keys
        
        Returns:
            List of prediction results
        """
        return [self.predict_from_dict(input_dict) for input_dict in input_list]
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Get OpenAI Function Calling schema for prediction tool.
        
        Returns:
            Dictionary with function schema for predict_ppd_risk
        """
        return {
            "name": "predict_ppd_risk",
            "description": "Predicts postpartum depression (PPD) risk based on patient symptoms and demographics. Returns risk score, level, and explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "string",
                        "description": "Age group (e.g., '25-30', '30-35', '35-40', etc.)"
                    },
                    "feeling_sad": {
                        "type": "string",
                        "enum": ["Yes", "No", "Sometimes"],
                        "description": "Feeling sad or tearful"
                    },
                    "irritable": {
                        "type": "string",
                        "enum": ["Yes", "No", "Sometimes"],
                        "description": "Irritable towards baby & partner"
                    },
                    "trouble_sleeping": {
                        "type": "string",
                        "enum": ["Yes", "No", "Two or more days a week"],
                        "description": "Trouble sleeping at night"
                    },
                    "concentration": {
                        "type": "string",
                        "enum": ["Yes", "No", "Often"],
                        "description": "Problems concentrating or making decision"
                    },
                    "appetite": {
                        "type": "string",
                        "enum": ["Yes", "No", "Not at all"],
                        "description": "Overeating or loss of appetite"
                    },
                    "feeling_anxious": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "description": "Feeling anxious"
                    },
                    "guilt": {
                        "type": "string",
                        "enum": ["Yes", "No", "Maybe"],
                        "description": "Feeling of guilt"
                    },
                    "bonding": {
                        "type": "string",
                        "enum": ["Yes", "No", "Sometimes"],
                        "description": "Problems of bonding with baby"
                    },
                    "suicide_attempt": {
                        "type": "string",
                        "enum": ["Yes", "No", "Not interested to say"],
                        "description": "Suicide attempt"
                    }
                },
                "required": ["age", "feeling_sad", "irritable", "trouble_sleeping", 
                           "concentration", "appetite", "feeling_anxious", "guilt", 
                           "bonding", "suicide_attempt"]
            }
        }
    
    def get_training_tool_schema(self) -> Dict[str, Any]:
        """
        Get OpenAI Function Calling schema for Random Forest training tool.
        
        Returns:
            Dictionary with function schema for train_random_forest
        """
        return {
            "name": "train_random_forest",
            "description": "Trains a Random Forest model for postpartum depression prediction. Requires training data (X_train, y_train) and optional test data. Updates the agent's pipeline with the trained model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_estimators": {
                        "type": "integer",
                        "description": "Number of trees in the forest (default: 100)",
                        "default": 100
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth of trees (default: 10)",
                        "default": 10
                    },
                    "min_samples_split": {
                        "type": "integer",
                        "description": "Minimum samples required to split a node (default: 2)",
                        "default": 2
                    },
                    "min_samples_leaf": {
                        "type": "integer",
                        "description": "Minimum samples required in a leaf node (default: 1)",
                        "default": 1
                    },
                    "max_features": {
                        "type": "string",
                        "enum": ["sqrt", "log2", "auto", "None"],
                        "description": "Number of features to consider for best split (default: 'sqrt')",
                        "default": "sqrt"
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Proportion of dataset to include in test split (default: 0.2)",
                        "default": 0.2,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "random_state": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (default: 42)",
                        "default": 42
                    }
                },
                "required": []
            }
        }
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tool schemas (both prediction and training).
        
        Returns:
            List of tool schema dictionaries
        """
        return [
            self.get_tool_schema(),
            self.get_training_tool_schema()
        ]
    
    def save(self, filepath: str):
        """Save the agent to a file."""
        agent_data = {
            "pipeline": self.pipeline,
            "X_train": self.X_train,
            "cat_cols": self.cat_cols,
            "feature_columns": self.feature_columns,
            "feature_dtypes": self.feature_dtypes
        }
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        print(f"Agent saved to {filepath}")
    
    def train_random_forest(self, 
                           X_train: pd.DataFrame = None,
                           y_train: pd.Series = None,
                           X_test: pd.DataFrame = None,
                           y_test: pd.Series = None,
                           test_size: float = 0.2,
                           random_state: int = 42,
                           n_estimators: int = 100,
                           max_depth: int = 10,
                           min_samples_split: int = 2,
                           min_samples_leaf: int = 1,
                           max_features: str = "sqrt",
                           n_jobs: int = -1) -> Dict[str, Any]:
        """
        Train a Random Forest model and update the agent's pipeline.
        
        Args:
            X_train: Training features (if None, uses agent's X_train)
            y_train: Training labels (required if X_train provided)
            X_test: Test features (optional, will split if not provided)
            y_test: Test labels (optional, will split if not provided)
            test_size: Test set size if splitting (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of trees (default: 10)
            min_samples_split: Minimum samples to split a node (default: 2)
            min_samples_leaf: Minimum samples in a leaf (default: 1)
            max_features: Number of features to consider for best split (default: "sqrt")
            n_jobs: Number of parallel jobs (default: -1)
        
        Returns:
            Dictionary with training results including:
            - success: Whether training was successful
            - roc_auc: ROC AUC score on test set
            - classification_report: Classification report text
            - message: Status message
        """
        try:
            # Use agent's training data if not provided
            if X_train is None:
                # Need to get y_train from somewhere - this is a limitation
                # For now, we'll require both X_train and y_train
                raise ValueError("X_train and y_train must be provided for training")
            
            if y_train is None:
                raise ValueError("y_train must be provided for training")
            
            # Split data if test set not provided
            if X_test is None or y_test is None:
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                    X_train, y_train, test_size=test_size, 
                    stratify=y_train, random_state=random_state
                )
            else:
                X_train_split = X_train
                X_test_split = X_test
                y_train_split = y_train
                y_test_split = y_test
            
            # 🔧 CRITICAL: Update cat_cols based on actual columns in X_train_split BEFORE creating pipeline
            # This ensures we only use categorical columns that actually exist in the training data
            X_train_cols = set(X_train_split.columns)
            actual_cat_cols = [col for col in X_train_split.columns if X_train_split[col].dtype == "object"]
            
            # Filter self.cat_cols to only include columns that exist in X_train_split
            valid_cat_cols = [col for col in self.cat_cols if col in X_train_cols]
            
            # If there's a mismatch, use the actual categorical columns from the data
            if set(valid_cat_cols) != set(actual_cat_cols):
                print(f"  ⚠️  Warning: cat_cols mismatch detected. Updating to match actual data.")
                print(f"     Old cat_cols: {self.cat_cols}")
                print(f"     Actual cat_cols in data: {actual_cat_cols}")
                valid_cat_cols = actual_cat_cols
            
            # Use the validated cat_cols for pipeline creation
            cat_cols_for_pipeline = valid_cat_cols
            
            # Create Random Forest pipeline
            print("Creating Random Forest pipeline...")
            # Handle max_features: convert "None" string to None, "auto" to "sqrt"
            max_features_param = None if max_features == "None" else ("sqrt" if max_features == "auto" else max_features)
            
            rf_pipeline = create_rf_pipeline(
                cat_cols=cat_cols_for_pipeline,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features_param,
                n_jobs=n_jobs,
                random_state=random_state
            )
            
            # Train and evaluate
            print("Training Random Forest model...")
            y_proba, y_pred, roc_auc = train_and_evaluate(
                rf_pipeline, X_train_split, y_train_split, X_test_split, y_test_split
            )
            
            # Update agent's pipeline and training data
            self.pipeline = rf_pipeline
            self.X_train = X_train_split
            
            # Update feature columns and dtypes based on new training data
            self.feature_columns = list(X_train_split.columns)
            self.feature_columns = [col for col in self.feature_columns if col not in ["PPD", "PPD_Composite"]]
            self.feature_dtypes = {col: str(X_train_split[col].dtype) for col in self.feature_columns}
            
            # Update cat_cols to match actual dtypes in X_train_split (already validated above)
            self.cat_cols = cat_cols_for_pipeline
            print(f"  ✅ Updated cat_cols to: {self.cat_cols}")
            
            # Reinitialize SHAP explainer with new model
            print("Reinitializing SHAP explainer with Random Forest model...")
            self.explainer = shap.TreeExplainer(self.pipeline.named_steps["model"])
            print("SHAP explainer ready!")
            
            return {
                "success": True,
                "roc_auc": float(roc_auc),
                "message": f"Random Forest model trained successfully! ROC AUC: {roc_auc:.4f}",
                "model_type": "RandomForest",
                "parameters": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features
                },
                "pipeline": rf_pipeline,
                "X_train": X_train_split,
                "X_test": X_test_split,
                "y_train": y_train_split,
                "y_test": y_test_split,
                "y_proba": y_proba,
                "y_pred": y_pred
            }
            
        except Exception as e:
            return {
                "success": False,
                "roc_auc": None,
                "message": f"Training failed: {str(e)}",
                "error": str(e)
            }
    
    def train_xgboost(self,
                     X_train: pd.DataFrame = None,
                     y_train: pd.Series = None,
                     X_test: pd.DataFrame = None,
                     y_test: pd.Series = None,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     use_optimization: bool = False,
                     n_iter: int = 30,
                     cv: int = 3,
                     scoring: str = 'roc_auc',
                     n_jobs: int = -1) -> Dict[str, Any]:
        """
        Train an XGBoost model and update the agent's pipeline.
        
        Args:
            X_train: Training features (required)
            y_train: Training labels (required)
            X_test: Test features (optional, will split if not provided)
            y_test: Test labels (optional, will split if not provided)
            test_size: Test set size if splitting (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            use_optimization: Whether to use RandomizedSearchCV for hyperparameter optimization (default: False)
            n_iter: Number of iterations for optimization (default: 30)
            cv: Number of cross-validation folds (default: 3)
            scoring: Scoring metric for optimization (default: 'roc_auc')
            n_jobs: Number of parallel jobs (default: -1)
        
        Returns:
            Dictionary with training results including:
            - success: Whether training was successful
            - roc_auc: ROC AUC score on test set
            - message: Status message
            - model_type: "XGBoost"
            - parameters: Best hyperparameters if optimization used
        """
        try:
            if X_train is None:
                raise ValueError("X_train must be provided for training")
            
            if y_train is None:
                raise ValueError("y_train must be provided for training")
            
            # Split data if test set not provided
            if X_test is None or y_test is None:
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                    X_train, y_train, test_size=test_size,
                    stratify=y_train, random_state=random_state
                )
            else:
                X_train_split = X_train
                X_test_split = X_test
                y_train_split = y_train
                y_test_split = y_test
            
            # 🔧 CRITICAL: Update cat_cols based on actual columns in X_train_split BEFORE creating pipeline
            # This ensures we only use categorical columns that actually exist in the training data
            X_train_cols = set(X_train_split.columns)
            actual_cat_cols = [col for col in X_train_split.columns if X_train_split[col].dtype == "object"]
            
            # Filter self.cat_cols to only include columns that exist in X_train_split
            valid_cat_cols = [col for col in self.cat_cols if col in X_train_cols]
            
            # If there's a mismatch, use the actual categorical columns from the data
            if set(valid_cat_cols) != set(actual_cat_cols):
                print(f"  ⚠️  Warning: cat_cols mismatch detected. Updating to match actual data.")
                print(f"     Old cat_cols: {self.cat_cols}")
                print(f"     Actual cat_cols in data: {actual_cat_cols}")
                valid_cat_cols = actual_cat_cols
            
            # Use the validated cat_cols for pipeline creation
            cat_cols_for_pipeline = valid_cat_cols
            
            # Create and train XGBoost pipeline
            if use_optimization:
                print("Creating XGBoost pipeline with hyperparameter optimization...")
                print("⏱ This may take several minutes...")
                xgb_pipeline, best_params, cv_results = optimize_XGBoost_hyperparameters(
                    X_train_split, y_train_split, cat_cols_for_pipeline,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    random_state=random_state,
                    n_jobs=n_jobs
                )
                optimization_info = "Hyperparameter optimization applied"
            else:
                print("Creating XGBoost pipeline with default parameters...")
                xgb_pipeline = create_XGBoost_pipeline(cat_cols_for_pipeline)
                best_params = {}
                optimization_info = "Using default hyperparameters"
            
            # Train and evaluate
            print("Training XGBoost model...")
            y_proba, y_pred, roc_auc = train_and_evaluate(
                xgb_pipeline, X_train_split, y_train_split, X_test_split, y_test_split
            )
            
            # Update agent's pipeline and training data
            self.pipeline = xgb_pipeline
            self.X_train = X_train_split
            
            # Update feature columns and dtypes based on new training data
            self.feature_columns = list(X_train_split.columns)
            self.feature_columns = [col for col in self.feature_columns if col not in ["PPD", "PPD_Composite"]]
            self.feature_dtypes = {col: str(X_train_split[col].dtype) for col in self.feature_columns}
            
            # Update cat_cols to match actual dtypes in X_train_split (already validated above)
            self.cat_cols = cat_cols_for_pipeline
            print(f"  ✅ Updated cat_cols to: {self.cat_cols}")
            
            # Reinitialize SHAP explainer with new model
            print("Reinitializing SHAP explainer with XGBoost model...")
            self.explainer = shap.TreeExplainer(self.pipeline.named_steps["model"])
            print("SHAP explainer ready!")
            
            return {
                "success": True,
                "roc_auc": float(roc_auc),
                "message": f"XGBoost model trained successfully! ROC AUC: {roc_auc:.4f}",
                "model_type": "XGBoost",
                "optimization_info": optimization_info,
                "parameters": best_params,
                "pipeline": xgb_pipeline,
                "X_train": X_train_split,
                "X_test": X_test_split,
                "y_train": y_train_split,
                "y_test": y_test_split,
                "y_proba": y_proba,
                "y_pred": y_pred
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "roc_auc": None,
                "message": f"Training failed: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    @classmethod
    def load(cls, filepath: str):
        """Load an agent from a file."""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        agent = cls(
            pipeline=agent_data["pipeline"],
            X_train=agent_data["X_train"],
            cat_cols=agent_data["cat_cols"],
            feature_columns=agent_data.get("feature_columns")
        )
        print(f"Agent loaded from {filepath}")
        return agent


def create_agent_from_training(pipeline, X_train, cat_cols, feature_columns=None):
    """
    Factory function to create a PPD Agent from training results.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_train: Training data
        cat_cols: List of categorical column names
        feature_columns: Optional list of feature column names
    
    Returns:
        PPDAgent instance
    """
    return PPDAgent(pipeline, X_train, cat_cols, feature_columns)

