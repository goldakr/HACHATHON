"""
Script to explain the structure of ppd_agent.pkl file
"""

import pickle
import pandas as pd
import os
import sys

print("=" * 70)
print("PPD Agent Pickle File Structure Explanation")
print("=" * 70)

try:
    # Try to load the pickle file
    agent_file = "output/agents/ppd_agent.pkl"
    if not os.path.exists(agent_file):
        print(f"ERROR: {agent_file} file not found!")
        print("Please run main.py first to create the agent.")
        sys.exit(1)
    
    with open(agent_file, "rb") as f:
        agent_data = pickle.load(f)
    
    print("\n1. FILE FORMAT:")
    print("   - Format: Python Pickle (.pkl)")
    print("   - Serialization: Binary format using pickle.dump()")
    print("   - Purpose: Stores trained machine learning model and metadata")
    
    print("\n2. FILE CONTENTS (Dictionary with 5 keys):")
    print("   " + "-" * 66)
    
    # Explain each component
    print("\n   a) 'pipeline' (sklearn Pipeline object):")
    print("      - Contains the complete ML pipeline:")
    print("        * Preprocessing step: ColumnTransformer with OneHotEncoder")
    print("          (converts categorical features to numerical)")
    print("        * Model step: XGBoost Classifier")
    print("          (trained XGBoost model for PPD prediction)")
    print(f"      - Type: {type(agent_data['pipeline'])}")
    print(f"      - Steps: {list(agent_data['pipeline'].named_steps.keys())}")
    
    print("\n   b) 'X_train' (pandas DataFrame):")
    print("      - Training dataset used to train the model")
    print("      - Required for SHAP explainer initialization")
    print("      - Contains all feature columns (Age, symptoms, etc.)")
    print(f"      - Shape: {agent_data['X_train'].shape}")
    print(f"      - Columns: {list(agent_data['X_train'].columns)[:5]}...")
    print(f"      - Type: {type(agent_data['X_train'])}")
    
    print("\n   c) 'cat_cols' (list of strings):")
    print("      - List of categorical column names")
    print("      - These columns are one-hot encoded during preprocessing")
    print(f"      - Value: {agent_data['cat_cols']}")
    print(f"      - Count: {len(agent_data['cat_cols'])} columns")
    
    print("\n   d) 'feature_columns' (list of strings):")
    print("      - Ordered list of all feature column names")
    print("      - Used to ensure correct feature order during prediction")
    print(f"      - Value: {agent_data.get('feature_columns', 'Not found')}")
    if 'feature_columns' in agent_data:
        print(f"      - Count: {len(agent_data['feature_columns'])} columns")
    
    print("\n   e) 'feature_dtypes' (dictionary):")
    print("      - Maps feature column names to their data types")
    print("      - Used for data validation and type checking")
    if 'feature_dtypes' in agent_data:
        print(f"      - Sample: {dict(list(agent_data['feature_dtypes'].items())[:3])}")
        print(f"      - Total features: {len(agent_data['feature_dtypes'])}")
    else:
        print("      - Not found in this version")
    
    print("\n3. WHY THESE COMPONENTS ARE NEEDED:")
    print("   - 'pipeline': Makes predictions on new patient data")
    print("   - 'X_train': Required for SHAP explainer to calculate feature importance")
    print("   - 'cat_cols': Ensures categorical features are properly encoded")
    print("   - 'feature_columns': Maintains correct feature order")
    print("   - 'feature_dtypes': Validates input data types")
    
    print("\n4. FILE SIZE:")
    file_size = os.path.getsize(agent_file)
    print(f"   - Size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print("   - Contains: Trained model weights, training data, metadata")
    
    print("\n5. USAGE:")
    print(f"   - Loaded by PPDAgent.load('{agent_file}')")
    print("   - Automatically loaded when API server starts")
    print("   - Used for making PPD risk predictions")
    print("   - Provides SHAP explanations for predictions")
    
    print("\n" + "=" * 70)
    print("Note: The SHAP explainer is NOT saved in the pickle file.")
    print("It is re-initialized when the agent is loaded because:")
    print("- SHAP explainers are lightweight and quick to create")
    print("- They reference the model from the pipeline")
    print("- This keeps the file size smaller")
    print("=" * 70)
    
except FileNotFoundError:
    print(f"ERROR: {agent_file} file not found!")
    print("Please run retrain_agent.py first to create the file.")
except Exception as e:
    print(f"ERROR loading file: {e}")
    import traceback
    traceback.print_exc()

