"""
Script to retrain and save the PPD agent with the current sklearn version.
This fixes version incompatibility issues when loading the agent.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from MLmodel import create_XGBoost_pipeline, train_and_evaluate
from ppd_agent import create_agent_from_training

print("=" * 60)
print("Retraining PPD Agent (fixing sklearn version compatibility)")
print("=" * 60)

# Load the data from multiple CSV files
print("\nLoading data from multiple CSV files...")
data_dir = Path("data")

# Load Demographics.csv (full)
demographics = pd.read_csv(data_dir / "Demographics.csv")
print(f"Demographics shape: {demographics.shape}")

# Load EPDS_answers.csv (only Name, Total Scores, מחשבות פגיעה עצמית columns)
epds_columns = ["ID", "Name", "Total Scores", "מחשבות פגיעה עצמית"]
epds = pd.read_csv(data_dir / "EPDS_answers.csv", usecols=epds_columns)
print(f"EPDS_answers shape: {epds.shape}")

# Load Clinical_data.csv (full)
clinical = pd.read_csv(data_dir / "Clinical_data.csv")
print(f"Clinical_data shape: {clinical.shape}")

# Load Psychiatric_data.csv (full)
psychiatric = pd.read_csv(data_dir / "Psychiatric_data.csv")
print(f"Psychiatric_data shape: {psychiatric.shape}")

# Load Functional_Psychosocial_data.csv (full)
functional = pd.read_csv(data_dir / "Functional_Psychosocial_data.csv")
print(f"Functional_Psychosocial_data shape: {functional.shape}")

# Merge all dataframes on ID (foreign key)
print("\nMerging dataframes on ID (foreign key)...")

# Validate ID integrity before merging
print("Validating ID integrity...")
demographics_ids = set(demographics['ID'].unique())
epds_ids = set(epds['ID'].unique())
clinical_ids = set(clinical['ID'].unique())
psychiatric_ids = set(psychiatric['ID'].unique())
functional_ids = set(functional['ID'].unique())

print(f"  Demographics: {len(demographics_ids)} unique IDs")
print(f"  EPDS_answers: {len(epds_ids)} unique IDs")
print(f"  Clinical_data: {len(clinical_ids)} unique IDs")
print(f"  Psychiatric_data: {len(psychiatric_ids)} unique IDs")
print(f"  Functional_Psychosocial_data: {len(functional_ids)} unique IDs")

# Check for duplicate IDs within each table
for name, df_check in [("Demographics", demographics), ("EPDS_answers", epds), 
                        ("Clinical_data", clinical), ("Psychiatric_data", psychiatric),
                        ("Functional_Psychosocial_data", functional)]:
    duplicates = df_check['ID'].duplicated().sum()
    if duplicates > 0:
        print(f"  ⚠️  Warning: {duplicates} duplicate IDs found in {name}")

# Perform inner joins on ID (foreign key) - keeps only records with matching IDs across all tables
df = demographics.copy()
df = df.merge(epds, on="ID", how="inner", suffixes=("", "_epds"), validate="one_to_one")
df = df.merge(clinical, on="ID", how="inner", validate="one_to_one")
df = df.merge(psychiatric, on="ID", how="inner", validate="one_to_one")
df = df.merge(functional, on="ID", how="inner", validate="one_to_one")

# Handle duplicate Name column
if "Name_epds" in df.columns:
    df.drop(columns=["Name_epds"], inplace=True)

print(f"Merged dataframe shape: {df.shape}")
print(f"Final unique IDs: {df['ID'].nunique()}")

# Drop rows with missing values
print("Dropping rows with missing values...")
df.dropna(axis=0, inplace=True)

# Create composite target based on EPDS Total Scores and self-harm thoughts
print("Creating composite target...")
target = "PPD_Composite"

# Convert Total Scores to numeric if it's not already
df['Total Scores'] = pd.to_numeric(df['Total Scores'], errors='coerce')
df['מחשבות פגיעה עצמית'] = pd.to_numeric(df['מחשבות פגיעה עצמית'], errors='coerce')

# Create composite target: PPD = 1 if Total Scores >= 13 (Likely PPD) OR self-harm thoughts > 0
# EPDS scoring: >= 13 indicates Likely PPD, 11-12 indicates Mild depression or dejection, <= 10 indicates Low PPD risk
epds_threshold = 13
df[target] = ((df['Total Scores'] >= epds_threshold) | 
              (df['מחשבות פגיעה עצמית'] > 0)).astype(int)

print(f"Target distribution: {df[target].value_counts().to_dict()}")

# Final dropna check
df = df.dropna()

# Ensure Age is numeric (convert if needed)
if 'Age' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    print(f"  Age column converted to numeric: {df['Age'].dtype}")

# Prepare features and target (drop ID and Name as they are identifiers)
# ID is used for data merging only, Name is only for display purposes
# Also drop Total Scores and מחשבות פגיעה עצמית as they are used for target creation, not as features
X = df.drop(columns=[target, 'ID', 'Name', 'Total Scores', 'מחשבות פגיעה עצמית'], errors='ignore')
y = df[target]

# Validate that ID and Name are not in features
if 'ID' in X.columns or 'Name' in X.columns:
    raise ValueError("ERROR: ID or Name columns found in features! They should not be used for model training.")

# 🧩 Identify categorical and numeric features AFTER creating X (to ensure we only use features in X)
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
numeric_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]

print(f"\n📊 Feature Analysis:")
print(f"  Total features: {X.shape[1]}")
print(f"  Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"  Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"  Feature columns: {list(X.columns)}")

print(f"\n✅ Final feature set:")
print(f"  Features shape: {X.shape}")

# Split the data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Create and train the model
print("\nTraining XGBoost model...")
pipeline = create_XGBoost_pipeline(cat_cols)
y_proba, y_pred, roc_auc = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)
print(f"Model trained! ROC-AUC: {roc_auc:.4f}")

# Create and save the agent
print("\nCreating PPD Agent...")
ppd_agent = create_agent_from_training(pipeline, X_train, cat_cols, list(X_train.columns))
print("PPD Agent created!")

# Save agent
import os
os.makedirs("output/agents", exist_ok=True)
agent_path = "output/agents/ppd_agent.pkl"
print(f"\nSaving agent to {agent_path}...")
ppd_agent.save(agent_path)
print(f"✅ Agent saved successfully to {agent_path}!")

print("\n" + "=" * 60)
print("Done! The agent is now compatible with your current sklearn version.")
print("You can now restart your API server and the agent should load correctly.")
print("=" * 60)

