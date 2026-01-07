# 📌 Standard libraries
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from MLmodel import create_XGBoost_pipeline, train_and_evaluate
from visualization import create_all_visualizations
from gradio_app import create_gradio_interface
from ppd_agent import create_agent_from_training

print("Welcome to the Postpartum Depression Prediction Agent Tool")

# 🗂 Load the PostPartum Depression dataset from multiple CSV files
# Get the script directory and construct path relative to it
script_dir = Path(__file__).parent
data_dir = script_dir / "data"

print("Loading data from multiple CSV files...")

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

# Handle duplicate Name column (keep the one from Demographics, drop _epds suffix if exists)
# Note: Name is kept in df for display purposes only, but will be excluded from model training
if "Name_epds" in df.columns:
    df.drop(columns=["Name_epds"], inplace=True)

print(f"\nMerged dataframe shape: {df.shape}")
print(f"Final unique IDs: {df['ID'].nunique()}")
print("Note: ID is used for data merging only, Name is for display only - both will be excluded from model training")

# 📊 Show basic info
print("\nDataframe info:")
print(df.head())
print(df.info())

# Check the unique values in each column of data
print("\nChecking unique values in key columns:")
for column in df.columns[:10]:  # Show first 10 columns
    unique_vals = df[column].unique()[:10]  # Show first 10 unique values
    print(f"{column}: {unique_vals}")

# Drop the rows with missing values
print("\nDropping rows with missing values...")
initial_shape = df.shape
df.dropna(axis=0, inplace=True)
print(f"Dropped {initial_shape[0] - df.shape[0]} rows with missing values")

# 🧩 Use PPD field from Psychiatric_data.csv as binary target value
target = "PPD"

# Convert PPD column from "Yes"/"No" to binary (1/0)
if target not in df.columns:
    raise ValueError(f"PPD column not found in merged dataframe. Available columns: {list(df.columns)}")

# Map "Yes" -> 1, "No" -> 0, handling case-insensitive values and any whitespace
df[target] = df[target].astype(str).str.strip().str.lower()
df[target] = df[target].map({'yes': 1, 'no': 0}).astype(int)

print(f"\n📊 PPD Target Distribution (from Psychiatric_data.csv):")
print(df[target].value_counts())
print(f"Proportions: {df[target].value_counts(normalize=True).to_dict()}")

# 🧪 Handle missing values (final check)
df = df.dropna()

# Ensure Age is numeric (convert if needed)
if 'Age' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    print(f"  Age column converted to numeric: {df['Age'].dtype}")

# Drop ID and Name from features (they are identifiers, not features)
# ID is used for data merging only, Name is only for display purposes
# Drop PPD as it is the target variable
X = df.drop(columns=[target, 'ID', 'Name'], errors='ignore')
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
print(f"  Feature dtypes:\n{X.dtypes}")

print(f"\n✅ Final feature set:")
print(f"  Target distribution: {y.value_counts().to_dict()}")
print(f"  Target proportions: {y.value_counts(normalize=True).to_dict()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 🧠 Create untrained pipeline (will be trained when user clicks "Start Train Model" button)
print("\n" + "="*50)
print("Preparing Model Interface")
print("="*50)
print("💡 Model will be trained when you click 'Start Train Model' in the Gradio interface.")

# Create untrained pipeline (just for initialization - will be trained via Gradio)
print(f"\n🔧 Initializing model pipeline...")
print(f"  Categorical columns for OneHotEncoder: {cat_cols}")
print(f"  Numeric columns (will pass through): {numeric_cols}")
print(f"  Total feature columns: {len(X.columns)}")

# Validate that all categorical columns are in X
missing_cat_cols = [col for col in cat_cols if col not in X.columns]
if missing_cat_cols:
    print(f"  ⚠️  Warning: Some categorical columns not found in X: {missing_cat_cols}")
    cat_cols = [col for col in cat_cols if col in X.columns]

# Validate that all columns are accounted for
all_feature_cols = set(cat_cols + numeric_cols)
X_cols = set(X.columns)
if all_feature_cols != X_cols:
    missing = X_cols - all_feature_cols
    extra = all_feature_cols - X_cols
    if missing:
        print(f"  ⚠️  Warning: Some X columns not categorized: {missing}")
    if extra:
        print(f"  ⚠️  Warning: Some categorized columns not in X: {extra}")

pipeline = create_XGBoost_pipeline(cat_cols)
print(f"  ✅ Pipeline initialized successfully!")

# Test pipeline with a sample row to ensure it works correctly
try:
    print(f"\n🧪 Testing pipeline with sample data...")
    sample_row = X_train.iloc[[0]]  # Get first row
    print(f"  Sample row shape: {sample_row.shape}")
    print(f"  Sample row columns: {list(sample_row.columns)}")
    
    # Try to transform the sample (this will fail if pipeline is misconfigured)
    # Note: We can't call transform on untrained pipeline, but we can check the structure
    preprocessor = pipeline.named_steps.get("preprocess")
    if preprocessor is not None:
        print(f"  ✅ Preprocessor found: {type(preprocessor).__name__}")
        print(f"  Preprocessor transformers: {[t[0] for t in preprocessor.transformers]}")
        if hasattr(preprocessor, 'remainder'):
            print(f"  Remainder handling: {preprocessor.remainder}")
    else:
        print(f"  ⚠️  Warning: No preprocessor found in pipeline")
    
    model = pipeline.named_steps.get("model")
    if model is not None:
        print(f"  ✅ Model found: {type(model).__name__}")
    else:
        print(f"  ⚠️  Warning: No model found in pipeline")
        
    print(f"  ✅ Pipeline structure validated!")
except Exception as e:
    print(f"  ⚠️  Warning during pipeline validation: {e}")

# Initialize variables as None (will be set after training)
y_proba = None
y_pred = None
roc_auc = None
ppd_agent = None

# 🚀 Launch Gradio Interface
print("\n" + "="*50)
print("Launching Gradio Web Interface...")
print("="*50)

# Create Gradio interface (model will be trained when user clicks "Start Train Model")
interface = create_gradio_interface(
    pipeline, X_train, cat_cols,
    df=df, X_test=X_test, y_test=y_test, y_pred=y_pred,
    y_proba=y_proba, roc_auc=roc_auc, target=target,
    X_train=X_train, y_train=y_train,
    ppd_agent=ppd_agent  # Will be created after first training
)

print("\n✅ Gradio interface is ready!")
print("📱 The web interface will open in your browser.")
print("💡 You can use the example cases below the form for quick testing.")
print("📊 Example cases include:")
print("   - High risk case (multiple symptoms)")
print("   - Low risk case (no symptoms)")
print("   - Moderate risk case (some symptoms)")
print("   - Very high risk case (all symptoms)")
print("   - Low-moderate risk (sleep issues only)")
print("\n🤖 The Gradio interface is now using the PPD Agent (Standalone Python usage - Example 1)")
print("="*50)

# Launch the interface
interface.launch(
    share=False, 
    server_name="127.0.0.1", 
    server_port=7860,
    css="""
        .tab-nav button,
        .tab-nav label,
        div[data-testid="tab"] button,
        div[data-testid="tab"] label {
            font-weight: bold !important;
        }
    """
)
