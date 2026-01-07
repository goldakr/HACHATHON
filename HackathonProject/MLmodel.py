# 📌 ML libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
import numpy as np


def create_XGBoost_pipeline(cat_cols, **xgb_params):
    """
    Create a preprocessing and modeling pipeline for XGBoost classifier.
    
    IMPORTANT: OneHotEncoder is ONLY applied to categorical columns.
    Numeric columns (like Age) are passed through unchanged, which is correct
    for tree-based models like XGBoost that can handle numeric features directly.
    
    Args:
        cat_cols: List of categorical column names (object dtype)
        **xgb_params: Optional XGBoost hyperparameters to override defaults
        
    Returns:
        sklearn Pipeline object with:
        - Categorical columns: OneHotEncoded
        - Numeric columns: Passed through unchanged (remainder='passthrough')
    """
    # 📌 OneHotEncode ONLY categorical columns
    # OneHotEncoder is designed for categorical data, NOT numeric data
    # Use sparse_output=False for newer sklearn, sparse=False for older versions
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Fallback for older sklearn versions
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    # 📌 Create transformers list
    transformers = []
    
    # Add categorical transformer ONLY if there are categorical columns
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))
    
    # 📌 Handle numeric columns with passthrough
    # remainder='passthrough' means all columns NOT in transformers are passed through unchanged
    # This is correct for tree-based models (XGBoost, Random Forest) which handle numeric features directly
    # Numeric columns like Age do NOT need OneHotEncoding - they are used as continuous values
    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'  # Numeric columns (int64, float64) pass through unchanged
    )
    
    # 📌 Default XGBoost parameters
    default_params = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    }
    
    # Update with any provided parameters
    default_params.update(xgb_params)
    
    # 📌 XGBoost classifier
    xgb_model = XGBClassifier(**default_params)
    
    # 📌 Pipeline
    pipeline = Pipeline(steps=[("preprocess", preprocess),
                               ("model", xgb_model)])
    
    return pipeline


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    """
    Train the pipeline and evaluate on test set.
    
    Args:
        pipeline: sklearn Pipeline object
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (y_proba, y_pred, roc_auc)
    """
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # 💡 Predict probabilities and classes
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    # 📊 Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", roc_auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return y_proba, y_pred, roc_auc


def optimize_XGBoost_hyperparameters(X_train, y_train, cat_cols, n_iter=50, cv=5, 
                             scoring='roc_auc', random_state=42, n_jobs=-1):
    """
    Optimize XGBoost hyperparameters using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cat_cols: List of categorical column names
        n_iter: Number of parameter settings sampled (default: 50)
        cv: Number of cross-validation folds (default: 5)
        scoring: Scoring metric (default: 'roc_auc')
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 uses all processors)
        
    Returns:
        tuple: (best_pipeline, best_params, cv_results)
    """
    print("\n" + "="*60)
    print("🔍 Starting RandomizedSearchCV Hyperparameter Optimization for XGBoost")
    print("="*60)
    print(f"📊 Parameters: n_iter={n_iter}, cv={cv}, scoring={scoring}")
    print(f"⏱ This may take a few minutes...\n")
    
    # 📌 Create base pipeline
    base_pipeline = create_XGBoost_pipeline(cat_cols)
    
    # 📌 Define hyperparameter search space
    # Using distributions for continuous parameters and lists for discrete ones
    param_distributions = {
        'model__n_estimators': randint(100, 500),  # 100 to 500 trees
        'model__learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
        'model__max_depth': randint(3, 10),  # 3 to 9
        'model__subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
        'model__colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
        'model__min_child_weight': randint(1, 7),  # 1 to 6
        'model__gamma': uniform(0, 0.5),  # 0 to 0.5
        'model__reg_alpha': uniform(0, 1),  # L1 regularization: 0 to 1
        'model__reg_lambda': uniform(0, 2),  # L2 regularization: 0 to 2
    }
    
    # 📌 Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )
    
    # 🔍 Perform search
    random_search.fit(X_train, y_train)
    
    # 📊 Display results
    print("\n" + "="*60)
    print("✅ XGBoost Optimization Complete!")
    print("="*60)
    print(f"\n🏆 Best {scoring} Score (CV): {random_search.best_score_:.4f}")
    print(f"\n📋 Best Hyperparameters:")
    for param, value in random_search.best_params_.items():
        # Format parameter names nicely (remove 'model__' prefix)
        param_name = param.replace('model__', '')
        print(f"   {param_name}: {value}")
    
    print(f"\n📈 Best Parameters Details:")
    best_params = random_search.best_params_.copy()
    for key, value in best_params.items():
        if isinstance(value, (int, np.integer)):
            print(f"   {key}: {int(value)}")
        elif isinstance(value, (float, np.floating)):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.cv_results_


def create_rf_pipeline(cat_cols, **rf_params):
    """
    Create a preprocessing and modeling pipeline for Random Forest classifier.
    
    IMPORTANT: OneHotEncoder is ONLY applied to categorical columns.
    Numeric columns (like Age) are passed through unchanged, which is correct
    for tree-based models like Random Forest that can handle numeric features directly.
    
    Args:
        cat_cols: List of categorical column names (object dtype)
        **rf_params: Optional Random Forest hyperparameters to override defaults
        
    Returns:
        sklearn Pipeline object with:
        - Categorical columns: OneHotEncoded
        - Numeric columns: Passed through unchanged (remainder='passthrough')
    """
    # 📌 OneHotEncode ONLY categorical columns
    # OneHotEncoder is designed for categorical data, NOT numeric data
    # Use sparse_output=False for newer sklearn, sparse=False for older versions
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Fallback for older sklearn versions
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    # 📌 Create transformers list
    transformers = []
    
    # Add categorical transformer ONLY if there are categorical columns
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))
    
    # 📌 Handle numeric columns with passthrough
    # remainder='passthrough' means all columns NOT in transformers are passed through unchanged
    # This is correct for tree-based models (XGBoost, Random Forest) which handle numeric features directly
    # Numeric columns like Age do NOT need OneHotEncoding - they are used as continuous values
    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'  # Numeric columns (int64, float64) pass through unchanged
    )
    
    # 📌 Default Random Forest parameters
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Update with any provided parameters
    default_params.update(rf_params)
    
    # 📌 Random Forest classifier
    rf_model = RandomForestClassifier(**default_params)
    
    # 📌 Pipeline
    pipeline = Pipeline(steps=[("preprocess", preprocess),
                               ("model", rf_model)])
    
    return pipeline


def optimize_rf_hyperparameters(X_train, y_train, cat_cols, n_iter=50, cv=5, 
                                scoring='roc_auc', random_state=42, n_jobs=-1):
    """
    Optimize Random Forest hyperparameters using RandomizedSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cat_cols: List of categorical column names
        n_iter: Number of parameter settings sampled (default: 50)
        cv: Number of cross-validation folds (default: 5)
        scoring: Scoring metric (default: 'roc_auc')
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 uses all processors)
        
    Returns:
        tuple: (best_pipeline, best_params, cv_results)
    """
    print("\n" + "="*60)
    print("🔍 Starting RandomizedSearchCV Hyperparameter Optimization for Random Forest")
    print("="*60)
    print(f"📊 Parameters: n_iter={n_iter}, cv={cv}, scoring={scoring}")
    print(f"⏱ This may take a few minutes...\n")
    
    # 📌 Create base pipeline
    base_pipeline = create_rf_pipeline(cat_cols)
    
    # 📌 Define hyperparameter search space for Random Forest
    # Using distributions for continuous parameters and lists for discrete ones
    param_distributions = {
        'model__n_estimators': randint(50, 300),  # 50 to 300 trees
        'model__max_depth': [None] + list(range(5, 25)),  # None or 5 to 24
        'model__min_samples_split': randint(2, 20),  # 2 to 19
        'model__min_samples_leaf': randint(1, 10),  # 1 to 9
        'model__max_features': ['sqrt', 'log2', None],  # Feature selection methods
        'model__bootstrap': [True, False],  # Whether to use bootstrap samples
        'model__class_weight': [None, 'balanced', 'balanced_subsample'],  # Class weight handling
    }
    
    # 📌 Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )
    
    # 🔍 Perform search
    random_search.fit(X_train, y_train)
    
    # 📊 Display results
    print("\n" + "="*60)
    print("✅ Random Forest Optimization Complete!")
    print("="*60)
    print(f"\n🏆 Best {scoring} Score (CV): {random_search.best_score_:.4f}")
    print(f"\n📋 Best Hyperparameters:")
    for param, value in random_search.best_params_.items():
        # Format parameter names nicely (remove 'model__' prefix)
        param_name = param.replace('model__', '')
        print(f"   {param_name}: {value}")
    
    print(f"\n📈 Best Parameters Details:")
    best_params = random_search.best_params_.copy()
    for key, value in best_params.items():
        if isinstance(value, (int, np.integer)):
            print(f"   {key}: {int(value)}")
        elif isinstance(value, (float, np.floating)):
            print(f"   {key}: {value:.4f}")
        elif value is None:
            print(f"   {key}: None")
        else:
            print(f"   {key}: {value}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.cv_results_