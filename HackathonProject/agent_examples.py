"""
Example Usage Scripts for PPD Agent Tool

This file demonstrates various ways to use the PPD agent:
1. Standalone Python usage
2. API usage
3. LangChain integration
4. OpenAI Function Calling
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppd_agent import PPDAgent, create_agent_from_training
from langchain_tool import create_langchain_tool
import pandas as pd
from sklearn.model_selection import train_test_split
from MLmodel import create_XGBoost_pipeline, train_and_evaluate


def setup_agent():
    """Setup and return a trained PPD agent."""
    print("Setting up PPD Agent...")
    
    # Load data from multiple CSV files
    from pathlib import Path
    data_dir = Path("data")
    
    # Load Demographics.csv (full)
    demographics = pd.read_csv(data_dir / "Demographics.csv")
    
    # Load EPDS_answers.csv (only Name, Total Scores, מחשבות פגיעה עצמית columns)
    epds_columns = ["ID", "Name", "Total Scores", "מחשבות פגיעה עצמית"]
    epds = pd.read_csv(data_dir / "EPDS_answers.csv", usecols=epds_columns)
    
    # Load Clinical_data.csv (full)
    clinical = pd.read_csv(data_dir / "Clinical_data.csv")
    
    # Load Psychiatric_data.csv (full)
    psychiatric = pd.read_csv(data_dir / "Psychiatric_data.csv")
    
    # Load Functional_Psychosocial_data.csv (full)
    functional = pd.read_csv(data_dir / "Functional_Psychosocial_data.csv")
    
    # Merge all dataframes on ID (foreign key)
    # Perform inner joins on ID - keeps only records with matching IDs across all tables
    df = demographics.copy()
    df = df.merge(epds, on="ID", how="inner", suffixes=("", "_epds"), validate="one_to_one")
    df = df.merge(clinical, on="ID", how="inner", validate="one_to_one")
    df = df.merge(psychiatric, on="ID", how="inner", validate="one_to_one")
    df = df.merge(functional, on="ID", how="inner", validate="one_to_one")
    
    # Handle duplicate Name column
    if "Name_epds" in df.columns:
        df.drop(columns=["Name_epds"], inplace=True)
    
    df = df.dropna()
    
    # Create target based on EPDS Total Scores and self-harm thoughts
    target = "PPD_Composite"
    
    # Convert Total Scores to numeric if it's not already
    df['Total Scores'] = pd.to_numeric(df['Total Scores'], errors='coerce')
    df['מחשבות פגיעה עצמית'] = pd.to_numeric(df['מחשבות פגיעה עצמית'], errors='coerce')
    
    # Create composite target: PPD = 1 if Total Scores >= 13 (Likely PPD) OR self-harm thoughts > 0
    # EPDS scoring: >= 13 indicates Likely PPD, 11-12 indicates Mild depression or dejection, <= 10 indicates Low PPD risk
    epds_threshold = 13
    df[target] = ((df['Total Scores'] >= epds_threshold) | 
                  (df['מחשבות פגיעה עצמית'] > 0)).astype(int)
    
    df = df.dropna()
    
    # Ensure Age is numeric (convert if needed)
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    # Drop ID, Name, and EPDS columns used for target creation (not features)
    # ID is used for data merging only, Name is only for display purposes
    X = df.drop(columns=[target, 'ID', 'Name', 'Total Scores', 'מחשבות פגיעה עצמית'], errors='ignore')
    y = df[target]
    
    # Validate that ID and Name are not in features
    if 'ID' in X.columns or 'Name' in X.columns:
        raise ValueError("ERROR: ID or Name columns found in features! They should not be used for model training.")
    
    # Identify categorical and numeric features AFTER creating X (to ensure we only use features in X)
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    pipeline = create_XGBoost_pipeline(cat_cols)
    y_proba, y_pred, roc_auc = train_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )
    
    # Create agent
    agent = create_agent_from_training(pipeline, X_train, cat_cols, list(X_train.columns))
    
    return agent


def example_1_standalone_usage():
    """Example 1: Standalone Python usage."""
    print("\n" + "="*60)
    print("Example 1: Standalone Python Usage")
    print("="*60)
    
    agent = setup_agent()
    
    # Single prediction using new feature structure
    patient_data = {
        "Age": "30",
        "Marital status": "Married",
        "SES": "High",
        "Population": "Secular",
        "Employment Category": "Employed (Full-Time)",
        "First birth": "No",
        "GDM": "No",
        "TSH": "Normal",
        "NVP": "Yes",
        "GH": "No",
        "Mode of birth": "Spontaneous Vaginal",
        "Depression History": "Not documented",
        "Anxiety History": "Not documented",
        "Depression or anxiety during pregnancy": "Yes",
        "Use of psychiatric medications": "No",
        "Sleep quality": "Insomnia",
        "Fatigue": "Yes",
        "Partner support": "High",
        "Family or social support": "Moderate",
        "Domestic violence": "No"
    }
    result = agent.predict_from_dict(patient_data)
    
    print("\nPrediction Result:")
    print(f"Risk Score: {result['risk_percentage']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Explanation: {result['explanation']}")
    print(f"\nTop Features:")
    for feature in result['feature_importance'][:3]:
        print(f"  - {feature['feature']}: {feature['impact']} risk")


def example_2_dict_usage():
    """Example 2: Using dictionary input."""
    print("\n" + "="*60)
    print("Example 2: Dictionary Input Usage")
    print("="*60)
    
    agent = setup_agent()
    
    # Predict from dictionary with new feature structure
    patient_data = {
        "Age": "27",
        "Marital status": "Married",
        "SES": "Low",
        "Population": "Peripheral Jewish towns",
        "Employment Category": "Employed (Full-Time)",
        "First birth": "No",
        "GDM": "No",
        "TSH": "Normal",
        "NVP": "No",
        "GH": "No",
        "Mode of birth": "Spontaneous Vaginal",
        "Depression History": "Not documented",
        "Anxiety History": "Not documented",
        "Depression or anxiety during pregnancy": "No",
        "Use of psychiatric medications": "No",
        "Sleep quality": "Normal",
        "Fatigue": "No",
        "Partner support": "High",
        "Family or social support": "High",
        "Domestic violence": "No"
    }
    
    result = agent.predict_from_dict(patient_data)
    print(f"\nRisk Score: {result['risk_percentage']}%")
    print(f"Risk Level: {result['risk_level']}")


def example_3_batch_prediction():
    """Example 3: Batch predictions."""
    print("\n" + "="*60)
    print("Example 3: Batch Predictions")
    print("="*60)
    
    agent = setup_agent()
    
    # Multiple patients with new feature structure
    patients = [
        {
            "Age": "30",
            "Marital status": "Married",
            "SES": "High",
            "Population": "Secular",
            "Employment Category": "Self-Employed",
            "First birth": "No",
            "GDM": "No",
            "TSH": "Normal",
            "NVP": "Yes",
            "GH": "No",
            "Mode of birth": "Spontaneous Vaginal",
            "Depression History": "Not documented",
            "Anxiety History": "Not documented",
            "Depression or anxiety during pregnancy": "Yes",
            "Use of psychiatric medications": "No",
            "Sleep quality": "Insomnia",
            "Fatigue": "Yes",
            "Partner support": "Interrupted",
            "Family or social support": "Moderate",
            "Domestic violence": "No"
        },
        {
            "Age": "27",
            "Marital status": "Married",
            "SES": "Low",
            "Population": "Peripheral Jewish towns",
            "Employment Category": "Employed (Full-Time)",
            "First birth": "No",
            "GDM": "No",
            "TSH": "Normal",
            "NVP": "No",
            "GH": "No",
            "Mode of birth": "Spontaneous Vaginal",
            "Depression History": "Not documented",
            "Anxiety History": "Not documented",
            "Depression or anxiety during pregnancy": "No",
            "Use of psychiatric medications": "No",
            "Sleep quality": "Normal",
            "Fatigue": "No",
            "Partner support": "High",
            "Family or social support": "High",
            "Domestic violence": "No"
        }
    ]
    
    results = agent.batch_predict(patients)
    
    print("\nBatch Prediction Results:")
    for i, result in enumerate(results, 1):
        print(f"\nPatient {i}:")
        print(f"  Risk: {result['risk_percentage']}% ({result['risk_level']})")


def example_4_api_usage():
    """Example 4: API usage (requires API server to be running)."""
    print("\n" + "="*60)
    print("Example 4: API Usage")
    print("="*60)
    print("\nTo use the API:")
    print("1. Start the API server: python api_server.py")
    print("2. Make HTTP requests:")
    print("""
    import requests

    response = requests.post('http://localhost:8000/predict', json={
        "age": "30-35",
        "feeling_sad": "Yes",
        "irritable": "Yes",
        "trouble_sleeping": "Yes",
        "concentration": "Yes",
        "appetite": "No",
        "feeling_anxious": "Yes",
        "guilt": "Yes",
        "bonding": "Sometimes",
        "suicide_attempt": "No"
    })
    
    result = response.json()
    print(f"Risk: {result['risk_percentage']}%")
    """)

# שהוא יכול לענות על תשאולים של מכולה על המידע שנלמד דרך אלגוריתם ML
def example_5_langchain_usage():
    """Example 5: LangChain integration."""
    print("\n" + "="*60)
    print("Example 5: LangChain Integration")
    print("="*60)
    print("\nTo use with LangChain:")
    print("""
    from langchain.agents import initialize_agent, AgentType
    from langchain.llms import OpenAI
    from langchain_tool import create_langchain_tool
    
    # Setup agent
    agent = setup_agent()
    tool = create_langchain_tool(agent)
    
    # Create LangChain agent
    llm = OpenAI(temperature=0)
    langchain_agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Use the agent
    result = langchain_agent.run(
        "What is the PPD risk for a 30-year-old patient "
        "who feels sad, is irritable, and has trouble sleeping?"
    )
    """)


def example_6_openai_function_calling():
    """Example 6: OpenAI Function Calling schema."""
    print("\n" + "="*60)
    print("Example 6: OpenAI Function Calling")
    print("="*60)
    
    agent = setup_agent()
    schema = agent.get_tool_schema()
    
    print("\nTool Schema (for OpenAI Function Calling):")
    import json
    print(json.dumps(schema, indent=2))
    
    print("\n\nTo use with OpenAI:")
    print("""
    import openai
    
    # Get schema
    schema = agent.get_tool_schema()
    
    # Use with OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Assess PPD risk for..."}],
        functions=[schema],
        function_call={"name": "predict_ppd_risk"}
    )
    """)


def example_7_save_load():
    """Example 7: Save and load agent."""
    print("\n" + "="*60)
    print("Example 7: Save and Load Agent")
    print("="*60)
    
    agent = setup_agent()
    
    # Save agent
    import os
    os.makedirs("output/agents", exist_ok=True)
    agent_path = "output/agents/ppd_agent.pkl"
    agent.save(agent_path)
    print(f"✅ Agent saved to {agent_path}")
    
    # Load agent
    loaded_agent = PPDAgent.load(agent_path)
    
    # Use loaded agent with new feature structure
    patient_data = {
        "Age": "30",
        "Marital status": "Married",
        "SES": "High",
        "Population": "Secular",
        "Employment Category": "Self-Employed",
        "First birth": "No",
        "GDM": "No",
        "TSH": "Normal",
        "NVP": "Yes",
        "GH": "No",
        "Mode of birth": "Spontaneous Vaginal",
        "Depression History": "Not documented",
        "Anxiety History": "Not documented",
        "Depression or anxiety during pregnancy": "Yes",
        "Use of psychiatric medications": "No",
        "Sleep quality": "Insomnia",
        "Fatigue": "Yes",
        "Partner support": "Interrupted",
        "Family or social support": "Moderate",
        "Domestic violence": "No"
    }
    result = loaded_agent.predict_from_dict(patient_data)
    
    print(f"\nLoaded agent prediction: {result['risk_percentage']}%")


def display_menu():
    """Display the example selection menu."""
    print("\n" + "="*60)
    print("PPD Agent Tool - Example Usage")
    print("="*60)
    print("\nAvailable Examples:")
    print("  1. Standalone Python Usage")
    print("  2. Dictionary Input Usage")
    print("  3. Batch Predictions")
    print("  4. API Usage (instructions)")
    print("  5. LangChain Integration (instructions)")
    print("  6. OpenAI Function Calling")
    print("  7. Save and Load Agent")
    print("  8. Run All Examples")
    print("  0. Exit")
    print("="*60)


def get_user_choice():
    """Get user's choice from the menu."""
    while True:
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                return int(choice)
            else:
                print("Invalid choice. Please enter a number between 0 and 8.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return 0
        except Exception as e:
            print(f"Error: {e}. Please try again.")


if __name__ == "__main__":
    examples = {
        1: ("Standalone Python Usage", example_1_standalone_usage),
        2: ("Dictionary Input Usage", example_2_dict_usage),
        3: ("Batch Predictions", example_3_batch_prediction),
        4: ("API Usage", example_4_api_usage),
        5: ("LangChain Integration", example_5_langchain_usage),
        6: ("OpenAI Function Calling", example_6_openai_function_calling),
        7: ("Save and Load Agent", example_7_save_load),
    }
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice == 0:
            print("\nGoodbye!")
            break
        elif choice == 8:
            print("\n" + "="*60)
            print("Running All Examples")
            print("="*60)
            for num, (name, func) in examples.items():
                print(f"\n>>> Running Example {num}: {name}")
                try:
                    func()
                except Exception as e:
                    print(f"Error running example {num}: {e}")
            print("\n" + "="*60)
            print("All examples completed!")
            print("="*60)
        elif choice in examples:
            name, func = examples[choice]
            print(f"\n>>> Running Example {choice}: {name}")
            try:
                func()
                print("\n" + "="*60)
                print(f"Example {choice} completed!")
                print("="*60)
            except Exception as e:
                print(f"\nError running example: {e}")
        else:
            print("Invalid choice. Please try again.")
        
        # Ask if user wants to continue
        if choice != 0:
            continue_choice = input("\nWould you like to run another example? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("\nGoodbye!")
                break

