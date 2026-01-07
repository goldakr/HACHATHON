"""
Gradio interface for Postpartum Depression Prediction Agent Tool.

This module has been refactored to use separate modules for:
- gradio_predictions: Prediction functions with type hints
- gradio_visualizations: Visualization functions with type hints
- exceptions: Custom exception classes for better error handling
"""
from typing import Optional, Any, List
import gradio as gr
import numpy as np
import pandas as pd
import os
import base64
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Optional import for chatbot (requires python-dotenv package)
try:
    from dotenv import load_dotenv  # type: ignore
    DOTENV_AVAILABLE = True
except ImportError:
    load_dotenv = None
    DOTENV_AVAILABLE = False

# Import from refactored modules
from gradio_predictions import predict_depression
from gradio_visualizations import (
    create_enhanced_shap_plot,
    create_shap_summary_plot_class1
)
from gradio_helpers import (
    clean_feature_name,
    format_feature_importance_line,
    generate_shap_explanation_markdown,
    get_save_path_for_algorithm
)
# Exception classes are imported but may be used for future error handling
# from exceptions import (
#     PredictionError,
#     SHAPExplanationError,
#     VisualizationError
# )
from MLmodel import (
    create_XGBoost_pipeline
)


# All prediction and visualization functions are imported from:
# - gradio_predictions: predict_depression, generate_personalized_explanation
# - gradio_visualizations: create_shap_summary_plot, create_enhanced_shap_plot, create_shap_summary_plot_class1
# - gradio_helpers: Helper functions for common operations

def create_gradio_interface(
    pipeline: Pipeline,
    X_train_sample: pd.DataFrame,
    cat_cols: List[str],
    df: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    roc_auc: Optional[float] = None,
    target: Optional[str] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    ppd_agent: Optional[Any] = None
) -> gr.Blocks:
    """
    Create a Gradio interface for postpartum depression prediction.

    Args:
        pipeline: Trained sklearn pipeline
        X_train_sample: Sample of training data for SHAP explainer (DataFrame with feature columns)
        cat_cols: List of categorical column names
        df: Full dataset (for visualizations)
        X_test: Test features (for visualizations)
        y_test: Test labels (for visualizations)
        y_pred: Predicted labels (for visualizations)
        y_proba: Predicted probabilities (for visualizations)
        roc_auc: ROC AUC score (for visualizations)
        target: Target variable name (for visualizations)
        X_train: Full training features (for model training, optional)
        y_train: Full training labels (for model training, optional)
        ppd_agent: PPD Agent instance (optional, if provided will use agent.predict() method)

    Returns:
        Gradio Interface object

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    # Get feature column names in the correct order
    feature_columns = list(X_train_sample.columns)

    # Get feature dtypes to ensure exact match
    feature_dtypes = X_train_sample.dtypes.to_dict()
    
    # Helper function to get unique values from dataframe for a given column
    def get_unique_values(column_name: str, default_choices: List[str] = None) -> List[str]:
        """Extract unique values from dataframe for a given column."""
        if df is not None and column_name in df.columns:
            unique_vals = sorted([str(v) for v in df[column_name].dropna().unique()])
            # Filter out empty strings
            unique_vals = [v for v in unique_vals if v.strip()]
            if unique_vals:
                return unique_vals
        # Fallback to default choices if column not found or no unique values
        return default_choices if default_choices else []
    
    # Helper function to get default value (first value from choices, or fallback)
    def get_default_value(column_name: str, fallback: str = "") -> str:
        """Get default value for a column (first unique value, or fallback)."""
        choices = get_unique_values(column_name)
        if choices:
            return choices[0]
        return fallback

    # Try to load existing trained agent if available (try both XGBoost and Random Forest)
    agent_loaded = False
    loaded_algorithm = None
    if ppd_agent is None:
        import os
        from pathlib import Path
        # Get script directory to resolve relative paths
        script_dir = Path(__file__).parent
        # Try to load XGBoost first, then Random Forest
        agent_paths = [
            script_dir / "output" / "agents" / "ppd_agent_xgboost.pkl",
            script_dir / "output" / "agents" / "ppd_agent_rf.pkl"
        ]
        
        for agent_path in agent_paths:
            if agent_path.exists():
                try:
                    from ppd_agent import PPDAgent
                    # Convert Path object to string for PPDAgent.load()
                    loaded_agent = PPDAgent.load(str(agent_path))
                    ppd_agent = loaded_agent
                    # Update pipeline from loaded agent
                    if hasattr(loaded_agent, 'pipeline') and loaded_agent.pipeline is not None:
                        pipeline = loaded_agent.pipeline
                        agent_loaded = True
                        # Determine algorithm type
                        model_type = type(pipeline.named_steps.get("model", None)).__name__
                        if "XGB" in model_type.upper():
                            loaded_algorithm = "XGBoost"
                        elif "RandomForest" in model_type or "Random" in model_type:
                            loaded_algorithm = "RandomForest"
                        print(f"✅ Loaded existing trained agent ({loaded_algorithm}) from {agent_path}")
                        break
                except Exception as e:
                    print(f"⚠️ Could not load agent from {agent_path}: {e}")
                    continue

    # Use mutable containers to store current pipeline and explainer
    current_pipeline = [pipeline]  # Use list to allow mutation
    current_explainer = [None]
    current_X_train = [X_train_sample]
    current_agent = [ppd_agent]  # Store agent if provided (may be loaded from file)
    
    # If agent was loaded, make sure current_agent is set
    if agent_loaded and ppd_agent is not None:
        current_agent[0] = ppd_agent
    
    # Store current predictions for visualizations (will be updated when model is trained)
    current_y_pred = [y_pred if y_pred is not None else None]
    current_y_proba = [y_proba if y_proba is not None else None]
    current_roc_auc = [roc_auc if roc_auc is not None else None]

    # LangChain agent for chatbot (will be initialized when agent is available)
    langchain_agent_instance = [None]
    langchain_available = False  # Track if LangChain is available
    
    # Check if LangChain is available (check once at startup)
    try:
        import langchain
        langchain_available = True
    except ImportError:
        langchain_available = False
        print("ℹ️ LangChain not installed. Chatbot feature will be disabled.")

    # Create SHAP explainer (only if pipeline is trained or agent is loaded)
    model_trained = False
    try:
        # Check if model is trained by trying to get feature importances
        model = pipeline.named_steps.get("model")
        if model is not None and hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            preprocessor = pipeline.named_steps["preprocess"]
            _ = preprocessor.transform(X_train_sample[:100])  # Use subset for speed
            explainer = shap.TreeExplainer(model)
            current_explainer[0] = explainer
            model_trained = True
        else:
            # Model not trained yet, explainer will be created after training
            current_explainer[0] = None
            model_trained = False
    except Exception as e:
        print(f"Warning: Could not create SHAP explainer: {e}")
        current_explainer[0] = None
        model_trained = False
    
    # If agent was loaded, mark model as trained
    if agent_loaded:
        model_trained = True
    
    # LangChain agent initialization function
    def initialize_langchain_agent():
        """Initialize LangChain agent with PPD tool if agent is available."""
        if langchain_agent_instance[0] is not None:
            return  # Already initialized
        
        if current_agent[0] is None:
            return  # Agent not available yet
        
        # Check if LangChain is available at all
        if not langchain_available:
            return  # LangChain not installed, skip silently
        
        try:
            # Check if python-dotenv is available
            if not DOTENV_AVAILABLE or load_dotenv is None:
                return  # Skip silently if dotenv not available
            
            # Load environment variables from current directory (.env file)
            load_dotenv()
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                openai_api_key = openai_api_key.strip()  # Remove any whitespace/newlines
            if not openai_api_key:
                return  # Skip silently if API key not found
            
            # Initialize LangChain agent (try both old and new API)
            try:
                from langchain.agents import initialize_agent, AgentType
                # Try both old and new OpenAI import locations
                try:
                    from langchain.llms import OpenAI  # type: ignore
                except ImportError:
                    from langchain_openai import OpenAI
                from langchain_tool import create_langchain_tool
                
                # Create tool from PPD agent
                tool = create_langchain_tool(current_agent[0])
                
                # Create OpenAI LLM
                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                
                # Initialize LangChain agent (old API)
                langchain_agent_instance[0] = initialize_agent(
                    tools=[tool],
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False
                )
                print("✅ LangChain agent initialized successfully for chatbot!")
            except (ImportError, AttributeError):
                # Try LangChain 1.x API (create_agent)
                try:
                    from langchain.agents import create_agent
                    from langchain_openai import ChatOpenAI
                    from langchain_tool import create_langchain_tool
                    
                    # Create tool from PPD agent
                    tool = create_langchain_tool(current_agent[0])
                    
                    # Create OpenAI LLM
                    model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-3.5-turbo")
                    
                    # Create agent (LangChain 1.x API)
                    langchain_agent_instance[0] = create_agent(
                        model=model,
                        tools=[tool],
                        system_prompt="You are a helpful assistant that can answer questions about postpartum depression (PPD) risk prediction using the available tool."
                    )
                    print("✅ LangChain agent initialized successfully for chatbot (using LangChain 1.x API)!")
                except ImportError:
                    # Try LangChain 0.3.x API (create_react_agent)
                    try:
                        from langchain.agents import create_react_agent, AgentExecutor
                        from langchain_core.prompts import PromptTemplate
                        from langchain_openai import ChatOpenAI
                        from langchain_tool import create_langchain_tool
                        
                        # Create tool from PPD agent
                        tool = create_langchain_tool(current_agent[0])
                        
                        # Create OpenAI LLM
                        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-3.5-turbo")
                        
                        # Create prompt template
                        prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tool:
                        
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")
                        
                        # Create agent (LangChain 0.3.x API)
                        agent = create_react_agent(llm, [tool], prompt)
                        langchain_agent_instance[0] = AgentExecutor(agent=agent, tools=[tool], verbose=False)
                        print("✅ LangChain agent initialized successfully for chatbot (using LangChain 0.3.x API)!")
                    except ImportError:
                        # LangChain API not compatible - skip silently
                        pass
                    except Exception as e3:
                        # Other errors - skip silently
                        pass
                except Exception as e3:
                    # Other errors - skip silently
                    pass
            except Exception:
                # Other errors - skip silently
                pass
        except Exception:
            # All errors - skip silently
            pass
    
    def chatbot_handler(message, history):
        """Handle chatbot messages using LangChain agent."""
        # Try to initialize agent if not already initialized
        if langchain_agent_instance[0] is None:
            initialize_langchain_agent()
        
        if langchain_agent_instance[0] is None:
            return "❌ Chatbot is not available. Please ensure:\n1. The model has been trained\n2. OPENAI_API_KEY is set in .env file\n3. LangChain and OpenAI packages are installed."
        
        if current_agent[0] is None:
            return "❌ PPD Agent is not available. Please train the model first using the 'Start Train Model' button."
        
        try:
            # Run the LangChain agent (handle both old and new API)
            agent = langchain_agent_instance[0]
            if hasattr(agent, 'run'):
                # Old API (LangChain 0.3.x with AgentExecutor)
                response = agent.run(message)
            elif hasattr(agent, 'invoke'):
                # New API (LangChain 1.x with create_agent)
                result = agent.invoke({"messages": [{"role": "user", "content": message}]})
                # Extract response from messages or structured_response
                if isinstance(result, dict):
                    if "messages" in result and len(result["messages"]) > 0:
                        last_message = result["messages"][-1]
                        response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    elif "output" in result:
                        response = result["output"]
                    elif "structured_response" in result:
                        response = str(result["structured_response"])
                    else:
                        response = str(result)
                else:
                    response = str(result)
            else:
                # Fallback: try to call directly
                response = agent(message)
            return str(response) if response else "I'm sorry, I couldn't generate a response. Please try again."
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Chatbot error: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"❌ Error: {error_msg}\n\nPlease try rephrasing your question or check if the model is properly trained."

    # Training function
    def train_model_wrapper(model_algorithm, use_optimization):
        """Train model based on selected algorithm."""
        if X_train is None or y_train is None:
            return "❌ Training data not available. Please provide X_train and y_train when creating the interface."
        
        try:
            # Split data if needed
            # Always use a consistent split for fair comparison between models
            if X_test is None or y_test is None:
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
                )
            else:
                # Use provided test data, but still split training data for consistency
                # This ensures we're always evaluating on the same test set
                X_train_split = X_train
                X_test_split = X_test
                y_train_split = y_train
                y_test_split = y_test
            
            # Initialize optimization_info and algorithm_name
            optimization_info = ""
            algorithm_name = "XGBoost" if model_algorithm == "Training XGBoost Model" else "Random Forest"
            
            # Use PPD Agent tool for training instead of direct algorithm calls
            if model_algorithm == "Training XGBoost Model":
                # Use agent's train_xgboost method
                if current_agent[0] is None:
                    # Create a new agent instance without training (will train via train_xgboost)
                    from ppd_agent import PPDAgent
                    # Create an untrained pipeline for the agent
                    temp_pipeline = create_XGBoost_pipeline(cat_cols)
                    # Create agent with untrained pipeline (SHAP explainer will be None initially)
                    temp_agent = PPDAgent(temp_pipeline, X_train_split, cat_cols, list(X_train_split.columns))
                    current_agent[0] = temp_agent
                
                print("Using PPD Agent tool to train XGBoost model...")
                training_result = current_agent[0].train_xgboost(
                    X_train=X_train_split,
                    y_train=y_train_split,
                    X_test=X_test_split,
                    y_test=y_test_split,
                    use_optimization=use_optimization,
                        n_iter=30,  # Reduced for faster training in UI
                        cv=3,       # Reduced for faster training in UI
                        scoring='roc_auc',
                        random_state=42,
                        n_jobs=-1
                    )
                
                if not training_result["success"]:
                    error_msg = f"❌ Agent training failed: {training_result.get('message', 'Unknown error')}"
                    if df is not None and X_test is not None and y_test is not None:
                        return (error_msg, "", "", "", "", gr.update(interactive=False), gr.update(visible=not agent_loaded), gr.update(visible=agent_loaded))
                    else:
                        return (error_msg, gr.update(interactive=False), gr.update(visible=not agent_loaded), gr.update(visible=agent_loaded))
                
                # Extract results from agent training
                new_pipeline = training_result["pipeline"]
                y_proba_new = training_result["y_proba"]
                y_pred_new = training_result["y_pred"]
                roc_auc_new = training_result["roc_auc"]
                best_params = training_result.get("parameters", {})
                
                # Format optimization info
                if use_optimization and best_params:
                    optimization_info = f"\n\n🔍 Hyperparameter Optimization Applied:\n"
                    for param, value in best_params.items():
                        param_name = param.replace('model__', '')
                        optimization_info += f"   • {param_name}: {value}\n"
                else:
                    optimization_info = "\n\nℹ️ Using default hyperparameters. Enable optimization for better performance."
            elif model_algorithm == "Training Random Forest Model":
                # Use agent's train_random_forest method
                if current_agent[0] is None:
                    # Create a new agent instance without training (will train via train_random_forest)
                    from ppd_agent import PPDAgent
                    from MLmodel import create_rf_pipeline
                    # Create an untrained pipeline for the agent
                    temp_pipeline = create_rf_pipeline(cat_cols)
                    # Create agent with untrained pipeline (SHAP explainer will be None initially)
                    temp_agent = PPDAgent(temp_pipeline, X_train_split, cat_cols, list(X_train_split.columns))
                    current_agent[0] = temp_agent
                
                print("Using PPD Agent tool to train Random Forest model...")
                training_result = current_agent[0].train_random_forest(
                    X_train=X_train_split,
                    y_train=y_train_split,
                    X_test=X_test_split,
                    y_test=y_test_split,
                        random_state=42,
                        n_jobs=-1
                    )
                
                if not training_result["success"]:
                    error_msg = f"❌ Agent training failed: {training_result.get('message', 'Unknown error')}"
                    if df is not None and X_test is not None and y_test is not None:
                        return (error_msg, "", "", "", "", gr.update(interactive=False), gr.update(visible=not agent_loaded), gr.update(visible=agent_loaded))
                    else:
                        return (error_msg, gr.update(interactive=False), gr.update(visible=not agent_loaded), gr.update(visible=agent_loaded))
                
                # Extract results from agent training (all returned by agent)
                new_pipeline = training_result["pipeline"]
                roc_auc_new = training_result["roc_auc"]
                best_params = training_result.get("parameters", {})
                y_proba_new = training_result["y_proba"]
                y_pred_new = training_result["y_pred"]
                
                # Format optimization info (Random Forest training doesn't support optimization in agent yet)
                if use_optimization:
                    optimization_info = "\n\n⚠️ Hyperparameter optimization for Random Forest is not yet available through the agent tool. Using default parameters."
                else:
                    optimization_info = "\n\nℹ️ Using default hyperparameters."
            else:
                return f"❌ Unknown algorithm: {model_algorithm}"
            
            # Note: Training and evaluation is now done by the agent tool above
            # y_proba_new, y_pred_new, roc_auc_new are already set from agent training results
            
            # Prediction statistics logged for debugging (commented out for production)
            # print(f"DEBUG: {algorithm_name} predictions - Unique values: {np.unique(y_pred_new, return_counts=True)}")
            # print(f"DEBUG: {algorithm_name} predictions - Sum: {np.sum(y_pred_new)}, Total: {len(y_pred_new)}")
            
            # Update current pipeline and predictions
            current_pipeline[0] = new_pipeline
            current_X_train[0] = X_train_split
            current_y_pred[0] = y_pred_new
            current_y_proba[0] = y_proba_new
            current_roc_auc[0] = roc_auc_new
            
            # Note: current_agent[0] already has the trained pipeline since training was done via agent
            # We just need to update the SHAP explainer and initialize LangChain agent
            if current_agent[0] is not None:
                # Agent's pipeline is already updated (training was done via agent.train_xgboost/train_random_forest)
                # Reinitialize SHAP explainer in agent (already done by agent training, but ensure it's set)
                # Initialize LangChain agent after training
                initialize_langchain_agent()
            
            # Reinitialize SHAP explainer
            try:
                model = new_pipeline.named_steps["model"]
                # For Random Forest, TreeExplainer works well
                # For better accuracy, we could pass background data, but TreeExplainer works without it
                current_explainer[0] = shap.TreeExplainer(model)
                explainer_status = "✅ SHAP explainer initialized (supports Top 5 Feature Contributions and Personalized Explanation)"
            except Exception as e:
                current_explainer[0] = None
                explainer_status = f"⚠️ SHAP explainer failed: {str(e)}"
            
            # Save the agent (already updated by training methods train_xgboost/train_random_forest)
            if current_agent[0] is not None:
                try:
                    # Agent's pipeline and explainer are already updated by train_xgboost/train_random_forest
                    # Just save the existing agent
                    from pathlib import Path
                    script_dir = Path(__file__).parent
                    agent_dir = script_dir / "output" / "agents"
                    agent_dir.mkdir(parents=True, exist_ok=True)
                    # Determine algorithm type from pipeline
                    model_type = type(new_pipeline.named_steps.get("model", None)).__name__
                    if "XGB" in model_type.upper():
                        algo_suffix = "xgboost"
                    elif "RandomForest" in model_type or "Random" in model_type:
                        algo_suffix = "rf"
                    else:
                        algo_suffix = "unknown"
                    
                    agent_path = agent_dir / f"ppd_agent_{algo_suffix}.pkl"
                    current_agent[0].save(str(agent_path))
                    print(f"✅ Agent saved to {agent_path} (Algorithm: {algorithm_name})")
                    explainer_status += "\n✅ PPD Agent saved with new model"
                except Exception as e:
                    print(f"Warning: Could not save agent: {e}")
            
            # Update visualizations if test data is available
            confusion_html = ""
            roc_html = ""
            prediction_html = ""
            
            # Initialize visualization HTML variables
            confusion_html = ""
            roc_html = ""
            prediction_html = ""
            shap_html = ""
            correlation_html = ""
            
            # Update visualizations using the test data and new predictions
            # Always use y_test_split and y_pred_new to ensure we're showing the current model's performance
            if y_test_split is not None and y_pred_new is not None:
                try:
                    from visualization import (
                        plot_confusion_matrix, plot_roc_curve,
                        plot_prediction_distribution
                    )
                    # Use the test data and predictions from the newly trained model
                    # Calculate confusion matrix values
                    from sklearn.metrics import confusion_matrix as cm_func
                    cm_values = cm_func(y_test_split, y_pred_new)
                    # Debug logs (commented out for production):
                    # print(f"DEBUG: Creating confusion matrix - Model: {algorithm_name}")
                    # print(f"DEBUG: y_test_split shape: {len(y_test_split)}, y_pred_new shape: {len(y_pred_new)}")
                    # print(f"DEBUG: {algorithm_name} Confusion Matrix values:\n{cm_values}")
                    
                    # Create save path for plots using the same helper function as load_visualizations
                    # This will use the Hackathon directory as base
                    save_path = get_save_path_for_algorithm(pipeline=new_pipeline, algorithm_name=algorithm_name)
                    if save_path:
                        os.makedirs(save_path, exist_ok=True)
                    else:
                        # Fallback if helper fails - use script directory as base
                        from pathlib import Path
                        script_dir = Path(__file__).parent
                        save_path = script_dir / "output" / "plots" / algorithm_name.replace(" ", "_")
                        os.makedirs(save_path, exist_ok=True)
                        save_path = str(save_path)
                    
                    confusion_html = plot_confusion_matrix(
                        y_test_split, y_pred_new, 
                        title=f"Confusion Matrix - {algorithm_name}",
                        return_image=True,
                        save_path=save_path
                    )
                    roc_html = plot_roc_curve(
                        y_test_split, y_proba_new, roc_auc_new,
                        title=f"ROC Curve - {algorithm_name}",
                        return_image=True,
                        save_path=save_path
                    )
                    prediction_html = plot_prediction_distribution(
                        y_proba_new, 
                        title=f"Prediction Probability Distribution - {algorithm_name}",
                        return_image=True,
                        save_path=save_path
                    )
                    # Update SHAP summary plot for Class 1 with the newly trained model
                    shap_html = create_shap_summary_plot_class1(
                        new_pipeline, X_test_split, max_display=15, return_image=True,
                        title=f"SHAP Summary Plot - Class 1 (Yes Depression) - {algorithm_name}",
                        save_path=save_path
                    )
                    # Generate correlation heatmap
                    if df is not None and target is not None:
                        from visualization import plot_correlation_heatmap
                        correlation_html = plot_correlation_heatmap(
                            df, target, return_image=True, save_path=save_path
                        )
                    
                    # Also regenerate target distribution and feature distributions plots after training
                    # These are needed so they're available in load_visualizations
                    if save_path and df is not None and target in df.columns:
                        from visualization import plot_target_distribution, plot_feature_distributions
                        try:
                            # Regenerate target distribution plot
                            plot_target_distribution(
                                df[target], 
                                title="PPD Target Distribution",
                                return_image=True, 
                                save_path=save_path
                            )
                            print(f"✅ Regenerated target_distribution.png in {save_path}")
                        except Exception as e:
                            print(f"⚠️ Could not regenerate target distribution plot: {e}")
                        
                        try:
                            # Regenerate feature distributions plot (use original cat_cols)
                            if cat_cols and len(cat_cols) > 0:
                                plot_feature_distributions(
                                    df, 
                                    cat_cols[:12], 
                                    target, 
                                    n_cols=3, 
                                    return_image=True, 
                                    save_path=save_path
                                )
                                print(f"✅ Regenerated feature_distributions.png in {save_path}")
                        except Exception as e:
                            print(f"⚠️ Could not regenerate feature distributions plot: {e}")
                except Exception as e:
                    import traceback
                    print(f"Warning: Could not update visualizations: {e}")
                    print(traceback.format_exc())
                    confusion_html = f"<p>Error updating confusion matrix: {str(e)}</p>"
                    roc_html = f"<p>Error updating ROC curve: {str(e)}</p>"
                    prediction_html = f"<p>Error updating prediction distribution: {str(e)}</p>"
                    shap_html = f"<p>Error updating SHAP summary plot: {str(e)}</p>"
                    correlation_html = f"<p>Error updating correlation heatmap: {str(e)}</p>"
            
            status_message = f"""✅ {algorithm_name} model trained successfully!

📊 Model Performance:
   • ROC AUC Score: {roc_auc_new:.4f}
   • Test Set Size: {len(X_test_split)} samples
   • Training Set Size: {len(X_train_split)} samples

{explainer_status}{optimization_info}

The model is now ready for predictions!"""
            
            # Return based on whether test data is available for visualizations
            # Also enable the "Assess Risk" button, hide "Start Train Model" button, and show "Retrain Model" button after successful training
            if df is not None and X_test is not None and y_test is not None:
                return (status_message, confusion_html, roc_html, prediction_html, shap_html, correlation_html, gr.update(interactive=True), gr.update(visible=False), gr.update(visible=True))
            else:
                return (status_message, gr.update(interactive=True), gr.update(visible=False), gr.update(visible=True))
            
        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            if df is not None and X_test is not None and y_test is not None:
                return (error_msg, "", "", "", "", "", gr.update(interactive=False), gr.update(visible=not agent_loaded), gr.update(visible=agent_loaded))
            else:
                return (error_msg, gr.update(interactive=False), gr.update(visible=not agent_loaded), gr.update(visible=agent_loaded))

    # Create a wrapper function that includes pipeline and explainer
    def predict_wrapper(
        patient_name,
        epds_total_score,
        age,
        marital_status,
        ses,
        population,
        employment_category,
        first_birth,
        gdm,
        tsh,
        nvp,
        gh,
        mode_of_birth,
        depression_history,
        anxiety_history,
        depression_or_anxiety_during_pregnancy,
        use_of_psychiatric_medications,
        sleep_quality,
        fatigue,
        partner_support,
        family_or_social_support,
        domestic_violence,
    ):
        # Check if model has been trained
        if current_agent[0] is None and (current_pipeline[0] is None or 
            not hasattr(current_pipeline[0].named_steps.get("model", None), 'feature_importances_')):
            return (
                "⚠️ Model not trained yet. Please click 'Start Train Model' to train the model first.",
                "Model not trained yet.",
                "Model not trained yet.",
                "Model not trained yet.",
                "Model not trained yet."
            )
        
        # Use agent if available (Standalone Python usage - Example 1)
        if current_agent[0] is not None:
            try:
                # Use agent's predict method (Standalone Python usage pattern)
                # Use predict_from_dict with the new feature structure
                agent_input_dict = {
                    "Age": str(int(age)) if age is not None and str(age).strip() != "" else "",
                    "Marital status": marital_status,
                    "SES": ses,
                    "Population": population,
                    "Employment Category": employment_category,
                    "First birth": first_birth,
                    "GDM": gdm,
                    "TSH": tsh,
                    "NVP": nvp,
                    "GH": gh,
                    "Mode of birth": mode_of_birth,
                    "Depression History": depression_history,
                    "Anxiety History": anxiety_history,
                    "Depression or anxiety during pregnancy": depression_or_anxiety_during_pregnancy,
                    "Use of psychiatric medications": use_of_psychiatric_medications,
                    "Sleep quality": sleep_quality,
                    "Fatigue": fatigue,
                    "Partner support": partner_support,
                    "Family or social support": family_or_social_support,
                    "Domestic violence": domestic_violence,
                }
                result = current_agent[0].predict_from_dict(agent_input_dict)
                
                # Format output to match Gradio interface expectations
                # Include Name and EPDS Total Score in the display
                name_display = f"Patient: {patient_name}\n" if patient_name and patient_name.strip() else ""
                epds_display = f"EPDS Total Score: {int(epds_total_score)}\n" if epds_total_score is not None and epds_total_score != 0 else ""
                epds_interpretation = ""
                if epds_total_score is not None and epds_total_score != 0:
                    if epds_total_score >= 13:
                        epds_interpretation = " (Likely PPD - Score >= 13)"
                    elif epds_total_score >= 11:
                        epds_interpretation = " (Mild depression or dejection - Score 11-12)"
                    else:
                        epds_interpretation = " (Low PPD risk - Score <= 10)"
                
                risk_score = f"{name_display}{epds_display}{epds_interpretation}\nPPD Risk Score: {result['risk_percentage']:.2f}%"
                
                # Format feature importance with detailed SHAP explanation
                feat_imp_lines = []
                shap_explanation_lines = []
                
                for i, feature in enumerate(result['feature_importance'][:5], 1):
                    shap_val = feature['shap_value']
                    
                    # Format for feature importance display using helper function
                    feat_imp_lines.append(
                        format_feature_importance_line(feature['feature'], shap_val, i)
                    )
                
                feat_imp_str = "\n\n".join(feat_imp_lines) if feat_imp_lines else "Feature importance not available"
                
                # Create comprehensive SHAP explanation using helper function
                shap_explanation = generate_shap_explanation_markdown(
                    result['feature_importance'],
                    risk_percentage=result.get('risk_percentage')
                )
                
                # Use agent's explanation
                personalized_explanation = result['explanation']
                
                # Create enhanced SHAP plot with waterfall-style visualization
                if result['feature_importance']:
                    top_features = result['feature_importance'][:5]
                    feat_names = [clean_feature_name(f['feature']) for f in top_features]
                    shap_vals = np.array([f['shap_value'] for f in top_features])
                    
                    # Create both bar plot and waterfall-style plot
                    # Use 0.5 as base value (average prediction) for waterfall visualization
                    # Determine save path based on current model
                    save_path_plot = get_save_path_for_algorithm(pipeline=current_agent[0].pipeline)
                    os.makedirs(save_path_plot, exist_ok=True)
                    
                    plot_html = create_enhanced_shap_plot(
                        list(zip(feat_names, shap_vals)),
                        shap_vals,
                        feat_names,
                        base_value=0.5,
                        save_path=save_path_plot
                    )
                else:
                    plot_html = ""
                
                return risk_score, feat_imp_str, personalized_explanation, shap_explanation, plot_html
                
            except Exception as e:
                import traceback
                error_msg = f"Agent prediction error: {str(e)}\n{traceback.format_exc()}"
                return (
                    "Error during prediction",
                    error_msg,
                    "Unable to generate explanation.",
                    "SHAP explanation unavailable due to error.",
                    ""
                )
        
        # Fallback to original predict_depression if agent not available
        if current_explainer[0] is None:
            name_display = f"Patient: {patient_name}\n" if patient_name and patient_name.strip() else ""
            epds_display = f"EPDS Total Score: {int(epds_total_score)}\n" if epds_total_score is not None and epds_total_score != 0 else ""
            return (
                f"{name_display}{epds_display}SHAP explainer not available",
                "Please train the model first",
                "Unable to generate explanation.",
                "SHAP explanation unavailable. Please train the model first.",
                ""
            )
        
        # Get results from original predict_depression (returns 5 values)
        # Pass all form field values as kwargs - map to actual column names (with spaces)
        prediction_kwargs = {
            "Age": str(int(age)) if age is not None and str(age).strip() != "" else "",
            "Marital status": marital_status,
            "SES": ses,
            "Population": population,
            "Employment Category": employment_category,
            "First birth": first_birth,
            "GDM": gdm,
            "TSH": tsh,
            "NVP": nvp,
            "GH": gh,
            "Mode of birth": mode_of_birth,
            "Depression History": depression_history,
            "Anxiety History": anxiety_history,
            "Depression or anxiety during pregnancy": depression_or_anxiety_during_pregnancy,
            "Use of psychiatric medications": use_of_psychiatric_medications,
            "Sleep quality": sleep_quality,
            "Fatigue": fatigue,
            "Partner support": partner_support,
            "Family or social support": family_or_social_support,
            "Domestic violence": domestic_violence,
        }
        risk_score_base, feat_imp_str, personalized_explanation, shap_explanation, plot_html = predict_depression(
            current_pipeline[0],
            current_explainer[0],
            feature_columns,
            feature_dtypes,
            **prediction_kwargs
        )
        
        # Add Name and EPDS Total Score to the risk_score display
        name_display = f"Patient: {patient_name}\n" if patient_name and patient_name.strip() else ""
        epds_display = f"EPDS Total Score: {int(epds_total_score)}\n" if epds_total_score is not None and epds_total_score != 0 else ""
        epds_interpretation = ""
        if epds_total_score is not None and epds_total_score != 0:
            if epds_total_score >= 13:
                epds_interpretation = " (Likely PPD - Score >= 13)\n"
            elif epds_total_score >= 11:
                epds_interpretation = " (Mild depression or dejection - Score 11-12)\n"
            else:
                epds_interpretation = " (Low PPD risk - Score <= 10)\n"
        
        risk_score = f"{name_display}{epds_display}{epds_interpretation}{risk_score_base}"
        
        return risk_score, feat_imp_str, personalized_explanation, shap_explanation, plot_html

    # Create Gradio interface
    with gr.Blocks(title="Postpartum Depression Prediction Agent Tool") as interface:
        gr.Markdown("# 🏥 Postpartum Depression Risk Assessment Model Training Agent")
        
        # Model selection and training section
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Select Model Training Algorithm**")
                model_algorithm = gr.Dropdown(
                    label=None,
                    choices=["Training XGBoost Model", "Training Random Forest Model"],
                    value="Training XGBoost Model",
                    info="Choose the algorithm to train the model"
                )
                use_optimization = gr.Checkbox(
                    label="Use RandomizedSearchCV optimization (slower but better performance)",
                    value=False,
                    info="Enable hyperparameter optimization for the selected model. This will take longer but may improve model performance."
                )
                # Set button visibility based on whether agent is loaded
                train_btn_visible = not agent_loaded
                retrain_btn_visible = agent_loaded
                train_btn = gr.Button("🚀 Start Train Model", variant="secondary", visible=train_btn_visible)
                retrain_btn = gr.Button("🔄 Retrain Model", variant="secondary", visible=retrain_btn_visible)
                training_status = gr.Textbox(
                    label="Training Status",
                    interactive=False,
                    lines=1,
                    value=f"✅ {loaded_algorithm} model loaded from saved agent. Ready for predictions!" if agent_loaded and loaded_algorithm else "Ready to train model. Select an algorithm and click 'Start Train Model'. ⚠️ Note: You must train the model before making predictions."
                )

        # Add visualization tabs
        if df is not None and X_test is not None and y_test is not None:
            with gr.Tabs() as viz_tabs:
                with gr.Tab("1. Target Distribution"):
                    target_dist_plot = gr.HTML(label="Target Distribution")
                
                with gr.Tab("2. Feature Distributions"):
                    feature_dist_plot = gr.HTML(label="Feature Distributions by Target")
                
                with gr.Tab("3. Confusion Matrix"):
                    confusion_matrix_plot = gr.HTML(label="Confusion Matrix")
                
                with gr.Tab("4. ROC Curve"):
                    roc_curve_plot = gr.HTML(label="ROC Curve")
                
                with gr.Tab("5. Prediction Distribution"):
                    prediction_dist_plot = gr.HTML(label="Prediction Probability Distribution")
                
                with gr.Tab("6. Correlation Heatmap"):
                    correlation_heatmap_plot = gr.HTML(label="Correlation Heatmap")
                
                with gr.Tab("7. SHAP Summary (Class 1 - Yes Depression)"):
                    shap_summary_class1_plot = gr.HTML(label="SHAP Summary Plot - Class 1 (Yes Depression)")
            
            # Load visualizations when interface loads
            def load_plot_if_exists(plot_filename, save_path):
                """Load a plot from file if it exists, return None if not found."""
                if save_path and plot_filename:
                    plot_path = os.path.join(save_path, plot_filename)
                    if os.path.exists(plot_path):
                        try:
                            import base64
                            with open(plot_path, 'rb') as f:
                                img_data = base64.b64encode(f.read()).decode('utf-8')
                            return f'<img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto;">'
                        except Exception as e:
                            print(f"⚠️ Could not load plot {plot_filename}: {e}")
                return None
            
            def load_visualizations():
                from visualization import (
                    plot_target_distribution, plot_feature_distributions,
                    plot_confusion_matrix, plot_roc_curve,
                    plot_prediction_distribution, plot_correlation_heatmap
                )
                
                # Use current pipeline (may be from loaded agent)
                viz_pipeline = current_pipeline[0] if current_pipeline[0] is not None else pipeline
                
                # Determine algorithm name from pipeline using helper function
                try:
                    save_path_viz = get_save_path_for_algorithm(pipeline=viz_pipeline)
                    if save_path_viz:
                        os.makedirs(save_path_viz, exist_ok=True)
                except:
                    save_path_viz = None
                
                # If y_pred or y_proba are None but agent is loaded, generate predictions
                viz_y_pred = current_y_pred[0] if current_y_pred[0] is not None else y_pred
                viz_y_proba = current_y_proba[0] if current_y_proba[0] is not None else y_proba
                viz_roc_auc = current_roc_auc[0] if current_roc_auc[0] is not None else roc_auc
                
                # If predictions are still None but we have a trained pipeline, generate them
                if (viz_y_pred is None or viz_y_proba is None) and viz_pipeline is not None:
                    try:
                        model = viz_pipeline.named_steps.get("model")
                        if model is not None and hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                            # Generate predictions from loaded agent
                            if X_test is not None and y_test is not None:
                                viz_y_proba = viz_pipeline.predict_proba(X_test)[:, 1]
                                viz_y_pred = viz_pipeline.predict(X_test)
                                from sklearn.metrics import roc_auc_score
                                viz_roc_auc = roc_auc_score(y_test, viz_y_proba)
                                # Update current predictions
                                current_y_pred[0] = viz_y_pred
                                current_y_proba[0] = viz_y_proba
                                current_roc_auc[0] = viz_roc_auc
                                print("✅ Generated predictions from loaded agent for visualizations")
                    except Exception as e:
                        print(f"⚠️ Could not generate predictions for visualizations: {e}")
                
                plots = {}
                
                # Try to load existing plots first, generate if not found
                plot_files = {
                    'target': 'target_distribution.png',
                    'features': 'feature_distributions.png',
                    'confusion': 'confusion_matrix.png',
                    'roc': 'roc_curve.png',
                    'prediction': 'prediction_distribution.png',
                    'correlation': 'correlation_heatmap.png',
                    'shap_class1': 'shap_summary_class1.png'
                }
                
                # Ensure df has target column and correct structure
                # Reconstruct df if needed to ensure it has all columns including target
                viz_df = df.copy() if (df is not None and target in df.columns) else None
                if viz_df is None:
                    # Try to reconstruct df from X_train and y_train (combine both train and test for full dataset)
                    if X_train is not None and y_train is not None:
                        train_df = X_train.copy()
                        train_df[target] = y_train.values
                        if X_test is not None and y_test is not None:
                            test_df = X_test.copy()
                            test_df[target] = y_test.values
                            viz_df = pd.concat([train_df, test_df], ignore_index=True)
                        else:
                            viz_df = train_df
                    elif X_test is not None and y_test is not None:
                        viz_df = X_test.copy()
                        viz_df[target] = y_test.values
                
                # Ensure cat_cols matches actual categorical columns in viz_df
                # Always use the original cat_cols list (which excludes ID, Name, target)
                # but filter it to only include columns that exist in viz_df
                if viz_df is not None:
                    # Use original cat_cols, but filter to columns that exist in viz_df
                    # Also exclude target column and ID/Name if they somehow made it through
                    viz_df_cols = set(viz_df.columns)
                    excluded_from_plot = {'ID', 'Name', target}
                    viz_cat_cols = [
                        c for c in (cat_cols if cat_cols else []) 
                        if c in viz_df_cols and c not in excluded_from_plot
                    ]
                    # If no valid cat_cols found, try to detect from dtype (fallback)
                    if len(viz_cat_cols) == 0:
                        actual_cat_cols = [
                            c for c in viz_df.columns 
                            if c not in excluded_from_plot and viz_df[c].dtype == "object"
                        ]
                        viz_cat_cols = actual_cat_cols if len(actual_cat_cols) > 0 else []
                else:
                    viz_cat_cols = cat_cols if cat_cols else []
                
                # Load target distribution plot
                plots['target'] = load_plot_if_exists(plot_files['target'], save_path_viz)
                if plots['target'] is None:
                    try:
                        if viz_df is not None and target in viz_df.columns:
                            plots['target'] = plot_target_distribution(viz_df[target], title="PPD Composite Target Distribution", return_image=True, save_path=save_path_viz)
                            print(f"Generated {plot_files['target']}")
                        else:
                            plots['target'] = f"<p>Error: Target column '{target}' not found in data. Viz_df: {viz_df is not None}, Target in df: {target in df.columns if df is not None else False}</p>"
                    except Exception as e:
                        plots['target'] = f"<p>Error: {str(e)}</p>"
                else:
                    print(f"Loaded existing {plot_files['target']}")
                
                # Load feature distributions plot
                plots['features'] = load_plot_if_exists(plot_files['features'], save_path_viz)
                if plots['features'] is None:
                    try:
                        if viz_df is not None and target in viz_df.columns and len(viz_cat_cols) > 0:
                            # Limit to first 12 features to avoid too many subplots
                            plots['features'] = plot_feature_distributions(viz_df, viz_cat_cols[:12], target, n_cols=3, return_image=True, save_path=save_path_viz)
                            print(f"Generated {plot_files['features']} ({len(viz_cat_cols[:12])} features)")
                        else:
                            plots['features'] = f"<p>Error: Cannot generate feature distributions. Data: {viz_df is not None}, Target: {target in viz_df.columns if viz_df is not None else False}, Cat cols: {len(viz_cat_cols)}</p>"
                    except Exception as e:
                        plots['features'] = f"<p>Error: {str(e)}</p>"
                else:
                    print(f"Loaded existing {plot_files['features']}")
                
                # Load confusion matrix plot
                if viz_y_pred is not None and y_test is not None:
                    plots['confusion'] = load_plot_if_exists(plot_files['confusion'], save_path_viz)
                    if plots['confusion'] is None:
                        try:
                            plots['confusion'] = plot_confusion_matrix(y_test, viz_y_pred, return_image=True, save_path=save_path_viz)
                            print(f"✅ Generated {plot_files['confusion']}")
                        except Exception as e:
                            plots['confusion'] = f"<p>Error: {str(e)}</p>"
                    else:
                        print(f"📂 Loaded existing {plot_files['confusion']}")
                else:
                    plots['confusion'] = "<p>Confusion matrix not available (model not trained or predictions not available)</p>"
                
                # Load ROC curve plot
                if viz_y_proba is not None and y_test is not None and viz_roc_auc is not None:
                    plots['roc'] = load_plot_if_exists(plot_files['roc'], save_path_viz)
                    if plots['roc'] is None:
                        try:
                            plots['roc'] = plot_roc_curve(y_test, viz_y_proba, roc_auc=viz_roc_auc, return_image=True, save_path=save_path_viz)
                            print(f"✅ Generated {plot_files['roc']}")
                        except Exception as e:
                            plots['roc'] = f"<p>Error: {str(e)}</p>"
                    else:
                        print(f"📂 Loaded existing {plot_files['roc']}")
                else:
                    plots['roc'] = "<p>ROC curve not available (model not trained or predictions not available)</p>"
                
                # Load prediction distribution plot
                if viz_y_proba is not None:
                    plots['prediction'] = load_plot_if_exists(plot_files['prediction'], save_path_viz)
                    if plots['prediction'] is None:
                        try:
                            plots['prediction'] = plot_prediction_distribution(viz_y_proba, return_image=True, save_path=save_path_viz)
                            print(f"✅ Generated {plot_files['prediction']}")
                        except Exception as e:
                            plots['prediction'] = f"<p>Error: {str(e)}</p>"
                    else:
                        print(f"📂 Loaded existing {plot_files['prediction']}")
                else:
                    plots['prediction'] = "<p>Prediction distribution not available (model not trained or predictions not available)</p>"
                
                # Load correlation heatmap plot
                plots['correlation'] = load_plot_if_exists(plot_files['correlation'], save_path_viz)
                if plots['correlation'] is None:
                    try:
                        if viz_df is not None and target in viz_df.columns:
                            print(f"Generating correlation heatmap...")
                            print(f"  DataFrame shape: {viz_df.shape}")
                            print(f"  DataFrame columns: {list(viz_df.columns)[:10]}...")
                            print(f"  Target: {target}")
                            plots['correlation'] = plot_correlation_heatmap(viz_df, target, return_image=True, save_path=save_path_viz)
                            if plots['correlation'] is not None:
                                print(f"Generated {plot_files['correlation']} successfully")
                            else:
                                print(f"Warning: plot_correlation_heatmap returned None")
                                plots['correlation'] = "<p>Error: Correlation heatmap generation returned None</p>"
                        else:
                            error_msg = f"Cannot generate correlation heatmap. Data: {viz_df is not None}, Target: {target in viz_df.columns if viz_df is not None else False}"
                            print(f"ERROR: {error_msg}")
                            plots['correlation'] = f"<p>Error: {error_msg}</p>"
                    except Exception as e:
                        import traceback
                        error_trace = traceback.format_exc()
                        print(f"ERROR generating correlation heatmap: {e}")
                        print(error_trace)
                        plots['correlation'] = f"<p>Error: {str(e)}<br><pre>{error_trace[:500]}</pre></p>"
                else:
                    print(f"Loaded existing {plot_files['correlation']}")
                
                # Load SHAP summary plot
                if viz_pipeline is not None and X_test is not None:
                    plots['shap_class1'] = load_plot_if_exists(plot_files['shap_class1'], save_path_viz)
                    if plots['shap_class1'] is None:
                        try:
                            plots['shap_class1'] = create_shap_summary_plot_class1(
                                viz_pipeline, X_test, max_display=15, return_image=True, save_path=save_path_viz
                            )
                            print(f"✅ Generated {plot_files['shap_class1']}")
                        except Exception as e:
                            plots['shap_class1'] = f"<p>Error: {str(e)}</p>"
                    else:
                        print(f"📂 Loaded existing {plot_files['shap_class1']}")
                else:
                    plots['shap_class1'] = "<p>SHAP summary plot not available (model not trained or test data not available)</p>"
                
                return (
                    plots['target'],
                    plots['features'],
                    plots['confusion'],
                    plots['roc'],
                    plots['prediction'],
                    plots['correlation'],
                    plots['shap_class1']
                )
            
            # Load visualizations on interface load
            interface.load(fn=load_visualizations, outputs=[
                target_dist_plot, feature_dist_plot, confusion_matrix_plot,
                roc_curve_plot, prediction_dist_plot, correlation_heatmap_plot,
                    shap_summary_class1_plot
            ])
        
        gr.Markdown("---")
        gr.Markdown(
            "**Enter the following information to assess the risk of postpartum depression.**"
        )

        gr.Markdown("### Patient Information")
        with gr.Row():
            with gr.Column():
                patient_name = gr.Textbox(
                    label="Name",
                    value="",
                    placeholder="Enter patient name",
                    info="Patient name"
                )
                epds_total_score = gr.Number(
                    label="EPDS Total Score",
                    value=None,
                    minimum=0,
                    maximum=30,
                    step=1,
                    info="EPDS Total Score (0-30). Score >= 13: Likely PPD, Score 11-12: Mild depression or dejection, Score <= 10: Low PPD risk. Leave empty if not available."
                )
        
        gr.Markdown("### Demographics")
        with gr.Row():
            with gr.Column():
                age = gr.Number(
                    label="Age",
                    value=None,
                    step=1,
                    info="Enter the actual age (numeric value from Demographics.csv)"
                )
                marital_status = gr.Radio(
                    label="Marital status",
                    choices=get_unique_values("Marital status", ["Married", "Single", "Divorced", "Widowed"]),
                    value=None,
                )
                ses = gr.Radio(
                    label="SES (Socioeconomic Status)",
                    choices=get_unique_values("SES", ["High", "Low", "Medium"]),
                    value=None,
                )
            with gr.Column():
                population = gr.Dropdown(
                    label="Population",
                    choices=get_unique_values("Population", ["Secular", "Peripheral Jewish towns", "Religious", "Other"]),
                    value=None,
                )
                employment_category = gr.Dropdown(
                    label="Employment Category",
                    choices=get_unique_values("Employment Category", ["Employed (Full-Time)", "Self-Employed", "Unemployed", "Part-Time", "Student"]),
                    value=None,
                )
        
        gr.Markdown("### Clinical Data")
        with gr.Row():
            with gr.Column():
                first_birth = gr.Radio(
                    label="First birth",
                    choices=get_unique_values("First birth", ["Yes", "No"]),
                    value=None,
                )
                gdm = gr.Radio(
                    label="GDM (Gestational Diabetes Mellitus)",
                    choices=get_unique_values("GDM", ["Yes", "No"]),
                    value=None,
                )
                tsh = gr.Radio(
                    label="TSH",
                    choices=get_unique_values("TSH", ["Normal", "Abnormal"]),
                    value=None,
                )
            with gr.Column():
                nvp = gr.Radio(
                    label="NVP (Nausea and Vomiting of Pregnancy)",
                    choices=get_unique_values("NVP", ["Yes", "No"]),
                    value=None,
                )
                gh = gr.Radio(
                    label="GH (Gestational Hypertension)",
                    choices=get_unique_values("GH", ["Yes", "No"]),
                    value=None,
                )
                mode_of_birth = gr.Dropdown(
                    label="Mode of birth",
                    choices=get_unique_values("Mode of birth", ["Spontaneous Vaginal", "Cesarean", "Assisted Vaginal", "Other"]),
                    value=None,
                )
        
        gr.Markdown("### Psychiatric Data")
        with gr.Row():
            with gr.Column():
                depression_history = gr.Dropdown(
                    label="Depression History",
                    choices=get_unique_values("Depression History", ["Not documented", "Yes", "No"]),
                    value=None,
                )
                anxiety_history = gr.Dropdown(
                    label="Anxiety History",
                    choices=get_unique_values("Anxiety History", ["Not documented", "Yes", "No"]),
                    value=None,
                )
            with gr.Column():
                depression_or_anxiety_during_pregnancy = gr.Radio(
                    label="Depression or anxiety during pregnancy",
                    choices=get_unique_values("Depression or anxiety during pregnancy", ["Yes", "No"]),
                    value=None,
                )
                use_of_psychiatric_medications = gr.Radio(
                    label="Use of psychiatric medications",
                    choices=get_unique_values("Use of psychiatric medications", ["Yes", "No"]),
                    value=None,
                )
        
        gr.Markdown("### Functional/Psychosocial Data")
        with gr.Row():
            with gr.Column():
                sleep_quality = gr.Dropdown(
                    label="Sleep quality",
                    choices=get_unique_values("Sleep quality", ["Normal", "Insomnia", "RLS"]),
                    value=None,
                )
                fatigue = gr.Radio(
                    label="Fatigue",
                    choices=get_unique_values("Fatigue", ["Yes", "No"]),
                    value=None,
                )
                partner_support = gr.Dropdown(
                    label="Partner support",
                    choices=get_unique_values("Partner support", ["High", "Moderate", "Interrupted", "Low"]),
                    value=None,
                )
            with gr.Column():
                family_or_social_support = gr.Dropdown(
                    label="Family or social support",
                    choices=get_unique_values("Family or social support", ["High", "Moderate", "Low"]),
                    value=None,
                )
                domestic_violence = gr.Dropdown(
                    label="Domestic violence",
                    choices=get_unique_values("Domestic violence", ["No", "Physical", "Sexual", "Emotional"]),
                    value=None,
                )

        # Add examples
        gr.Markdown("### 📋 Example Cases")
        gr.Markdown("Click on any example below to load it and see the prediction:")

        gr.Examples(
            examples=[
                # High risk case - Direct criteria met (EPDS > 12, Total Scores = 14)
                [
                    "יעל חמו", 7, 35, "Married", "High", "Secular", "Self-Employed",
                    "No", "No", "Normal", "Yes", "No", "Spontaneous Vaginal",
                    "Not documented", "Not documented", "Yes", "No",
                    "Normal", "Yes", "Moderate", "Moderate", "No"
                ],
                # Low risk case - Very low EPDS score (Total Scores = 2)
                [
                    "ענת גרוס", 9, 27, "Married", "Low", "Peripheral Jewish towns", "Employed (Full-Time)",
                    "No", "No", "Normal", "No", "No", "Spontaneous Vaginal",
                    "Not documented", "Not documented", "Yes", "No",
                    "RLS", "No", "Interrupted", "High", "Verbal"
                ],
                # Very high risk case - Direct criteria met (EPDS = 27 > 12, self-harm thoughts = 2)
                [
                    "חנה ברק", 5, 27, "Married", "Low", "Peripheral Jewish towns", "Employed (Full-Time)",
                    "No", "No", "Normal", "Yes", "No", "Spontaneous Vaginal",
                    "Not documented", "Not documented", "No", "No",
                    "Normal", "No", "Interrupted", "High", "Economic"
                ],
                # Moderate risk case - Risk factors present (Very Low SES, Haredi, Verbal DV, Anxiety History)
                [
                    "ענת דדון", 14, 24, "Married", "Very Low", "Haredi", "Employed (Part-Time)",
                    "No", "No", "Normal", "Yes", "No", "Spontaneous Vaginal",
                    "Not documented", "Documented", "Yes", "No",
                    "RLS", "Yes", "Moderate", "High", "Verbal"
                ],
                # Risk-based case - Single, Unemployed, Interrupted support, Low family support
                [
                    "עדן אבוטבול", 4, 39, "Never-Married (Single)", "High", "Secular", "Unemployed",
                    "No", "No", "Normal", "No", "No", "Spontaneous Vaginal",
                    "Not documented", "Not documented", "No", "No",
                    "Normal", "No", "Interrupted", "Moderate", "No"
                ],
            ],
            inputs=[
                patient_name,
                epds_total_score,
                age,
                marital_status,
                ses,
                population,
                employment_category,
                first_birth,
                gdm,
                tsh,
                nvp,
                gh,
                mode_of_birth,
                depression_history,
                anxiety_history,
                depression_or_anxiety_during_pregnancy,
                use_of_psychiatric_medications,
                sleep_quality,
                fatigue,
                partner_support,
                family_or_social_support,
                domestic_violence,
            ],
            label="Example Cases",
        )

        predict_btn = gr.Button("🔍 Assess Risk", variant="primary", interactive=model_trained)
        gr.Markdown(
            "ℹ️ **Note:** This button is disabled until the model is trained. Please train the model using the 'Retrain Model' button above.",
            visible=True
        )

        with gr.Row():
            risk_output = gr.Textbox(
                label="Risk Assessment",
                interactive=False,
                lines=5,
                info="Shows patient name, EPDS Total Score, and PPD risk assessment"
            )
            personalized_explanation = gr.Textbox(
                label="Personalized Explanation",
                interactive=False,
                lines=4,
            )

        with gr.Row():
            feature_importance = gr.Textbox(
                label="Top 5 Feature Contributions (SHAP)",
                interactive=False,
                lines=8,
            )
            shap_explanation = gr.Markdown(
                label="Detailed SHAP Explanation",
                value="SHAP explanation will appear here after prediction."
            )
        
        with gr.Row():
            shap_plot = gr.HTML(
                label="SHAP Visualization (Bar Plot & Waterfall)",
                elem_classes=["shap-visualization"]
            )

        # Connect training button
        # If visualizations are available, update them when training
        if df is not None and X_test is not None and y_test is not None:
            train_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status, confusion_matrix_plot, roc_curve_plot, prediction_dist_plot, shap_summary_class1_plot, correlation_heatmap_plot, predict_btn, train_btn, retrain_btn],
            )
            # Retrain button uses the same function
            retrain_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status, confusion_matrix_plot, roc_curve_plot, prediction_dist_plot, shap_summary_class1_plot, correlation_heatmap_plot, predict_btn, train_btn, retrain_btn],
            )
        else:
            # If no test data, only update training status
            train_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status, predict_btn, train_btn, retrain_btn],
            )
            # Retrain button uses the same function
            retrain_btn.click(
                fn=train_model_wrapper,
                inputs=[model_algorithm, use_optimization],
                outputs=[training_status, predict_btn, train_btn, retrain_btn],
            )
        
        # Connect prediction button
        predict_btn.click(
            fn=predict_wrapper,
            inputs=[
                patient_name,
                epds_total_score,
                age,
                marital_status,
                ses,
                population,
                employment_category,
                first_birth,
                gdm,
                tsh,
                nvp,
                gh,
                mode_of_birth,
                depression_history,
                anxiety_history,
                depression_or_anxiety_during_pregnancy,
                use_of_psychiatric_medications,
                sleep_quality,
                fatigue,
                partner_support,
                family_or_social_support,
                domestic_violence,
            ],
            outputs=[risk_output, feature_importance, personalized_explanation, shap_explanation, shap_plot],
        )
        
        # Add EPDS Chatbot Tab
        gr.Markdown("---")
        gr.Markdown("## 📋 Edinburgh Postnatal Depression Scale (EPDS) Assessment")
        gr.Markdown("סוכן דינמי עם LangChain: יודע מתי לשאול שאלות EPDS, אוסף טקסט חופשי, משתמש ב-NLP, ומתחבר למודל XGBoost")
        
        # Import EPDS agent module
        try:
            from epds_agent import EPDSAgent
            EPDS_AVAILABLE = True
        except ImportError as e:
            print(f"⚠️ EPDS agent module not available: {e}")
            EPDS_AVAILABLE = False
        
        if EPDS_AVAILABLE:
            # Create EPDS agent instance (with PPD agent connection if available)
            epds_agent_instance = [EPDSAgent(ppd_agent=current_agent[0] if current_agent[0] is not None else None)]
            
            def update_epds_agent():
                """Update EPDS agent when PPD agent becomes available."""
                if current_agent[0] is not None and epds_agent_instance[0] is not None:
                    epds_agent_instance[0].ppd_agent = current_agent[0]
                    if epds_agent_instance[0].langchain_agent is not None:
                        # Reinitialize with updated PPD agent
                        epds_agent_instance[0]._initialize_langchain()
            
            with gr.Row():
                epds_name = gr.Textbox(
                    label="שם (אופציונלי)",
                    placeholder="הזיני את שמך (אופציונלי)",
                    scale=2
                )
                epds_start_btn = gr.Button("התחל הערכה", variant="primary", scale=1)
            
            epds_chatbot = gr.Chatbot(
                label="EPDS Assessment Chat",
                height=400
            )
            
            epds_status = gr.Markdown("לחצי על 'התחל הערכה' כדי להתחיל את ההערכה.")
            
            # Import EPDS_QUESTIONS for status display
            try:
                from epds_agent import EPDS_QUESTIONS
            except ImportError:
                EPDS_QUESTIONS = []  # Fallback
            
            def epds_start_handler(name):
                """Start EPDS conversation."""
                try:
                    # Update agent connection before starting
                    update_epds_agent()
                    epds_agent = epds_agent_instance[0]
                    # Ensure name is properly handled (None -> empty string, strip whitespace)
                    name_clean = name.strip() if name and isinstance(name, str) else ""
                    response = epds_agent.start_conversation(name_clean)
                    history = [{"role": "assistant", "content": response}]
                    return history, "הזיני תשובה או טקסט חופשי:", "הערכה בתהליך... שאלה 1/10"
                except Exception as e:
                    import traceback
                    print(f"EPDS start error: {traceback.format_exc()}")
                    return [], "", f"שגיאה: {str(e)}"
            
            def epds_chat_handler(message, history):
                """Handle EPDS chat messages using intelligent agent."""
                try:
                    # Update agent connection
                    update_epds_agent()
                    epds_agent = epds_agent_instance[0]
                    
                    if epds_agent.state is None:
                        return history if history else [], "אנא לחצי על 'התחל הערכה' תחילה.", "מוכן להתחלה"
                    
                    # Normalize history
                    if history is None:
                        history = []
                    
                    # Process message using intelligent agent
                    response = epds_agent.process_message(str(message))
                    
                    # Update history from agent's conversation history
                    if epds_agent.state and epds_agent.state.conversation_history:
                        history = epds_agent.state.conversation_history.copy()
                    
                    # Determine status
                    state = epds_agent.state
                    if state:
                        if state.assessment_complete:
                            status_text = "הערכה הושלמה ✅"
                        elif state.needs_free_text:
                            status_text = "ממתין לתשובה חופשית..."
                        else:
                            status_text = f"שאלה {state.current_question_index + 1}/{len(EPDS_QUESTIONS) if EPDS_QUESTIONS else 10}"
                    else:
                        status_text = "מוכן"
                    
                    return history, "", status_text
                except Exception as e:
                    import traceback
                    print(f"EPDS chat error: {traceback.format_exc()}")
                    return history if history else [], "", f"שגיאה: {str(e)}"
            
            epds_msg = gr.Textbox(
                label="תשובה",
                placeholder="הזיני תשובה (0-3) או טקסט חופשי",
                lines=2
            )
            
            epds_send_btn = gr.Button("שלח", variant="primary")
            epds_clear_btn = gr.Button("נקה", variant="secondary")
            
            def epds_clear():
                """Clear EPDS conversation."""
                epds_agent_instance[0].reset()
                return [], "", "לחצי על 'התחל הערכה' כדי להתחיל את ההערכה."
            
            epds_start_btn.click(
                fn=epds_start_handler,
                inputs=[epds_name],
                outputs=[epds_chatbot, epds_msg, epds_status]
            )
            
            epds_send_btn.click(
                fn=epds_chat_handler,
                inputs=[epds_msg, epds_chatbot],
                outputs=[epds_chatbot, epds_msg, epds_status]
            )
            
            epds_msg.submit(
                fn=epds_chat_handler,
                inputs=[epds_msg, epds_chatbot],
                outputs=[epds_chatbot, epds_msg, epds_status]
            )
            
            epds_clear_btn.click(
                fn=epds_clear,
                outputs=[epds_chatbot, epds_msg, epds_status]
            )
        
        gr.Markdown("---")
        gr.Markdown("## 💬 Medical Staff Chatbot")
        gr.Markdown(
            "Ask questions about postpartum depression risk assessment. "
            "The chatbot can help you understand risk factors, interpret results, and answer questions about the model."
        )
        
        # Example questions for chatbot
        chatbot_examples = [
            "What is the PPD risk for a 30-year-old patient who feels sad, is irritable, and has trouble sleeping?",
            "Can you explain what factors contribute most to postpartum depression risk?",
            "What does a high risk score mean?",
            "How does age affect the risk of postpartum depression?",
            "What symptoms are most important for predicting postpartum depression?"
        ]
        
        # Helper function to ensure list-of-lists format for chatbot
        # Gradio expects: List[List[str | None | Tuple]] - list of lists!
        def ensure_dict_format(history):
            """Ensure history is in dictionary format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]"""
            if history is None or not isinstance(history, list):
                return []
            result = []
            for item in history:
                try:
                    # Handle dictionary format with 'role' and 'content' (standard Gradio format)
                    if isinstance(item, dict) and "role" in item and "content" in item:
                        result.append({"role": str(item["role"]), "content": str(item["content"])})
                    # Handle dictionary format with 'text' and 'type' (alternative Gradio format)
                    elif isinstance(item, dict) and "text" in item:
                        # Convert 'text' format to 'content' format, infer role from context
                        text_content = str(item.get("text", ""))
                        # If it's a list of dicts with 'text', extract the text
                        if isinstance(text_content, list) and len(text_content) > 0:
                            if isinstance(text_content[0], dict) and "text" in text_content[0]:
                                text_content = text_content[0]["text"]
                        # Try to infer role - if we have alternating pattern, use it; otherwise default to user
                        role = item.get("role", "user")
                        result.append({"role": role, "content": str(text_content)})
                    # Handle tuple/list format (old format) - convert to dict
                    elif isinstance(item, (tuple, list)) and len(item) == 2:
                        result.append({"role": "user", "content": str(item[0])})
                        result.append({"role": "assistant", "content": str(item[1])})
                    # Handle ChatMessage objects (if available)
                    elif hasattr(item, 'role') and hasattr(item, 'content'):
                        result.append({"role": str(item.role), "content": str(item.content)})
                except (TypeError, ValueError, AttributeError):
                    continue
            return result
        
        # Initialize chatbot - Gradio 3.50+ expects dictionary format with 'role' and 'content'
        chatbot = gr.Chatbot(
            label="Chat with Medical Staff Assistant",
            height=400
        )
        
        with gr.Row():
            chatbot_msg = gr.Textbox(
                label="Ask a question",
                placeholder="Type your question here...",
                lines=2,
                scale=4
            )
            chatbot_submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            chatbot_clear = gr.Button("Clear Chat", variant="secondary")
        
        gr.Examples(
            examples=chatbot_examples,
            inputs=chatbot_msg,
            label="Example Questions"
        )
        
        def chat_handler(message, history):
            """Handle chatbot messages - ensures dictionary format for Gradio compatibility."""
            try:
                # Normalize history to dictionary format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
                normalized_history = ensure_dict_format(history) if history else []
                
                # Handle empty message
                if not message:
                    return normalized_history, ""
                
                # Extract message text - handle both string and dict formats
                message_str = ""
                if isinstance(message, dict):
                    if "text" in message:
                        text_val = message["text"]
                        # Handle nested list format: [{'text': '...', 'type': 'text'}]
                        if isinstance(text_val, list) and len(text_val) > 0:
                            if isinstance(text_val[0], dict) and "text" in text_val[0]:
                                message_str = str(text_val[0]["text"]).strip()
                            else:
                                message_str = str(text_val[0]).strip()
                        else:
                            message_str = str(text_val).strip()
                    elif "content" in message:
                        message_str = str(message["content"]).strip()
                    else:
                        message_str = str(message).strip()
                elif isinstance(message, list) and len(message) > 0:
                    # Handle list format: [{'text': '...', 'type': 'text'}]
                    if isinstance(message[0], dict):
                        if "text" in message[0]:
                            message_str = str(message[0]["text"]).strip()
                        elif "content" in message[0]:
                            message_str = str(message[0]["content"]).strip()
                        else:
                            message_str = str(message[0]).strip()
                    else:
                        message_str = str(message[0]).strip()
                else:
                    message_str = str(message).strip()
                
                if not message_str:
                    return normalized_history, ""
                
                # Get response from LangChain agent
                try:
                    response = chatbot_handler(message_str, normalized_history)
                    response = str(response) if response else "I'm sorry, I couldn't generate a response. Please try again."
                except Exception as e:
                    import traceback
                    print(f"Chatbot handler error: {traceback.format_exc()}")
                    response = f"❌ Error processing your message: {str(e)}"
                
                # Append new message and response as dictionaries (Gradio expects dict format with 'role' and 'content')
                normalized_history.append({"role": "user", "content": message_str})
                normalized_history.append({"role": "assistant", "content": str(response)})
                return normalized_history, ""
            except Exception as e:
                import traceback
                print(f"Chat handler error: {traceback.format_exc()}")
                # Return safe default format - empty list
                return [], ""
        
        # Clear handler with safe format
        def clear_chat():
            """Clear chat and return empty list (dictionary format)."""
            return [], ""  # Empty list is valid format
        
        # Submit on button click
        chatbot_submit_btn.click(
            fn=chat_handler,
            inputs=[chatbot_msg, chatbot],
            outputs=[chatbot, chatbot_msg]
        )
        
        # Also submit on Enter key press
        chatbot_msg.submit(
            fn=chat_handler,
            inputs=[chatbot_msg, chatbot],
            outputs=[chatbot, chatbot_msg]
        )
        
        chatbot_clear.click(fn=clear_chat, outputs=[chatbot, chatbot_msg])
        
        # Initialize LangChain agent if agent is already loaded (non-blocking)
        def init_langchain_after_load():
            """Initialize LangChain agent after interface loads."""
            if agent_loaded and current_agent[0] is not None:
                try:
                    initialize_langchain_agent()
                except Exception as e:
                    print(f"⚠️ Could not initialize LangChain agent on startup: {e}")
        
        # Schedule initialization after interface loads (non-blocking)
        interface.load(fn=init_langchain_after_load, inputs=None, outputs=None)

        gr.Markdown("### ⚠️ Disclaimer")
        gr.Markdown(
            "This tool is for informational purposes only and should not replace "
            "professional medical advice."
        )

    return interface