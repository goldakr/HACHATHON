"""
LangChain Tool Definition for PPD Prediction Agent

This module provides a LangChain tool wrapper for the PPD agent,
making it easy to integrate with LangChain agents and chains.
"""

from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from ppd_agent import PPDAgent


class PPDPredictionInput(BaseModel):
    """Input schema for PPD prediction tool with new feature structure.
    
    Note: Field names should match the exact column names from the merged dataset.
    Use Field aliases if column names contain spaces or special characters.
    """
    # Demographics (matching exact column names)
    Age: Optional[str] = Field(default="", description="Actual age as numeric value (e.g., '30', '35', '27', etc.)")
    Marital_status: Optional[str] = Field(default="", alias="Marital status", description="Marital status (e.g., 'Married', 'Single', etc.)")
    SES: Optional[str] = Field(default="", description="Socioeconomic status (e.g., 'High', 'Low', 'Medium')")
    Population: Optional[str] = Field(default="", description="Population group")
    Employment_Category: Optional[str] = Field(default="", alias="Employment Category", description="Employment category")
    
    # Clinical data
    First_birth: Optional[str] = Field(default="", alias="First birth", description="First birth: Yes/No")
    GDM: Optional[str] = Field(default="", description="Gestational Diabetes Mellitus: Yes/No")
    TSH: Optional[str] = Field(default="", description="TSH level: Normal/Abnormal")
    NVP: Optional[str] = Field(default="", description="Nausea and Vomiting of Pregnancy: Yes/No")
    GH: Optional[str] = Field(default="", description="Gestational Hypertension: Yes/No")
    Mode_of_birth: Optional[str] = Field(default="", alias="Mode of birth", description="Mode of birth (e.g., 'Spontaneous Vaginal', 'Cesarean', etc.)")
    
    # Psychiatric data
    Depression_History: Optional[str] = Field(default="", alias="Depression History", description="Depression history")
    Anxiety_History: Optional[str] = Field(default="", alias="Anxiety History", description="Anxiety history")
    Depression_or_anxiety_during_pregnancy: Optional[str] = Field(default="", alias="Depression or anxiety during pregnancy", description="Depression or anxiety during pregnancy: Yes/No")
    Use_of_psychiatric_medications: Optional[str] = Field(default="", alias="Use of psychiatric medications", description="Use of psychiatric medications: Yes/No")
    
    # Functional/Psychosocial data
    Sleep_quality: Optional[str] = Field(default="", alias="Sleep quality", description="Sleep quality (e.g., 'Normal', 'Insomnia', 'RLS')")
    Fatigue: Optional[str] = Field(default="", description="Fatigue: Yes/No")
    Partner_support: Optional[str] = Field(default="", alias="Partner support", description="Partner support level (e.g., 'High', 'Moderate', 'Interrupted')")
    Family_or_social_support: Optional[str] = Field(default="", alias="Family or social support", description="Family or social support level (e.g., 'High', 'Moderate', 'Low')")
    Domestic_violence: Optional[str] = Field(default="", alias="Domestic violence", description="Domestic violence: No/Physical/Sexual/Emotional")
    
    class Config:
        populate_by_name = True  # Allow both field names and aliases


class PPDPredictionTool(BaseTool):
    """
    LangChain tool for PPD risk prediction.
    
    This tool can be used with LangChain agents to predict
    postpartum depression risk based on patient symptoms.
    """
    
    name: str = "predict_ppd_risk"
    description: str = (
        "Predicts postpartum depression (PPD) risk based on patient symptoms and demographics. "
        "Returns risk score (0-100%), risk level (Low/Moderate/High/Very High), "
        "feature importance, and personalized explanation. "
        "Use this tool when you need to assess PPD risk for a patient."
    )
    args_schema: Type[BaseModel] = PPDPredictionInput
    
    agent: Optional[PPDAgent] = None
    
    def __init__(self, ppd_agent: PPDAgent, **kwargs):
        """
        Initialize the tool with a PPD agent.
        
        Args:
            ppd_agent: PPDAgent instance
            **kwargs: Additional arguments for BaseTool
        """
        super().__init__(**kwargs)
        self.agent = ppd_agent
    
    def _run(self, **kwargs) -> str:
        """
        Execute the tool.
        
        Args:
            **kwargs: Feature names and values matching agent's feature columns
        
        Returns:
            Formatted string with prediction results
        """
        if self.agent is None:
            return "Error: PPD Agent not initialized"
        
        try:
            # Map field names to actual column names (handling aliases)
            field_to_column = {
                "Marital_status": "Marital status",
                "Employment_Category": "Employment Category",
                "First_birth": "First birth",
                "Mode_of_birth": "Mode of birth",
                "Depression_History": "Depression History",
                "Anxiety_History": "Anxiety History",
                "Depression_or_anxiety_during_pregnancy": "Depression or anxiety during pregnancy",
                "Use_of_psychiatric_medications": "Use of psychiatric medications",
                "Sleep_quality": "Sleep quality",
                "Partner_support": "Partner support",
                "Family_or_social_support": "Family or social support",
                "Domestic_violence": "Domestic violence",
            }
            
            # Convert kwargs to dict with proper column names
            input_dict = {}
            for key, value in kwargs.items():
                # Use mapped column name if available, otherwise use key as-is
                column_name = field_to_column.get(key, key)
                input_dict[column_name] = value
            
            # Use predict_from_dict with the converted dictionary
            result = self.agent.predict_from_dict(input_dict)
            
            # Format result as a readable string
            output = f"""
PPD Risk Assessment:
- Risk Score: {result['risk_percentage']}%
- Risk Level: {result['risk_level']}
- Prediction: {'High Risk' if result['prediction'] == 1 else 'Low Risk'}

Explanation:
{result['explanation']}

Top Contributing Factors:
"""
            for i, feature in enumerate(result['feature_importance'][:5], 1):
                feature_name = feature['feature'].split('_')[-1] if '_' in feature['feature'] else feature['feature']
                output += f"{i}. {feature_name}: {feature['impact']} risk (contribution: {abs(feature['shap_value']):.4f})\n"
            
            return output.strip()
            
        except Exception as e:
            return f"Error during prediction: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Async version of _run."""
        return self._run(**kwargs)


def create_langchain_tool(ppd_agent: PPDAgent) -> PPDPredictionTool:
    """
    Create a LangChain tool from a PPD agent.
    
    Args:
        ppd_agent: PPDAgent instance
    
    Returns:
        PPDPredictionTool instance
    """
    return PPDPredictionTool(ppd_agent=ppd_agent)


# Example usage with LangChain agent:
"""
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

# Create tool
tool = create_langchain_tool(ppd_agent)

# Initialize agent with the tool
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
result = agent.run(
    "What is the PPD risk for a 30-year-old patient who feels sad, "
    "is irritable, has trouble sleeping, and reports feeling anxious?"
)
"""

