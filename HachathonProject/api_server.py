"""
FastAPI REST API Server for PPD Prediction Agent

This provides a REST API interface for the PPD prediction agent,
making it accessible via HTTP requests.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import os
from ppd_agent import PPDAgent

# Global agent instance (will be initialized on startup)
agent: Optional[PPDAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI (replaces deprecated on_event).
    Handles startup and shutdown events.
    """
    # Startup
    global agent
    if agent is None:
        # Try to load from saved file
        agent_file = "output/agents/ppd_agent.pkl"
        if os.path.exists(agent_file):
            try:
                print(f"Loading agent from {agent_file}...")
                agent = PPDAgent.load(agent_file)
                print("Agent loaded successfully!")
            except Exception as e:
                print(f"WARNING: Could not load agent from {agent_file}: {e}")
                print("WARNING: Please initialize the agent using initialize_agent() or train a new model.")
        else:
            print("WARNING: Agent not initialized and no saved agent file found.")
            print("WARNING: Please initialize the agent using initialize_agent() or train a new model.")
            print("WARNING: The API will start but prediction endpoints will return errors.")
    
    yield
    
    # Shutdown (if needed)
    # Cleanup code can go here


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="PPD Prediction API",
    description="API for Postpartum Depression Risk Prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for single prediction with new feature structure."""
    # Demographics
    Age: Optional[str] = Field(default="", description="Actual age as numeric value (e.g., '30', '35', '27')")
    Marital_status: Optional[str] = Field(default="", description="Marital status (e.g., 'Married', 'Single')")
    SES: Optional[str] = Field(default="", description="Socioeconomic status (e.g., 'High', 'Low', 'Medium')")
    Population: Optional[str] = Field(default="", description="Population group")
    Employment_Category: Optional[str] = Field(default="", description="Employment category")
    
    # Clinical data
    First_birth: Optional[str] = Field(default="", description="First birth: Yes/No")
    GDM: Optional[str] = Field(default="", description="Gestational Diabetes Mellitus: Yes/No")
    TSH: Optional[str] = Field(default="", description="TSH level: Normal/Abnormal")
    NVP: Optional[str] = Field(default="", description="Nausea and Vomiting of Pregnancy: Yes/No")
    GH: Optional[str] = Field(default="", description="Gestational Hypertension: Yes/No")
    Mode_of_birth: Optional[str] = Field(default="", description="Mode of birth (e.g., 'Spontaneous Vaginal', 'Cesarean')")
    
    # Psychiatric data
    Depression_History: Optional[str] = Field(default="", description="Depression history")
    Anxiety_History: Optional[str] = Field(default="", description="Anxiety history")
    Depression_or_anxiety_during_pregnancy: Optional[str] = Field(default="", description="Depression or anxiety during pregnancy: Yes/No")
    Use_of_psychiatric_medications: Optional[str] = Field(default="", description="Use of psychiatric medications: Yes/No")
    
    # Functional/Psychosocial data
    Sleep_quality: Optional[str] = Field(default="", description="Sleep quality (e.g., 'Normal', 'Insomnia', 'RLS')")
    Fatigue: Optional[str] = Field(default="", description="Fatigue: Yes/No")
    Partner_support: Optional[str] = Field(default="", description="Partner support level (e.g., 'High', 'Moderate', 'Interrupted')")
    Family_or_social_support: Optional[str] = Field(default="", description="Family or social support level (e.g., 'High', 'Moderate', 'Low')")
    Domestic_violence: Optional[str] = Field(default="", description="Domestic violence: No/Physical/Sexual/Emotional")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    patients: List[PredictionRequest] = Field(..., description="List of patient data")


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    risk_score: float = Field(..., description="PPD risk score (0-1)")
    risk_percentage: float = Field(..., description="PPD risk as percentage")
    risk_level: str = Field(..., description="Risk level: Low/Moderate/High/Very High")
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    feature_importance: List[dict] = Field(..., description="Top 5 feature contributions")
    explanation: str = Field(..., description="Personalized explanation")
    probabilities: dict = Field(..., description="Class probabilities")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    results: List[PredictionResponse] = Field(..., description="List of prediction results")




def initialize_agent(ppd_agent: PPDAgent):
    """
    Initialize the global agent instance.
    
    Args:
        ppd_agent: PPDAgent instance
    """
    global agent
    agent = ppd_agent
    print("PPD Agent initialized for API server")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PPD Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/predict/batch": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/schema": "GET - Get tool schema"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_loaded": agent is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict PPD risk for a single patient.
    
    Args:
        request: Prediction request with patient data
    
    Returns:
        Prediction response with risk score and explanation
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Convert request to dictionary with proper column names
        input_dict = {
            "Age": request.Age or "",
            "Marital status": request.Marital_status or "",
            "SES": request.SES or "",
            "Population": request.Population or "",
            "Employment Category": request.Employment_Category or "",
            "First birth": request.First_birth or "",
            "GDM": request.GDM or "",
            "TSH": request.TSH or "",
            "NVP": request.NVP or "",
            "GH": request.GH or "",
            "Mode of birth": request.Mode_of_birth or "",
            "Depression History": request.Depression_History or "",
            "Anxiety History": request.Anxiety_History or "",
            "Depression or anxiety during pregnancy": request.Depression_or_anxiety_during_pregnancy or "",
            "Use of psychiatric medications": request.Use_of_psychiatric_medications or "",
            "Sleep quality": request.Sleep_quality or "",
            "Fatigue": request.Fatigue or "",
            "Partner support": request.Partner_support or "",
            "Family or social support": request.Family_or_social_support or "",
            "Domestic violence": request.Domestic_violence or "",
        }
        
        result = agent.predict_from_dict(input_dict)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict PPD risk for multiple patients.
    
    Args:
        request: Batch prediction request with list of patients
    
    Returns:
        Batch prediction response with results for all patients
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        input_list = [
            {
                "Age": p.Age or "",
                "Marital status": p.Marital_status or "",
                "SES": p.SES or "",
                "Population": p.Population or "",
                "Employment Category": p.Employment_Category or "",
                "First birth": p.First_birth or "",
                "GDM": p.GDM or "",
                "TSH": p.TSH or "",
                "NVP": p.NVP or "",
                "GH": p.GH or "",
                "Mode of birth": p.Mode_of_birth or "",
                "Depression History": p.Depression_History or "",
                "Anxiety History": p.Anxiety_History or "",
                "Depression or anxiety during pregnancy": p.Depression_or_anxiety_during_pregnancy or "",
                "Use of psychiatric medications": p.Use_of_psychiatric_medications or "",
                "Sleep quality": p.Sleep_quality or "",
                "Fatigue": p.Fatigue or "",
                "Partner support": p.Partner_support or "",
                "Family or social support": p.Family_or_social_support or "",
                "Domestic violence": p.Domestic_violence or "",
            }
            for p in request.patients
        ]
        
        results = agent.batch_predict(input_list)
        return BatchPredictionResponse(
            results=[PredictionResponse(**r) for r in results]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/schema")
async def get_schema():
    """
    Get OpenAI Function Calling schema for this agent.
    
    Returns:
        Tool schema in OpenAI format
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    return agent.get_tool_schema()


if __name__ == "__main__":
    # This will be called when running the server directly
    # The agent will be automatically loaded from output/agents/ppd_agent.pkl if it exists
    # Otherwise, you can initialize it programmatically:
    #   from api_server import initialize_agent
    #   from ppd_agent import PPDAgent
    #   agent = PPDAgent.load("output/agents/ppd_agent.pkl")
    #   initialize_agent(agent)
    print("Starting PPD Prediction API Server...")
    print("The agent will be automatically loaded from output/agents/ppd_agent.pkl if available.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

