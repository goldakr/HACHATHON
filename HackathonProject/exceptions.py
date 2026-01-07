"""
Custom exception classes for the PPD Prediction Agent Tool.
"""


class PPDException(Exception):
    """Base exception for all PPD-related errors."""
    pass


class ModelTrainingError(PPDException):
    """Raised when model training fails."""
    pass


class PredictionError(PPDException):
    """Raised when prediction fails."""
    pass


class SHAPExplanationError(PPDException):
    """Raised when SHAP explanation generation fails."""
    pass


class VisualizationError(PPDException):
    """Raised when visualization generation fails."""
    pass


class AgentError(PPDException):
    """Raised when agent operations fail."""
    pass


class DataValidationError(PPDException):
    """Raised when data validation fails."""
    pass


class PipelineError(PPDException):
    """Raised when pipeline operations fail."""
    pass

