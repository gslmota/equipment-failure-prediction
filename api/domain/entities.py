from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Optional, Any, TypedDict
from datetime import datetime

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class EquipmentData(BaseModel):
    Cycle: int
    Temperature: float
    Pressure: float
    VibrationX: float
    VibrationY: float
    VibrationZ: float
    Frequency: float
    Preset_1: int
    Preset_2: int

class PredictionResult(BaseModel):
    failure_probability: float
    risk_level: RiskLevel
    dominant_feature: Optional[str] = None

class HealthStatus(BaseModel):
    status: str                 
    version: Optional[str] = None  
    timestamp: datetime

class TrainingParameters(BaseModel):
    data_path: str
    sheet_name: str = 'O&G Equipment Data'
    class_weights: List[float] = [1, 3]
    depth: int = 6
    l2_leaf_reg: float = 1
    learning_rate: float = 0.05
    window_size: int = 15

class TrainingResult(BaseModel):
    status: str
    metrics: Dict[str, float]
    message: str


class ModelArtifacts(TypedDict):
    model: Any          
    scaler: Any         
    preset_risk: Any    
    features: List[str]
