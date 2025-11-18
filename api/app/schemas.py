from pydantic import BaseModel
from typing import Dict, Optional

class FlowSchema(BaseModel):
    flow_id: Optional[str] = None
    features: Dict[str, float]

class FeedbackSchema(BaseModel):
    flow_id: str
    true_label: str
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    flow_id: Optional[str]
    prediction: str
    confidence: float
    latency_ms: float
