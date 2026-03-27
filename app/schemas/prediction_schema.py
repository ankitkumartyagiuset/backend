from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class PredictionCreate(BaseModel):
    filename: str
    condition: str
    severity: str
    confidence: str
    recommendation: str
    explanation: Optional[str] = None
    prevention: Optional[str] = None

class PredictionResponse(PredictionCreate):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True
        orm_mode = True
