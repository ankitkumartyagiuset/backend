from sqlalchemy import Column, Integer, String, DateTime
from app.core.database import Base
from datetime import datetime

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    condition = Column(String)
    severity = Column(String)
    confidence = Column(String)
    recommendation = Column(String)
    explanation = Column(String, nullable=True)
    prevention = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
