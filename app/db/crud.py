from sqlalchemy.orm import Session
from app.models.prediction_model import PredictionRecord
from app.schemas.prediction_schema import PredictionCreate

def create_prediction(db: Session, prediction: PredictionCreate):
    # Using .dict() for v1 and .model_dump() fallback for v2 compatibility
    data = prediction.dict() if hasattr(prediction, "dict") else prediction.model_dump()
    db_prediction = PredictionRecord(**data)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction
