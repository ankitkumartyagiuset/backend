from sqlalchemy.orm import Session
from app.services.inference import run_inference
from app.db.crud import create_prediction
from app.schemas.prediction_schema import PredictionCreate, PredictionResponse
from datetime import datetime

def process_prediction(image_bytes: bytes, filename: str, db: Session, contribute_data: bool = False):
    # 1. Simulate Model Inference
    result_dict = run_inference(image_bytes)
    
    # 2. Save securely for Future AI up-gradation (Active Learning Pipeline)
    if contribute_data:
        import os
        import uuid
        import json
        
        # Define analytics storage path safely ignoring the public uploads flow
        dataset_dir = os.path.join(os.getcwd(), "analytics_dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        unique_id = str(uuid.uuid4())
        
        # Save image matrix
        img_path = os.path.join(dataset_dir, f"{unique_id}.jpg")
        with open(img_path, "wb") as f:
            f.write(image_bytes)
            
        # Save Analytical JSON context to pair for model retraining
        label_path = os.path.join(dataset_dir, f"{unique_id}.json")
        with open(label_path, "w") as f:
            json.dump({
                "source": "Rural User Upload",
                "condition": result_dict["condition"],
                "severity": result_dict["severity"],
                "confidence": result_dict["confidence"],
                "date": str(datetime.utcnow())
            }, f)

    # 3. Save standard History Metadata Record (Database)
    db_schema = PredictionCreate(
        filename=filename,
        condition=result_dict["condition"],
        severity=result_dict["severity"],
        confidence=result_dict["confidence"],
        recommendation=result_dict["recommendation"],
        explanation=result_dict["explanation"],
        prevention=result_dict["prevention"]
    )
    db_record = create_prediction(db, db_schema)
    
    # 4. Return API Response
    return PredictionResponse(
        id=db_record.id,
        filename=db_schema.filename,
        condition=db_schema.condition,
        severity=db_schema.severity,
        confidence=db_schema.confidence,
        recommendation=db_schema.recommendation,
        explanation=db_schema.explanation,
        prevention=db_schema.prevention
    )
