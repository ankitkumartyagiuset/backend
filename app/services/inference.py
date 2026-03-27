from app.ml.model_loader import get_model
from app.ml.preprocess import preprocess_image
import torch
import random
from typing import Dict

CONDITIONS = [
    {"name": "Actinic Keratosis", "sev": "Low", "rec": "Treat locally with cryotherapy.", "exp": "Detected rough, scaly patches typical of sun damage.", "prev": "Use broad-spectrum sunscreen daily and wear protective clothing."},
    {"name": "Dermatitis", "sev": "Medium", "rec": "Monitor and prescribe antihistamines.", "exp": "Visible inflammation and redness patterns consistent with eczema.", "prev": "Moisturize regularly and avoid known allergens or harsh soaps."},
    {"name": "Melanoma Suspicion", "sev": "High", "rec": "Refer to oncologist immediately.", "exp": "Irregular borders and asymmetrical coloring detected.", "prev": "Avoid excessive sun exposure and perform monthly self-examinations."},
    {"name": "Basal Cell Carcinoma", "sev": "Medium", "rec": "Monitor and schedule dermatology visit.", "exp": "Pearly papule with telangiectasia observed.", "prev": "Minimize midday sun exposure and rely on protective hats."},
    {"name": "Benign Nevus", "sev": "Low", "rec": "No action needed.", "exp": "Symmetrical mole with uniform color.", "prev": "Simply monitor for any sudden changes in size, shape, or color."}
]

def run_inference(image_bytes: bytes) -> dict:
    model = get_model()
    tensor = preprocess_image(image_bytes)
    input_batch = tensor.unsqueeze(0)
    
    # Run simulation model inference
    with torch.no_grad():
        output = model(input_batch)
    
    # Mock realistic results with Explainable Output
    confidence = round(random.uniform(85.0, 99.9), 1)
    condition = random.choice(CONDITIONS)
    
    return {
        "condition": condition["name"],
        "severity": condition["sev"],
        "confidence": f"{confidence}%",
        "recommendation": condition["rec"],
        "explanation": condition["exp"],
        "prevention": condition["prev"]
    }
