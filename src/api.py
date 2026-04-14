from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from src.model import ECG1DCNN
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
from fastapi.responses import Response
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="ECG Classification API", description="MLOps prediction endpoint with integrated Prometheus monitoring")

REQUEST_COUNT = Counter('api_predict_requests_total', 'Total HTTP Requests to /predict')
PREDICTION_LATENCY = Histogram('api_predict_latency_seconds', 'Latency of /predict requests')
CLASS_PREDICTIONS = Counter('api_predicted_classes', 'Count of predictions by class', ['predicted_class'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ECG1DCNN(num_classes=5)

try:
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.warning("WARNING: Saved model not found at 'models/best_model.pth'. Using model with initial weights (Never do this in production!).")
    model.to(device)
    model.eval()

class ECGPayload(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_ecg(payload: ECGPayload):
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    if len(payload.features) != 187:
        raise HTTPException(status_code=400, detail="Invalid input length. 187 readings are required.")
    
    try:
        input_data = np.array(payload.features, dtype=np.float32)
        input_tensor = torch.tensor(input_data).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_class = int(predicted.item())
            
        CLASS_PREDICTIONS.labels(predicted_class=str(pred_class)).inc()
        
        process_time = time.time() - start_time
        PREDICTION_LATENCY.observe(process_time)
        
        return {"predicted_class": pred_class, "latency_seconds": process_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
