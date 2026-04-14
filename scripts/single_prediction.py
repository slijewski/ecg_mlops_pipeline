import requests
import random
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_MAP = {
    0: "Normal",
    1: "Supraventricular premature",
    2: "Premature ventricular contraction",
    3: "Fusion",
    4: "Unclassifiable"
}

def test_single_prediction():
    url = "http://localhost:8000/predict"
    
    payload = {
        "features": [random.uniform(0, 1) for _ in range(187)]
    }
    
    logger.info(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            class_id = result['predicted_class']
            class_name = CLASS_MAP.get(class_id, "Unknown")
            
            logger.info("SUCCESS!")
            logger.info(f"Prediction: Class {class_id} ({class_name})")
            logger.info(f"Latency: {result['latency_seconds']:.5f}s")
        else:
            logger.error(f"FAILED! Status code: {response.status_code}")
            logger.error(f"Server response: {response.text}")
            
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        logger.info("Make sure the Docker containers or local API are running.")

if __name__ == "__main__":
    test_single_prediction()
