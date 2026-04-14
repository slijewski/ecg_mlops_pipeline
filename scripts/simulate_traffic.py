import requests
import random
import time
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

url = "http://localhost:8000/predict"

CLASS_MAP = {
    0: "Normal",
    1: "Supraventricular premature",
    2: "Premature ventricular contraction",
    3: "Fusion",
    4: "Unclassifiable"
}

def generate_random_ecg():
    return [random.uniform(0, 1) for _ in range(187)]

def run_simulation(num_requests=50, max_delay=0.5):
    logger.info(f"Starting simulation - Sending {num_requests} requests to MLOps API...")
    
    for i in range(num_requests):
        payload = {"features": generate_random_ecg()}
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                class_id = result['predicted_class']
                class_name = CLASS_MAP.get(class_id, "Unknown")
                logger.info(f"[{i+1}/{num_requests}] Success: Predicted class {class_id} ({class_name}) | Latency: {result['latency_seconds']:.5f}s")
            else:
                logger.error(f"[{i+1}/{num_requests}] HTTP Error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Connection failed. Is the API running? Details: {e}")
            break
            
        time.sleep(random.uniform(0.01, max_delay))

if __name__ == "__main__":
    run_simulation(num_requests=100)
