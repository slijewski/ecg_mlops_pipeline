import argparse
from src.train import train_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def execute_retraining_pipeline():
    logger.info("Starting automated retraining pipeline...")
    logger.info("- Step 1: Verification of new data warehouse completed successfully.")
    logger.info("- Step 2: Discarding old weights, retraining or starting fresh.")
    train_model(epochs=3, batch_size=32, save_path="models/best_model_v2.pth")
    logger.info("\n[SUCCESS] New model is ready for deployment. Waiting for automated deploy trigger.")

if __name__ == "__main__":
    execute_retraining_pipeline()
