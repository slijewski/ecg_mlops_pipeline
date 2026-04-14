import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import kagglehub
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ECGDataset(Dataset):
    def __init__(self, csv_file: str, download_kaggle: bool = True):
        try:
            if download_kaggle:
                logger.info("Downloading real data from Kaggle in the background (shayanfazeli/heartbeat)...")
                path = kagglehub.dataset_download("shayanfazeli/heartbeat")
                dataset_path = os.path.join(path, "mitbih_train.csv")
                try:
                    self.data = pd.read_csv(dataset_path, header=None, compression='zip')
                except Exception:
                    self.data = pd.read_csv(dataset_path, header=None)
            else:
                self.data = pd.read_csv(csv_file, header=None)
                
            self.x = self.data.iloc[:, :-1].values
            self.y = self.data.iloc[:, -1].values
            logger.info(f"Successfully loaded {len(self.y)} real ECG samples.")
        except Exception as e:
            logger.warning(f"Kaggle loading error: {e}. Generating demonstration noise.")
            self.x = np.random.randn(100, 187)
            self.y = np.random.randint(0, 5, 100)

        self.x = np.expand_dims(self.x, axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = torch.tensor(self.x[idx], dtype=torch.float32)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return features, label

def get_dataloader(csv_path: str, batch_size: int = 32, shuffle: bool = True):
    dataset = ECGDataset(csv_file=csv_path, download_kaggle=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
