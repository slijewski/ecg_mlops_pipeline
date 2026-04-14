import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECG1DCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        
        # Calculated input for the linear layer:
        # 187 -> pool(2) -> 93 -> pool(2) -> 46
        # 64 channels * 46 length
        self.fc1 = nn.Linear(64 * 46, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Dimensionality reduction block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Block 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flattening for Fully Connected layer
        x = torch.flatten(x, 1)
        
        # Final dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    model = ECG1DCNN()
    x_test = torch.randn(16, 1, 187)
    output = model(x_test)
    logger.info(f"Test output shape: {output.shape}")
