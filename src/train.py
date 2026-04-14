import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ECG1DCNN
from src.data_loader import get_dataloader
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path="data/mitbih_train.csv", epochs=3, batch_size=32, save_path="models/best_model.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader = get_dataloader(csv_path=data_path, batch_size=batch_size, shuffle=True)
    model = ECG1DCNN(num_classes=5).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                logger.info(f"[{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / (total if total > 0 else 1)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            logger.info("Best model so far - saving to file...")
            torch.save(model.state_dict(), save_path)
            
    logger.info("Training finished!")

if __name__ == "__main__":
    train_model(epochs=1, batch_size=64)
