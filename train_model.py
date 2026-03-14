#!/usr/bin/env python3
"""
Full Training Script for Deepfake Detection Model

This script uses ResNet18 to train on image frames with data augmentation,
learning rate scheduling, and optional pretrained weights.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    """Dataset for training/validation using extracted frames"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.rglob('*.jpg'):
                    self.samples.append((str(img_path), label))
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(pretrained_path=None):
    """Create ResNet18 model and optionally load pretrained weights"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
    
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        # Filter out fc layer
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        model.load_state_dict(filtered_state_dict, strict=False)
    
    return model.to(device)

def train(model, train_loader, val_loader, epochs=20, lr=1e-4, save_dir="models"):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100*correct/total:.2f}%"
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    _, pred = output.max(1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            val_acc = 100 * val_correct / val_total
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                logger.info(f"New best model saved with Val Acc={val_acc:.2f}%")
        else:
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%")
        
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    logger.info(f"Training completed! Best Val Acc={best_val_acc:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument("--train_data", type=str, default="data/train", help="Training data dir")
    parser.add_argument("--val_data", type=str, default="data/val", help="Validation data dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained ResNet18 weights")
    args = parser.parse_args()
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    train_dataset = DeepfakeDataset(args.train_data, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    val_loader = None
    if os.path.exists(args.val_data):
        val_dataset = DeepfakeDataset(args.val_data, val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model = create_model(args.pretrained)
    
    train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
