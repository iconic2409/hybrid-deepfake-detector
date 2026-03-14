#!/usr/bin/env python3
"""
Evaluate Deepfake Detection Model

This script evaluates a trained ResNet-18 model on a dataset and generates:
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix
- JSON report saved to temp_out/evaluation_report.json
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset for evaluation
class EvalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.rglob("*.jpg"):
                    self.samples.append((str(img_path), label))
                for img_path in class_dir.rglob("*.png"):
                    self.samples.append((str(img_path), label))
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224,224,3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Model loader
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.error(f"Model not found at {model_path}")
        return None, device
    
    model = model.to(device)
    model.eval()
    return model, device

# Transform for evaluation
eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Evaluation function
def evaluate(model, device, data_loader):
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = output.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())  # Probability of fake class
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds)
    rec = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names=["real","fake"], save_path="confusion_matrix.png"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    logger.info(f"Confusion matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--data", type=str, required=True, help="Evaluation data directory (real/fake subfolders)")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    model, device = load_model(args.model)
    if model is None:
        return
    
    dataset = EvalDataset(args.data, transform=eval_transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    metrics = evaluate(model, device, data_loader)
    
    logger.info("Evaluation Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Precision: {metrics['precision']*100:.2f}%")
    logger.info(f"Recall: {metrics['recall']*100:.2f}%")
    logger.info(f"F1-score: {metrics['f1']*100:.2f}%")
    logger.info(f"ROC-AUC: {metrics['roc_auc']*100:.2f}%")
    
    plot_confusion_matrix(metrics["confusion_matrix"])
    
    # Save metrics to JSON
    output_dir = Path("temp_out")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    
    metrics_to_save = metrics.copy()
    metrics_to_save["confusion_matrix"] = metrics_to_save["confusion_matrix"].tolist()
    
    with open(report_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    
    logger.info(f"Evaluation metrics saved to {report_path}")

if __name__ == "__main__":
    main()
