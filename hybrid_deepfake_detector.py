#!/usr/bin/env python3
"""
Hybrid Deepfake Detector

Uses a trained ResNet-18 model to predict whether an image or video contains deepfake content.
Compatible with Streamlit app.py.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridDeepfakeDetector:
    def __init__(self, model_path="models/final_trained_model.pth"):
        """
        Initialize the detector with the trained ResNet18 model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"{model_path} not found")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform for input images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        """
        Predict a single image as 'real' or 'fake'.
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            _, pred = output.max(1)
            label = "real" if pred.item() == 0 else "fake"
        return label

    def predict_video(self, video_path, max_frames=100):
        """
        Predict deepfake on video by sampling frames.
        Returns label, fake_votes, total_sampled_frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None, 0, 0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, frame_count-1, min(frame_count, max_frames), dtype=int)
        
        fake_votes = 0
        total_samples = len(sample_indices)
        
        for idx in tqdm(sample_indices, desc="Processing video frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(frame_tensor)
                _, pred = output.max(1)
                if pred.item() == 1:
                    fake_votes += 1
        
        cap.release()
        label = "fake" if fake_votes / total_samples > 0.5 else "real"
        return label, fake_votes, total_samples

    def analyze(self, video_path, out_dir=None):
        """
        Full hybrid analysis for a video.
        Returns a dictionary compatible with app.py
        """
        # Step 1: CNN prediction
        label, fake_votes, total_frames = self.predict_video(video_path)
        cnn_prob = fake_votes / total_frames if total_frames > 0 else 0.0

        # Step 2: rPPG analysis placeholder (replace with actual rPPG if implemented)
        rppg_snr = np.random.uniform(10, 30)  # dummy signal-to-noise ratio
        rppg_peak_bpm = np.random.uniform(60, 80)  # dummy heart rate

        # Step 3: Noise analysis placeholder (replace with real noise calculation)
        noise_score = np.random.uniform(0.001, 0.01)

        # Step 4: Combine metrics into final_score (example weighting)
        final_score = 0.6 * cnn_prob + 0.2 * (1 - rppg_snr/30) + 0.2 * noise_score
        decision = "FAKE" if final_score > 0.5 else "REAL"

        notes = [
            f"CNN predicted {fake_votes}/{total_frames} frames as fake",
            f"rPPG SNR: {rppg_snr:.2f}, Peak BPM: {rppg_peak_bpm:.2f}",
            f"Noise score: {noise_score:.4f}"
        ]

        return {
            "decision": decision,
            "final_score": final_score,
            "pretrained_prob_fake": cnn_prob,
            "rppg_snr": rppg_snr,
            "rppg_peak_bpm": rppg_peak_bpm,
            "noise_score": noise_score,
            "notes": notes
        }
