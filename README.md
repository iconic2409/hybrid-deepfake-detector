# 🔍 Hybrid Deepfake Detector

A **Hybrid AI-based Deepfake Detection System** that analyzes videos using multiple signals to determine whether a video is **REAL or FAKE**.

This system combines **deep learning, physiological signals, and artifact detection** to improve reliability compared to traditional single-model detectors.

---

## 🚀 Features

* 🎥 **Video Deepfake Detection**
* 🧠 **CNN-based Visual Artifact Detection (ResNet18)**
* 💓 **rPPG Heart Signal Analysis**
* 🔊 **Noise & Compression Artifact Detection**
* 📊 **Interactive Visual Analytics (Plotly)**
* 📄 **Automated PDF Report Generation**
* 🖥 **Streamlit Web Interface**
* 📈 **Model Training & Evaluation Pipeline**

---

## 🧠 Detection Method

The system uses a **Hybrid Detection Approach** combining three techniques:

### 1️⃣ CNN Visual Analysis

* ResNet18 model trained on deepfake datasets
* Detects visual artifacts introduced during face manipulation

### 2️⃣ rPPG Physiological Signal Detection

* Extracts subtle **heartbeat signals from facial color changes**
* Fake videos often fail to replicate natural biological signals

### 3️⃣ Temporal Noise / Artifact Detection

* Analyzes **frame-level noise patterns**
* Detects inconsistencies caused by deepfake generation

### 🔗 Hybrid Fusion

Final deepfake score is computed using weighted fusion:

```
Final Score = (CNN × 0.6) + (rPPG × 0.3) + (Noise × 0.1)
```

---

## 🏗 System Architecture

```
Video Input
     │
     ▼
Frame Extraction
     │
     ▼
Face Detection (MediaPipe)
     │
     ├── CNN Deepfake Detection
     │
     ├── rPPG Heart Signal Extraction
     │
     └── Noise / Artifact Analysis
            │
            ▼
     Hybrid Score Fusion
            │
            ▼
        REAL / FAKE
```

---

## 🖥 Web Interface

The project includes a **Streamlit-based web application**.

### Features

* Upload video files
* Real-time detection
* Confidence visualization
* Detailed metrics dashboard
* Downloadable PDF report

Run the UI:

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 📂 Project Structure

```
deepfake-detector
│
├── app.py
├── hybrid_deepfake_detector.py
├── prepare_data.py
├── quick_train.py
├── train_model.py
├── evaluate_model.py
│
├── models/
│   └── resnet18_ffpp.pth
│
├── samples/
│   └── fake.mp4
│
├── requirements.txt
├── training_history.json
│
├── README.md
├── PROJECT_REPORT.md
├── TRAINING_GUIDE.md
├── UI_README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/iconic2409/hybrid-deepfake-detector.git
cd deepfake-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🏋️ Model Training

Prepare dataset:

```bash
python prepare_data.py \
--real_videos path/to/real/videos \
--fake_videos path/to/fake/videos \
--output_dir data
```

Quick training:

```bash
python quick_train.py --epochs 10
```

Advanced training:

```bash
python train_model.py --architecture resnet50 --epochs 50
```

---

## 📊 Model Evaluation

Evaluate trained model:

```bash
python evaluate_model.py \
--model_path models/resnet18_ffpp.pth \
--data_dir data/val
```

Metrics generated:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

## 📈 Example Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~98%  |
| Precision | ~99%  |
| Recall    | ~98%  |
| F1 Score  | ~98%  |

*(Results depend on dataset and training configuration)*

---

## 🧰 Technologies Used

* **Python**
* **PyTorch**
* **OpenCV**
* **MediaPipe**
* **Streamlit**
* **Plotly**
* **NumPy / SciPy**
* **Scikit-learn**
* **ReportLab**

---

## ⚠️ Limitations

* Works best when **faces are clearly visible**
* rPPG may fail with **low-quality videos**
* Performance depends on **training dataset diversity**

---

## 🔮 Future Improvements

* Vision Transformer based detection
* Multi-face detection support
* Lip-sync detection using audio analysis
* Ensemble models for higher accuracy

---

## 📜 License

This project is developed for **educational and research purposes**.

---

## 👨‍💻 Author

**Ankit Singh**

Computer Science Student
AI / Machine Learning Projects

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ on GitHub**.
