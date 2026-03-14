# Hybrid Deepfake Detector (College Mini Project)

This project is a **hybrid deepfake detection system** that combines multiple cues to detect fake media:
- **rPPG (heartbeat signal)** from subtle face color changes.
- **Lip–audio synchronization** using Mediapipe + audio features.
- **Noise/artifact analysis** using frequency filters and heatmaps.
- **Optional pretrained deepfake CNNs** (plug-in ready).
- **Explainability** with heatmaps + plots for feedback.

---

## 🔧 Setup

1. Install **Python 3.10 or 3.11** (mediapipe does not support 3.12+).
2. Create a virtual environment:
   ```bash
   py -3.11 -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # Linux/Mac
 # to create .venv file 
 3) Create a fresh virtual environment (in the project folder)

From your project folder C:\Users\kumar\OneDrive\Desktop\deepfake_mini_project:

PowerShell:

python -m venv .venv


CMD:

python -m venv .venv


This creates a new .venv that points to your machine’s Python.

4) Activate the venv
PowerShell (recommended)

If PowerShell blocks scripts, allow activation for the session:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process


Then activate:

.\.venv\Scripts\Activate.ps1


You should see the prompt change to (.venv) PS C:\...

Command Prompt (cmd.exe)
\.venv\Scripts\activate

If using Git Bash / WSL use the corresponding activate script:
source .venv/Scripts/activate    # Git Bash (Windows)

5) Install required packages

If you have a requirements.txt, run:

pip install -r requirements.txt