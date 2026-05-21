# AI Golf Swing Biomechanical Tracker

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Computer_Vision](https://img.shields.io/badge/Computer_Vision-MediaPipe%20%7C%20YOLOv8-orange)
![Data_Science](https://img.shields.io/badge/Data_Science-Pandas%20%7C%20SciPy-lightgrey)

An industry-level, AI-driven biomechanical analysis tool designed to extract key kinematic metrics from golf swing videos. This pipeline leverages MediaPipe for human pose estimation and a custom-trained YOLOv8 model (`best.pt`) for club tracking. It applies advanced signal processing to render a real-time HUD and calculate professional-grade swing mechanics.



https://github.com/user-attachments/assets/c896c5af-c7ce-4702-91a4-da8c2591ba86



## 🚀 Core Capabilities
* **Intelligent Swing Detection:** Utilizes `scipy.signal.find_peaks` on wrist-y coordinates combined with velocity thresholds to automatically segment the swing into Address, Top of Backswing, and Impact phases with a built-in detection cooldown.
* **Camera Validation Engine:** Implements OpenCV's Lucas-Kanade optical flow to detect camera shake and Hough Line transforms to estimate horizon roll angle, ensuring metric reliability.
* **Kinematic Smoothing:** Applies weighted interpolation and Savitzky-Golay (`savgol_filter`) filtering to handle missing landmarks (e.g., occluded wrists) and smooth high-frequency noise from the raw tensor data.
* **Tempo & Amplitude:** Automatically calculates the Backswing-to-Downswing ratio (Tempo) and overall swing amplitude in pixels.

## 📂 Modular Architecture
The codebase is structured into production-ready modules alongside exploratory Jupyter notebooks.

### `/modules/`
* **`common.py`**: Defines the core data structures and state machines using Python `@dataclasses`. Includes `SwingEvent`, `MetricResult`, and Enums for swing `Phase` and tracking `Reliability`.
* **`vision.py`**: The computer vision engine. Houses the `CameraValidator` class for optical flow/roll detection, the MediaPipe/YOLO inference wrappers, and the custom OpenCV HUD drawing tools.
* **`biomech.py`**: The physics and signal processing backend. Contains the `detect_swings_improved` algorithm, temporal velocity calculators, and the `weighted_smooth` functions to clean raw spatial coordinates.

## 🛠️ Installation & Usage

**Prerequisites:** Ensure you have Python 3.11+ installed.

```bash
# Clone the repository
git clone [https://github.com/pineapple-bits/golf-tracker-ai.git](https://github.com/pineapple-bits/golf-tracker-ai.git)
cd golf-tracker-ai

# Install dependencies (OpenCV, Ultralytics, MediaPipe, Pandas, SciPy)
pip install -r requirements.txt

# Run the analysis pipeline on a sample video
python main.py --input data/raw/sample1.mp4 --output data/processed/output_final.mp4
