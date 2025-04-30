# Emotion-Detection-With-CNN
# 🎭 Emotion Detection System

A real-time emotion detection system that predicts human emotions using **facial expressions** and **audio cues**. This project uses deep learning for facial analysis and machine learning for speech-based emotion classification.

---

## 📌 Project Overview

This project aims to identify and classify human emotions such as:

- 😄 Happy
- 😢 Sad
- 😠 Angry
- 😐 Neutral
- 😲 Surprise *(optional if included)*

It integrates **computer vision** (using CNNs) and **audio signal processing** (using MFCC features) to perform multimodal emotion detection.

---

## 🚀 Features

✅ Facial emotion recognition using CNNs  
✅ Audio emotion recognition using feature extraction + ML models  
✅ Real-time emotion detection using webcam and microphone  
✅ Data preprocessing pipelines for both image and audio  
✅ Visualization of results with accuracy/loss curves and confusion matrices  

---

## 🛠️ Tech Stack

- **Programming**: Python  
- **Libraries**: OpenCV, TensorFlow/Keras, librosa, NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  
- **Notebook**: Jupyter  
- *(Optionally add: Flask/Streamlit if deployed)*

---

## 📁 Project Structure


---

## 📦 Installation

```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt

# For webcam-based emotion detection
python webcam_emotion_detector.py

# For audio-based emotion classification
python audio_emotion_classifier.py
