# AI-Based Lie Detection System

## Micro-Emotion and Voice Frequency Fusion

**Author:** Gedamu Getahun  

## Description

An AI system that detects deception by fusing two modalities:
- **Micro-Emotion Detection:** Facial expression analysis using DeepFace
- **Voice Frequency Analysis:** Pitch, stress, and vocal pattern detection using a trained Gradient Boosting model

## Features

- Live webcam + microphone detection
- Upload video file analysis
- Real-time voice frequency display (Pitch, Stress Level, Range)
- Micro-emotion shift detection
- Weighted fusion of audio and video scores

## How to Run

```bash
cd Ai-Based-Lie-Detection-main
streamlit run app.py
