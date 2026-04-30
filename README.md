**AI-Based Lie Detection System**  
**Micro-Emotion and Voice Frequency Fusion**  
**Author:** Gedamu Getahun  
**Description**  
An AI system that detects deception by fusing two modalities:  
- **Micro-Emotion Detection:** Facial expression analysis using DeepFace  
- **Voice Frequency Analysis:** Pitch, stress, and vocal pattern detection using a trained Gradient Boosting model  
**Features**  
- Live webcam + microphone detection  
- Upload video file analysis  
- Real-time voice frequency display (Pitch, Stress Level, Range)  
- Micro-emotion shift detection  
- Weighted fusion of audio and video scores  
**How to Run**  
cd Ai-Based-Lie-Detection-main  
 streamlit run app.py  
   
   
## **Model Training**  
bash  
python train_audio_model.py  
## **Requirements**  
- Python 3.8+  
- Streamlit  
- OpenCV  
- DeepFace  
- Librosa  
- Scikit-learn  
- Sounddevice  
   
