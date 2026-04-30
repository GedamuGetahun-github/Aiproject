import streamlit as st
import cv2
import os
import tempfile
import numpy as np
import librosa
import joblib
from PIL import Image
import sounddevice as sd
import wave

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except:
    DEEPFACE_AVAILABLE = False

st.set_page_config(page_title="AI Deception Detector", layout="wide")

st.markdown("""
<style>
body, .stApp, .stMarkdown, h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: white !important;
}
.stButton > button {
    width: 100%;
    padding: 15px;
    font-size: 18px;
    border-radius: 10px;
    background-color: #00d4ff;
    color: #0a0a2e;
    font-weight: bold;
    border: none;
}
.result-truth {
    padding: 20px;
    border-radius: 15px;
    background-color: #00cc66;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin: 20px 0;
}
.result-lie {
    padding: 20px;
    border-radius: 15px;
    background-color: #ff3333;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin: 20px 0;
}
.status-box {
    background: rgba(26, 26, 78, 0.8);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0;
}
.freq-box {
    background: rgba(0, 212, 255, 0.2);
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    margin: 5px 0;
    border: 1px solid #00d4ff;
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "captured_frames" not in st.session_state:
    st.session_state.captured_frames = []
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "audio_recorded" not in st.session_state:
    st.session_state.audio_recorded = False
if "cam_on" not in st.session_state:
    st.session_state.cam_on = False

def load_gif_base64(path):
    import base64
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_bg(gif_file):
    if gif_file is None:
        return
    st.markdown(f"""
        <style>
        .stApp {{
            background: url(data:image/gif;base64,{gif_file});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_audio_model():
    MODEL_PATH = "/home/gedex/Aiproject/Ai-Based-Lie-Detection-main/audio_model.pkl"
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        return None
    except:
        return None

def extract_audio_features_detailed(audio_path):
    """Extract features AND detailed voice frequency info"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3)
        
        # PITCH (Fundamental Frequency F0) - The core of "Voice Frequency"
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_max = np.max(pitch_values)
            pitch_min = np.min(pitch_values)
        else:
            pitch_mean = pitch_std = pitch_max = pitch_min = 0
        
        # Voice frequency range classification
        if pitch_mean < 100:
            freq_range = "Low (Deep voice)"
            stress_hint = "Relaxed"
        elif pitch_mean < 150:
            freq_range = "Normal"
            stress_hint = "Normal"
        elif pitch_mean < 200:
            freq_range = "Elevated"
            stress_hint = "Possible stress"
        else:
            freq_range = "High"
            stress_hint = "High stress likely"
        
        # MFCC (Voice texture/timbre)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Energy (Loudness)
        rms = librosa.feature.rms(y=y)
        energy = np.mean(rms)
        energy_std = np.std(rms)
        
        if energy_std < 0.02:
            stability = "Stable"
        elif energy_std < 0.05:
            stability = "Slightly variable"
        else:
            stability = "Unstable (shaky)"
        
        # Zero Crossing Rate (Voice roughness)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        zcr_std = np.std(librosa.feature.zero_crossing_rate(y=y))
        
        # Speaking rate approximation
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Combine features for model
        features = np.concatenate([
            [pitch_mean, pitch_std, pitch_max, pitch_min, energy, zcr],
            mfcc_means, mfcc_stds
        ])
        
        # Detailed frequency report
        detail = {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': f"{pitch_min:.0f} - {pitch_max:.0f} Hz",
            'freq_range': freq_range,
            'stress_hint': stress_hint,
            'energy': energy,
            'volume_stability': stability,
            'zcr': zcr,
            'tempo': tempo,
            'samples': len(pitch_values)
        }
        
        return features, detail
    except Exception as e:
        return None, None

def record_audio(duration=5, samplerate=16000):
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
        audio_path = "live_recording.wav"
        with wave.open(audio_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        return audio_path
    except Exception as e:
        st.error(f"Audio error: {e}")
        return None

def analyze_face(frame):
    if not DEEPFACE_AVAILABLE:
        return "neutral", 0.6
    
    try:
        small = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        analysis = DeepFace.analyze(img_path=rgb, 
                                     actions=['emotion'], 
                                     enforce_detection=False, 
                                     silent=True,
                                     detector_backend='opencv')
        emotion = analysis[0]['dominant_emotion']
        
        deception_emotions = {'fear': 0.95, 'surprise': 0.85, 'disgust': 0.90, 
                              'angry': 0.75, 'sad': 0.65, 'neutral': 0.60, 'happy': 0.35}
        prob = deception_emotions.get(emotion, 0.50)
        
        return emotion, prob
    except:
        return "unknown", 0.60

def show_landing():
    set_bg(load_gif_base64("landing.gif"))
    st.markdown("<h1 style='text-align: center;'>AI Based Deception Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Reveal truth through vision and emotion</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Upload Video File", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
    with col2:
        if st.button("📷 Live Webcam + Microphone", use_container_width=True):
            st.session_state.page = "live"
            st.session_state.cam_on = False
            st.session_state.captured_frames = []
            st.session_state.audio_recorded = False
            st.rerun()

def show_upload_analysis():
    set_bg(load_gif_base64("analysis.gif"))
    st.markdown("<h3>Upload a video for analysis</h3>", unsafe_allow_html=True)
    
    if st.button("Back to Home"):
        st.session_state.page = "landing"
        st.rerun()

    uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.video(tmp_path)
        
        with st.spinner("Analyzing..."):
            audio_path = tmp_path.replace(".mp4", ".wav")
            os.system(f"ffmpeg -i {tmp_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y 2>/dev/null")
            
            audio_features, audio_details = extract_audio_features_detailed(audio_path)
            audio_model = load_audio_model()
            
            if audio_model and audio_features is not None:
                pred = audio_model.predict([audio_features])[0]
                proba = audio_model.predict_proba([audio_features])[0]
                audio_prob = proba[1] if pred == 1 else proba[0]
            else:
                audio_prob = 0.5
                audio_details = None
            
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in [0, total//4, total//2, 3*total//4, total-1]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i))
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
            
            emotions_list, shifts = [], []
            for f in frames:
                em, sc = analyze_face(f)
                emotions_list.append((em, sc))
            
            if emotions_list:
                video_score = np.mean([s for _, s in emotions_list])
                for i in range(1, len(emotions_list)):
                    if emotions_list[i][0] != emotions_list[i-1][0]:
                        shifts.append(f"{emotions_list[i-1][0]} → {emotions_list[i][0]}")
                if shifts:
                    video_score = min(0.98, video_score * 1.35)
                dominant = max(set([e for e, _ in emotions_list]), key=[e for e, _ in emotions_list].count)
            else:
                video_score = 0.60
                dominant = "unknown"
            
            fused = (audio_prob * 0.6) + (video_score * 0.4)
            prediction = "DECEPTION DETECTED" if fused > 0.4 else "TRUTH"
            result_class = "result-lie" if fused > 0.4 else "result-truth"
        
        st.markdown(f"<div class='{result_class}'>{prediction}</div>", unsafe_allow_html=True)
        st.markdown(f"**Fused Confidence:** {fused:.1%}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Audio Score", f"{audio_prob:.1%}")
        c2.metric("Video Score", f"{video_score:.1%}")
        c3.metric("Emotion", dominant)
        
        if audio_details:
            st.markdown("### 🎤 Voice Frequency Analysis")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.markdown(f"<div class='freq-box'><b>Pitch</b><br>{audio_details['pitch_mean']:.1f} Hz</div>", unsafe_allow_html=True)
            fc2.markdown(f"<div class='freq-box'><b>Range</b><br>{audio_details['freq_range']}</div>", unsafe_allow_html=True)
            fc3.markdown(f"<div class='freq-box'><b>Level</b><br>{audio_details['freq_range']}</div>", unsafe_allow_html=True)
            fc4.markdown(f"<div class='freq-box'><b>Stability</b><br>{audio_details['volume_stability']}</div>", unsafe_allow_html=True)
        
        if shifts:
            st.info(f"⚡ Micro-emotion shifts: {', '.join(shifts)}")

def show_live_analysis():
    set_bg(load_gif_base64("analysis.gif"))
    st.markdown("<h3>Live Micro-Emotion & Voice Frequency Fusion</h3>", unsafe_allow_html=True)
    
    if st.button("Back to Home"):
        st.session_state.page = "landing"
        st.rerun()
    
    st.markdown("""
    <div class='status-box'>
    <b>Instructions:</b><br>
    1. Click <b>Start Webcam</b><br>
    2. Change expression → <b>Capture Frame</b> (3-5 times)<br>
    &nbsp;&nbsp;&nbsp;Try: Neutral → Surprise → Neutral → Fear<br>
    3. Click <b>Record 5s Audio</b> → Speak<br>
    4. Click <b>ANALYZE NOW</b>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not st.session_state.cam_on:
            if st.button("📷 Start Webcam", use_container_width=True):
                st.session_state.cam_on = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Webcam", use_container_width=True):
                st.session_state.cam_on = False
                st.rerun()
    
    if st.session_state.cam_on:
        cam_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_placeholder.image(frame_rgb, channels="RGB", width=450)
                
                with col2:
                    if st.button("📸 Capture Frame", use_container_width=True):
                        st.session_state.captured_frames.append(frame_rgb)
                        st.success(f"Captured: {len(st.session_state.captured_frames)}")
                        st.rerun()
                
                with col3:
                    if not st.session_state.audio_recorded:
                        if st.button("🎤 Record 5s Audio", use_container_width=True):
                            with st.spinner("Recording..."):
                                audio_path = record_audio(duration=5)
                                if audio_path:
                                    st.session_state.audio_path = audio_path
                                    st.session_state.audio_recorded = True
                                    st.success("Done!")
                                    st.rerun()
            cap.release()
        
        if len(st.session_state.captured_frames) > 0:
            st.info(f"📸 {len(st.session_state.captured_frames)} frames captured")
        if st.session_state.audio_recorded:
            st.success("🎤 Audio ready")
        
        if len(st.session_state.captured_frames) >= 2 and st.session_state.audio_recorded:
            if st.button("🔍 ANALYZE NOW", use_container_width=True):
                with st.spinner("Analyzing micro-emotions and voice frequencies..."):
                    emotions_list, shifts = [], []
                    
                    for f in st.session_state.captured_frames:
                        f_bgr = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
                        em, sc = analyze_face(f_bgr)
                        emotions_list.append((em, sc))
                    
                    if emotions_list:
                        video_score = np.mean([s for _, s in emotions_list])
                        for i in range(1, len(emotions_list)):
                            if emotions_list[i][0] != emotions_list[i-1][0]:
                                shifts.append(f"{emotions_list[i-1][0]} → {emotions_list[i][0]}")
                        if shifts:
                            video_score = min(0.98, video_score * 1.35)
                        dominant = max(set([e for e, _ in emotions_list]), 
                                      key=[e for e, _ in emotions_list].count)
                    else:
                        video_score = 0.60
                        dominant = "unknown"
                    
                    audio_features, audio_details = extract_audio_features_detailed(st.session_state.audio_path)
                    audio_model = load_audio_model()
                    
                    if audio_model and audio_features is not None:
                        pred = audio_model.predict([audio_features])[0]
                        proba = audio_model.predict_proba([audio_features])[0]
                        audio_prob = proba[1] if pred == 1 else proba[0]
                    else:
                        audio_prob = 0.5
                        audio_details = None
                    
                    fused = (audio_prob * 0.6) + (video_score * 0.4)
                    prediction = "DECEPTION DETECTED" if fused > 0.4 else "TRUTH"
                    result_class = "result-lie" if fused > 0.4 else "result-truth"
                
                st.markdown("---")
                st.markdown(f"<div class='{result_class}'>{prediction}</div>", unsafe_allow_html=True)
                st.markdown(f"**Fused Confidence:** {fused:.1%}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Audio Score", f"{audio_prob:.1%}")
                c2.metric("Video Score", f"{video_score:.1%}")
                c3.metric("Dominant Emotion", dominant)
                
                # VOICE FREQUENCY DISPLAY - Matching the title
                if audio_details:
                    st.markdown("---")
                    st.markdown("### 🎤 Voice Frequency Analysis (Fusion)")
                    fc1, fc2, fc3, fc4 = st.columns(4)
                    fc1.markdown(f"<div class='freq-box'><b>📊 Pitch</b><br><h3>{audio_details['pitch_mean']:.1f} Hz</h3></div>", unsafe_allow_html=True)
                    fc2.markdown(f"<div class='freq-box'><b>📏 Range</b><br>{audio_details['freq_range']}</div>", unsafe_allow_html=True)
                    fc3.markdown(f"<div class='freq-box'><b>🔊 Level</b><br>{audio_details['freq_range']}</div>", unsafe_allow_html=True)
                    fc4.markdown(f"<div class='freq-box'><b>📈 Stress Hint</b><br>{audio_details['stress_hint']}</div>", unsafe_allow_html=True)
                    
                    fc5, fc6 = st.columns(2)
                    fc5.markdown(f"<div class='freq-box'><b>Volume Stability</b><br>{audio_details['volume_stability']}</div>", unsafe_allow_html=True)
                    fc6.markdown(f"<div class='freq-box'><b>Pitch Variation</b><br>{audio_details['pitch_std']:.1f} Hz</div>", unsafe_allow_html=True)
                
                if shifts:
                    st.warning(f"⚡ Micro-emotion shifts: {', '.join(shifts)}")
                else:
                    st.info("No micro-emotion shifts detected")
                
                if st.button("🔄 Reset"):
                    st.session_state.captured_frames = []
                    st.session_state.audio_recorded = False
                    st.rerun()

if st.session_state.page == "landing":
    show_landing()
elif st.session_state.page == "upload":
    show_upload_analysis()
elif st.session_state.page == "live":
    show_live_analysis()
