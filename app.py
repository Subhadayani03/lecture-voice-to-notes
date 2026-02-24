import streamlit as st
import io
import numpy as np
import librosa
from transformers import pipeline

st.set_page_config(page_title="Lecture Voice to Notes", page_icon="🎙️")

st.title("🎙️ Lecture Voice → Notes Generator")

@st.cache_resource
def load_asr():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

asr = load_asr()

audio_file = st.file_uploader("Upload lecture audio", type=["mp3", "wav"])

if audio_file:
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    
    if st.button("🎤 Generate Notes"):
        with st.spinner("Transcribing..."):
            try:
                # NO FFMPEG NEEDED - pure Python
                audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                result = asr(audio_array)
                st.success("✅ Done!")
                st.subheader("📝 Lecture Notes")
                st.write(result["text"])
            except Exception as e:
                st.error(f"Error: {e}")
