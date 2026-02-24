import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(
    page_title="Lecture Voice to Notes",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Lecture Voice → Notes Generator")
st.write("Upload a lecture audio file and get text notes automatically.")

# Load Whisper model once
@st.cache_resource
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )

asr = load_asr()

# Upload audio
audio_file = st.file_uploader(
    "Upload lecture audio (MP3 or WAV)",
    type=["mp3", "wav"]
)

if audio_file is not None:
    st.audio(audio_file)

    # Save file
    with open("lecture.wav", "wb") as f:
        f.write(audio_file.read())

    if st.button("Generate Notes"):
        with st.spinner("Transcribing lecture..."):
            result = asr("lecture.wav")
            text = result["text"]

        st.success("Done!")

        st.subheader("📝 Lecture Notes")
        st.write(text)