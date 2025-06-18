import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os

# Load model
@st.cache_resource
def load_model():
    model_path = r"C:\Users\ROG\Downloads\DL\dialogpt_model"  # UPDATE this path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Transcribe audio
def transcribe(audio_path):
    if audio_path.endswith(".mp3"):
        sound = AudioSegment.from_mp3(audio_path)
        audio_path_wav = audio_path.replace(".mp3", ".wav")
        sound.export(audio_path_wav, format="wav")
    else:
        audio_path_wav = audio_path

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path_wav) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except:
            return ""

# Generate reply
def generate_response(user_input):
    encoded_input = tokenizer(user_input + tokenizer.eos_token, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Convert text to speech
def synthesize(text):
    tts = gTTS(text)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# --- STREAMLIT UI ---

st.title("🎙️ Speech-to-Speech Chatbot")

uploaded_file = st.file_uploader("Upload your voice (.mp3 or .wav)", type=["mp3", "wav"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        audio_path = temp_file.name

    st.audio(audio_path, format="audio/mp3")

    if st.button("Transcribe and Chat"):
        with st.spinner("Processing..."):
            text_input = transcribe(audio_path)
            st.subheader("🗣️ You said:")
            st.write(text_input)

            reply = generate_response(text_input)
            st.subheader("🤖 Bot says:")
            st.write(reply)

            reply_audio_path = synthesize(reply)
            st.audio(reply_audio_path, format="audio/mp3")