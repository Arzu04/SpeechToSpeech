import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os

# --- Upload model folder to Hugging Face (only run once, comment out after uploading) ---
# from huggingface_hub import upload_folder
# upload_folder(
#     folder_path=r"C:\Users\ROG\PycharmProjects\speech_to_speech_chatbot\dialogpt_model",
#     repo_id="aarzu004/dialog2"
# )
# st.write("Model folder upload triggered.")  # Optional status message

# Load model and tokenizer from Hugging Face repo
@st.cache_resource
def load_model():
    model_path = "aarzu004/dialog2"  # Hugging Face repo ID
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Transcribe audio file to text
def transcribe(audio_path):
    try:
        # Convert mp3 to wav if needed
        if audio_path.endswith(".mp3"):
            sound = AudioSegment.from_file(audio_path, format="mp3")
            audio_path_wav = audio_path.replace(".mp3", ".wav")
            sound.export(audio_path_wav, format="wav")
        else:
            audio_path_wav = audio_path
    except Exception as e:
        st.error(f"Error decoding audio file: {e}")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path_wav) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")
        return ""

# Generate chatbot reply from user input text
def generate_response(user_input):
    encoded_input = tokenizer(
        user_input + tokenizer.eos_token,
        return_tensors="pt", padding=True, truncation=True
    )
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

# Convert text reply to speech audio file path
def synthesize(text):
    tts = gTTS(text)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# --- STREAMLIT UI ---

st.title("🎙️ Speech-to-Speech Chatbot")

uploaded_file = st.file_uploader("Upload your voice (.mp3 or .wav)", type=["mp3", "wav"])
if uploaded_file:
    # Save uploaded file to temp file with correct extension
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(uploaded_file.read())
        audio_path = temp_file.name

    st.audio(audio_path, format=f"audio/{ext[1:]}")

    if st.button("Transcribe and Chat"):
        with st.spinner("Processing..."):
            text_input = transcribe(audio_path)
            if text_input:
                st.subheader("🗣️ You said:")
                st.write(text_input)

                reply = generate_response(text_input)
                st.subheader("🤖 Bot says:")
                st.write(reply)

                reply_audio_path = synthesize(reply)
                st.audio(reply_audio_path, format="audio/mp3")
            else:
                st.error("Could not transcribe audio. Please try another file.")
