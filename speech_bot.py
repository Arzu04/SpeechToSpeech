import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import tempfile

# Load model
model_path = r'C:\Users\ROG\Downloads\DL\dialogpt_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Step 1: Speech to Text
from pydub import AudioSegment
import os

def transcribe(audio_path):
    recognizer = sr.Recognizer()

    # Convert MP3 to WAV using pydub
    if audio_path.lower().endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_path)
        wav_path = audio_path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
    else:
        wav_path = audio_path

    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return ""

# Step 2: Text Generation
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

# Step 3: Text to Speech
def synthesize(text):
    tts = gTTS(text)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Run pipeline
input_audio = r'C:\Users\ROG\Downloads\DL\New Recording 5.mp3'
input_text = transcribe(input_audio)
print("Recognized Speech:", input_text)

reply_text = generate_response(input_text)
print("Generated Reply:", reply_text)

output_audio_path = synthesize(reply_text)
print("Reply Audio saved to:", output_audio_path)
