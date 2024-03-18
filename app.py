from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json
import warnings
import os
from gtts import gTTS
from io import BytesIO
import base64
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import speech_recognition as sr

app = Flask(__name__)
app.secret_key = "your_secret_key"

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the dataset to get the intents and responses
with open('voicebot3.1.json') as file:
    data = json.load(file)

tokenizer = Tokenizer(oov_token="<OOV>")
all_patterns = [pattern for intent in data['intents'] for pattern in intent['patterns']]
tokenizer.fit_on_texts(all_patterns)

model = load_model('lstm1.1.3.h5')

def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=13, padding='post')
    prediction = model.predict(padded)
    return np.argmax(prediction)

def get_response(intent_index):
    tag = data['intents'][intent_index]['tag']
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return base64.b64encode(audio_bytes.read()).decode()

def record_audio(duration=5, samplerate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Recording stopped.")
    return audio_data, samplerate

def audio_to_text(audio_data, samplerate):
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        write(tmpfile.name, samplerate, audio_data)
        r = sr.Recognizer()
        with sr.AudioFile(tmpfile.name) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])
    

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    chat_history = session.get('chat_history', [])
    
    predicted_intent_index = predict_intent(user_input)
    response = get_response(predicted_intent_index)
    audio_bytes = text_to_speech(response)
    
    chat_history.append({"user": user_input, "bot": response, "audio": audio_bytes})
    session['chat_history'] = chat_history
    
    return render_template('index.html', user_input=user_input, response=response, chat_history=chat_history)

@app.route('/record_audio', methods=['POST'])
def record_audio_route():
    audio_data, samplerate = record_audio()
    input_text = audio_to_text(audio_data, samplerate)
    
    chat_history = session.get('chat_history', [])
    chat_history.append({"user": input_text, "bot": ""})
    session['chat_history'] = chat_history
    
    return input_text

@app.route('/reset', methods=['POST'])
def reset():
    session.pop('chat_history', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
