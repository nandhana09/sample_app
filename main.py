import pandas as pd
import speech_recognition as sr
from tensorflow.keras.models import load_model
from gtts import gTTS
import os

# Load the dataset
file_path = 'your_dataset.csv'
data = pd.read_csv(file_path)

# Load the trained model
model_path = 'trained_lstm_model.h5'
model = load_model(model_path)

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")  # For Windows
    # For Linux, you may use `os.system("mpg321 output.mp3")`

# Main loop
while True:
    user_input = recognize_speech()
    if user_input.strip().lower() == 'exit':
        break
    # You can process user_input, make predictions using the model, and generate a response
    # For example:
    # response = model.predict(user_input)
    response = "This is an example response from the chatbot."
    text_to_speech(response)
