import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json

# Load the dataset to get the intents and responses
with open(r'E:\sample_app\voicebot.json') as file:  # Update the path accordingly
    data = json.load(file)

# Initialize and load the tokenizer (this should be the same tokenizer used during training)
tokenizer = Tokenizer(oov_token="<OOV>")
# Assuming 'patterns' were used to fit the tokenizer during training
all_patterns = [pattern for intent in data['intents'] for pattern in intent['patterns']]
tokenizer.fit_on_texts(all_patterns)
word_index = tokenizer.word_index

# Load the trained model
model = load_model(r'E:\sample_app\lstm1.0.h5')  # Update the path accordingly

# Function to predict the intent
def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    # Adjust maxlen to match the model's expected input shape
    padded = pad_sequences(sequence, maxlen=13, padding='post')  # Updated maxlen to 13
    prediction = model.predict(padded)
    return np.argmax(prediction)


# Function to get a response for the predicted intent
def get_response(intent_index):
    tag = data['intents'][intent_index]['tag']
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

# Main loop to get user input, predict the intent, and respond
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    predicted_intent_index = predict_intent(user_input)
    response = get_response(predicted_intent_index)
    print(f"Bot: {response}")

