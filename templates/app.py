# Save this file as app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# Load your LSTM chatbot model
lstm_model = load_model(r'E:\sample_app\lstm1.1.2.h5')

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    
    # Preprocess the input message here to make it compatible with your LSTM model input
    # This might involve tokenization, padding, etc.

    # Generate a response from your chatbot model
    response = lstm_model.predict(np.array([processed_message]))
    
    # Postprocess the response if necessary

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
