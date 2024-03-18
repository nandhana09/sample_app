from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load your pre-trained LSTM model
model_path = r'E:\flaskapp\lstm1.1.2.h5'
model = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return "Welcome to Sakhi!"

def preprocess_input(user_input):
    # This function should convert user input into the format your model expects
    # For example, tokenizing text and padding sequences
    return processed_input

def generate_response(processed_input):
    # Here, interact with your model and return its response
    # This could involve predicting with the model and post-processing the output
    return model_response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    processed_input = preprocess_input(user_input)
    response = generate_response(processed_input)
    return jsonify({"response": response})
def index():
    return send_from_directory('web', 'ui.html')
if __name__ == "__main__":
    app.run(debug=True)
