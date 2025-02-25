from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Importing CORS
from loadwithjoblib import load_models  # Import the load_models function
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enabling CORS for all routes

# Load both the sentiment analysis model and CountVectorizer model
sentiment_model, vectorizer = load_models()

# Route to serve the index.html file (homepage)
@app.route('/')

def index():
    return send_from_directory('templates', 'index.html')
    
# Endpoint for sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the request
        data = request.get_json()
        text = data['text']

        # Preprocess the text using CountVectorizer
        vectorized_text = vectorizer.transform([text])  # Assuming you're using CountVectorizer

        # Make prediction
        sentiment = sentiment_model.predict(vectorized_text)

        # Return the sentiment in the response
        response = {
            'sentiment': sentiment[0]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
