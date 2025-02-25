# loadwithmodel.py
import joblib

def load_models():
    # Load the sentiment analysis model and CountVectorizer model
    sentiment_model = joblib.load('sentiment_model.pkl')  # Update the path
    count_vectorizer = joblib.load('vectorizer.pkl')  # Update the path
    return sentiment_model, count_vectorizer

