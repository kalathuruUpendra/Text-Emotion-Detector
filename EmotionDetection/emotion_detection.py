import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore')  # Suppress other warnings

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define the path for the model and tokenizer
MODEL_PATH = 'C:/Users/upend/OneDrive/Desktop/new_final_project/emotion_model.h5'
TOKENIZER_PATH = 'C:/Users/upend/OneDrive/Desktop/new_final_project/tokenizer.json'

# Load the trained LSTM model
model = load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, 'r') as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

# Preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def emotion_detector(text_to_analyze):
    """
    Detect emotion in the provided text using the pre-trained model.
    
    Args:
        text_to_analyze (str): Text to analyze.
        
    Returns:
        dict: Emotion prediction result along with the model name.
    """
    try:
        print(f"Text to analyze: {text_to_analyze}")  # Debug print
        # Preprocess and tokenize the input text
        preprocessed_text = preprocess_text(text_to_analyze)
        text_sequence = tokenizer.texts_to_sequences([preprocessed_text])
        text_padded = pad_sequences(text_sequence, maxlen=250)

        # Predict probabilities
        predictions = model.predict(text_padded)[0]

        # Define the emotion labels
        emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        # Create a dictionary for the predictions
        emotion_scores = {label: float(score) for label, score in zip(emotion_labels, predictions)}
        
        # Determine the dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        return {
            'dominant_emotion': dominant_emotion
        }
    except Exception as e:
        print(f"Error in emotion_detector: {str(e)}")  # Debug print
        return {"error": str(e)}
