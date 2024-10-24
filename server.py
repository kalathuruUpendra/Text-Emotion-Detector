import sys
import os
from flask import Flask, request, jsonify, render_template

# Add the EmotionDetection directory to the PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'EmotionDetection'))

from emotion_detection import emotion_detector

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emotionDetector', methods=['GET'])
def detect_emotion():
    text = request.args.get('text')
    print(f"Received text: {text}")  # Debug print
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = emotion_detector(text)
    print(f"Emotion detection result: {result}")  # Debug print
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

