from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load the trained model
model = tf.keras.models.load_model("your_model_path.h5")

# Define label mapping
LABELS = ["A", "B", "C", "D"]  # Update this with your actual class labels

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        file = request.files['image']
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({"error": "No hands detected"}), 400
        
        # Extract keypoints
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        # Convert to model input format
        keypoints = np.array(keypoints).reshape(1, -1)
        
        # Predict
        predictions = model.predict(keypoints)
        predicted_label = LABELS[np.argmax(predictions)]
        
        return jsonify({"gesture": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

'''
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend communication

# Load Model and Label Encoder
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'keypoint_classifier')
model_path = os.path.join(MODEL_DIR, 'keypoint_classifier.hdf5')
label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.npy')

model = load_model(model_path)
le = LabelEncoder()
le.classes_ = np.load(label_encoder_path, allow_pickle=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Translation dictionary
translations = {
    "Hello": {"hi": "नमस्ते", "kn": "ಹಲೋ"},
    "Thank you": {"hi": "धन्यवाद", "kn": "ಧನ್ಯವಾದ"},
    # Add more as needed
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return jsonify({"error": "No hands detected"}), 400

    for hand_landmarks in results.multi_hand_landmarks:
        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.extend([lm.x, lm.y])

        landmark_list = np.array(landmark_list).reshape(1, -1)
        prediction = model.predict(landmark_list)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

        return jsonify({"gesture": predicted_label})

    return jsonify({"error": "No valid landmarks"}), 400

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    gesture = data.get('gesture')
    language = data.get('language', 'en')

    if not gesture:
        return jsonify({"error": "No gesture provided"}), 400

    translated_text = translations.get(gesture, {}).get(language, gesture)
    return jsonify({"translated": translated_text})

if __name__ == '__main__':
    app.run(debug=True)
# '''
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import numpy as np
# import cv2
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend communication

# # Load Model and Label Encoder
# MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'keypoint_classifier')
# model_path = os.path.join(MODEL_DIR, 'keypoint_classifier.hdf5')
# label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.npy')

# model = load_model(model_path)
# le = LabelEncoder()
# le.classes_ = np.load(label_encoder_path, allow_pickle=True)

# # Mediapipe setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# # Translation dictionary
# translations = {
#     "Hello": {"hi": "नमस्ते", "kn": "ಹಲೋ"},
#     "Thank you": {"hi": "धन्यवाद", "kn": "ಧನ್ಯವಾದ"},
# }

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400

#     file = request.files['image']
#     file_bytes = np.frombuffer(file.read(), np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
#     # Process the image
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
    
#     if not results.multi_hand_landmarks:
#         return jsonify({"error": "No hands detected"}), 400

#     for hand_landmarks in results.multi_hand_landmarks:
#         landmark_list = []
#         for lm in hand_landmarks.landmark:
#             landmark_list.extend([lm.x, lm.y])

#         landmark_list = np.array(landmark_list).reshape(1, -1)
#         prediction = model.predict(landmark_list)
#         predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

#         return jsonify({"gesture": predicted_label})

#     return jsonify({"error": "No valid landmarks"}), 400

# @app.route('/translate', methods=['POST'])
# def translate():
#     data = request.json
#     gesture = data.get('gesture')
#     language = data.get('language', 'en')

#     if not gesture:
#         return jsonify({"error": "No gesture provided"}), 400

#     translated_text = translations.get(gesture, {}).get(language, gesture)
#     return jsonify({"translated": translated_text})

# if __name__ == '__main__':
#     app.run(debug=True)
