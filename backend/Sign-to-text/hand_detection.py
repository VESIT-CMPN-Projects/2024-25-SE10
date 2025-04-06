import math
import cv2
import numpy as np
import os
import base64
from flask import Flask, request, jsonify
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and labels
print("Loading ISL model for inference...")
model_path = "Model/retrained_model8.h5"
labels_path = "Model/labels.txt"

if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found!")
    print("Please run train.py first to create the model.")
    exit(1)

labels = []
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels.append(parts[1])
else:
    print("Warning: Labels file not found, using default labels")
    labels = list("123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print(f"Loaded {len(labels)} labels: {', '.join(labels)}")

print("Loading classifier...")
classifier = Classifier(model_path, labels_path)
print("Classifier loaded successfully!")

# Initialize hand detector
detector = HandDetector(maxHands=2, detectionCon=0.5)

# Parameters for processing
offset = 20
imgSize = 300
confidence_threshold = 0.5
prediction_smoothing = 3
recent_predictions = []
current_prediction = None

def process_frame(img):
    global recent_predictions, current_prediction
    # Use draw=True to get an image with MediaPipe hand landmarks
    hands, imgDrawn = detector.findHands(img, draw=True)
    print("Received image shape:", img.shape)
    label_text = "No hands detected"
    current_prediction = None

    if hands:
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        x_min = max(x_min - offset, 0)
        y_min = max(y_min - offset, 0)
        x_max = min(x_max + offset, img.shape[1])
        y_max = min(y_max + offset, img.shape[0])

        if x_max > x_min and y_max > y_min:
            imgCrop = img[y_min:y_max, x_min:x_max]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = (y_max - y_min) / (x_max - x_min)

            if aspectRatio > 1:
                k = imgSize / (y_max - y_min)
                new_w = math.ceil(k * (x_max - x_min))
                imgResize = cv2.resize(imgCrop, (new_w, imgSize))
                wGap = math.ceil((imgSize - new_w) / 2)
                imgWhite[:, wGap:new_w + wGap] = imgResize
            else:
                k = imgSize / (x_max - x_min)
                new_h = math.ceil(k * (y_max - y_min))
                imgResize = cv2.resize(imgCrop, (imgSize, new_h))
                hGap = math.ceil((imgSize - new_h) / 2)
                imgWhite[hGap:new_h + hGap, :] = imgResize

            # Resize to the size expected by the classifier
            imgWhite = cv2.resize(imgWhite, (224, 224))
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if prediction[index] >= confidence_threshold:
                if len(recent_predictions) >= prediction_smoothing:
                    recent_predictions.pop(0)
                recent_predictions.append(index)
                import collections
                counter = collections.Counter(recent_predictions)
                smoothed_index = counter.most_common(1)[0][0]
                label_text = labels[smoothed_index]
                confidence = prediction[smoothed_index] * 100
                current_prediction = (label_text, confidence)
            else:
                if prediction[index] >= 0.3:
                    label_text = labels[index]
                    confidence = prediction[index] * 100
                    current_prediction = (label_text, confidence)
                else:
                    recent_predictions = []
                    current_prediction = None
    else:
        recent_predictions = []
        current_prediction = None

    # Encode the drawn image (with MediaPipe landmarks) to base64 for preview
    _, buffer = cv2.imencode('.jpg', imgDrawn)
    preview_base64 = base64.b64encode(buffer).decode('utf-8')
    preview_data = "data:image/jpeg;base64," + preview_base64

    if current_prediction:
        return {
            "label": current_prediction[0],
            "confidence": f"{current_prediction[1]:.1f}%",
            "preview": preview_data,
        }
    else:
        return {"label": label_text, "confidence": "0.0%", "preview": preview_data}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the base64 image
        image_data = data["image"]
        if "," in image_data:
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        result = process_frame(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=True)
