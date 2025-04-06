import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras.layers import InputLayer as OriginalInputLayer
from tensorflow.keras.mixed_precision import Policy as DTypePolicy

# Define a custom InputLayer to convert "batch_shape" to "batch_input_shape"
class CustomInputLayer(OriginalInputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

# Patch tf.keras.models.load_model to include our custom objects
original_load_model = tf.keras.models.load_model

def patched_load_model(filepath, *args, **kwargs):
    custom_objects = kwargs.get("custom_objects", {})
    # Register our custom InputLayer and DTypePolicy
    custom_objects["InputLayer"] = CustomInputLayer
    custom_objects["DTypePolicy"] = DTypePolicy
    kwargs["custom_objects"] = custom_objects
    return original_load_model(filepath, *args, **kwargs)

tf.keras.models.load_model = patched_load_model

# --------------------------
# Rest of your code follows:
# --------------------------

print("Loading ISL model for inference...")
model_path = "./Model/retrained_model8.h5"
labels_path = "./Model/labels.txt"

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
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

print(f"Loaded {len(labels)} labels: {', '.join(labels)}")

print("Loading classifier...")
classifier = Classifier(model_path, labels_path)
print("Classifier loaded successfully!")

# Initialize video capture and hand detector
print("Initializing camera and hand detector...")
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)

offset = 20
imgSize = 300

confidence_threshold = 0.5
prediction_smoothing = 3
recent_predictions = []
current_prediction = None
debug_mode = True

print("\nStarting camera feed for detection...")
print("Press 'q' to quit")
print("Press 'd' to toggle debug mode")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to get frame from camera")
        break
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=True)
    
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
            imgWhiteSmall = cv2.resize(imgWhite, (128, 128))
            h, w, _ = imgOutput.shape
            imgOutput[10:10+128, w-138:w-10] = imgWhiteSmall
            imgWhite = cv2.resize(imgWhite, (224, 224)) 
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            all_predictions = []
            if debug_mode:
                top_indices = np.argsort(prediction)[-5:][::-1]
                for idx in top_indices:
                    if prediction[idx] > 0.05:
                        all_predictions.append((labels[idx], prediction[idx] * 100))
            label_text = ""
            if prediction[index] >= confidence_threshold:
                if len(recent_predictions) >= prediction_smoothing:
                    recent_predictions.pop(0)
                recent_predictions.append(index)
                if recent_predictions:
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
        label_text = "No hands detected"
        recent_predictions = []
        current_prediction = None
        all_predictions = []

    cv2.putText(imgOutput, "ISL Recognition", (10, 30), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    if current_prediction:
        label, conf = current_prediction
        overlay = imgOutput.copy()
        panel_height = 70
        panel_y_start = imgOutput.shape[0] - panel_height
        cv2.rectangle(overlay, (0, panel_y_start), (imgOutput.shape[1], imgOutput.shape[0]), 
                     (0, 0, 0), -1)
        imgOutput = cv2.addWeighted(overlay, 0.6, imgOutput, 0.4, 0)
        x_pos = imgOutput.shape[1] // 2
        y_pos = imgOutput.shape[0] - 30
        if conf > 90:
            color = (0, 255, 0)
        elif conf > 70:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)
        letter_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 3)[0]
        letter_x = x_pos - letter_size[0] // 2
        cv2.putText(imgOutput, label, (letter_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 3)
        conf_text = f"{conf:.1f}%"
        conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
        conf_x = x_pos - conf_size[0] // 2
        cv2.putText(imgOutput, conf_text, (conf_x, y_pos + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    elif 'label_text' in locals() and label_text == "No hands detected":
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)[0]
        text_x = (imgOutput.shape[1] - text_size[0]) // 2
        text_y = imgOutput.shape[0] - 30
        overlay = imgOutput.copy()
        bg_y1 = text_y - text_size[1] - 10
        bg_y2 = text_y + 10
        cv2.rectangle(overlay, (text_x - 20, bg_y1), (text_x + text_size[0] + 20, bg_y2), 
                    (0, 0, 0), -1)
        alpha = 0.6
        imgOutput = cv2.addWeighted(overlay, alpha, imgOutput, 1 - alpha, 0)
        cv2.putText(imgOutput, label_text, (text_x, text_y), 
                  cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
    
    if debug_mode and 'all_predictions' in locals() and all_predictions:
        y_pos = 60
        cv2.putText(imgOutput, "Debug Mode - All Predictions:", (10, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for i, (label, conf) in enumerate(all_predictions):
            y_pos += 25
            debug_text = f"{label}: {conf:.1f}%"
            cv2.putText(imgOutput, debug_text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Indian Sign Language Recognition", imgOutput)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("d"):
        debug_mode = not debug_mode
        print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")

print("Closing application...")
cap.release()
cv2.destroyAllWindows()
