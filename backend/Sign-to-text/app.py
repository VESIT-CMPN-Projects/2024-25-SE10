from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import os
from cvzone.HandTrackingModule import HandDetector
import time
import base64
import json
import math

app = Flask(__name__)

# Global variables
model = None
labels = []
detector = None
offset = 20
imgSize = 300
counter = 0
# Reduce history size for faster response
prediction_history = []
FRAMES_TO_KEEP = 10  # Reduced from 30 for faster response

# Add preprocessing options - set these to match training
NORMALIZE_METHOD = "inception"  # Options: "simple", "inception"
USE_RGB = True  # Set to True if model was trained on RGB, False for BGR

def preprocess_image(img_array):
    """Preprocess the image to match training conditions"""
    if NORMALIZE_METHOD == "inception":
        # Inception/MobileNet preprocessing: scale to [-1, 1]
        img_array = img_array / 127.5 - 1
    else:
        # Simple normalization to [0, 1]
        img_array = img_array / 255.0
    return img_array

def load_model():
    global model, labels, detector
    
    # Check if model exists
    model_path = 'Model/retrained_model.h5'
    labels_path = 'Model/labels.txt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        return False
    
    # Load the model
    try:
        # Use TF-Lite if available for better performance
        try:
            # Load TF Lite model if exists
            interpreter = tf.lite.Interpreter(model_path=model_path.replace('.h5', '.tflite'))
            interpreter.allocate_tensors()
            print("Using TFLite model for better performance!")
            model = interpreter
        except:
            # Otherwise load normal model
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        
        # Load the labels
        with open(labels_path, 'r') as file:
            labels = [line.strip() for line in file.readlines()]
        print(f"Labels loaded successfully! Found {len(labels)} classes: {labels}")
        
        # Initialize hand detector with maxHands=2 to detect both hands
        detector = HandDetector(maxHands=2, detectionCon=0.8, minTrackCon=0.5)
        
        return True
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    global counter, prediction_history
    
    if request.method == 'POST':
        try:
            # Get image data from POST request
            image_data = request.json.get('image')
            if not image_data:
                return jsonify({'error': 'No image data provided'}), 400
            
            # Decode base64 image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Create a copy of the image for debug overlay
            debug_img = img.copy()
            
            # Add ISL Recognition title at the top
            cv2.putText(debug_img, "ISL Recognition", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Process the image to find hands
            hands, img = detector.findHands(img, draw=False)
            
            if hands:
                # --------- Handle multiple hands like in test.py ---------
                # Find bounding box that contains ALL hands in the frame
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')
                
                # Get the coordinates that encompass all hands
                for hand in hands:
                    x, y, w, h = hand['bbox']
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)
                
                # Add padding to the bounding box
                x_min = max(x_min - offset, 0)
                y_min = max(y_min - offset, 0)
                x_max = min(x_max + offset, img.shape[1])
                y_max = min(y_max + offset, img.shape[0])
                
                # Ensure valid dimensions
                if x_max <= x_min or y_max <= y_min:
                    return jsonify({
                        'prediction': 'Hand detection error',
                        'confidence': 0,
                        'debug_image': base64.b64encode(cv2.imencode('.jpg', debug_img)[1]).decode('utf-8')
                    })
                
                # Crop the region containing all hands
                imgCrop = img[y_min:y_max, x_min:x_max]
                
                if imgCrop.size == 0:
                    return jsonify({
                        'prediction': 'Hand not properly detected', 
                        'confidence': 0,
                        'debug_image': base64.b64encode(cv2.imencode('.jpg', debug_img)[1]).decode('utf-8')
                    })
                
                # Create a white background image
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                
                # Resize while maintaining aspect ratio
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
                
                # Show the hand crop in debug mode
                crop_resized = cv2.resize(imgWhite, (200, 200))
                debug_img_h, debug_img_w = debug_img.shape[:2]
                debug_img[10:210, debug_img_w-210:debug_img_w-10] = crop_resized
                
                # Draw bounding box around all hands in debug image
                cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add text showing number of hands detected
                hand_text = f"{len(hands)} hand{'s' if len(hands) > 1 else ''} detected"
                cv2.putText(debug_img, hand_text, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Resize to 224x224 for MobileNetV2
                imgWhite = cv2.resize(imgWhite, (224, 224))
                
                # Convert color if needed
                if USE_RGB:
                    imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
                
                # Preprocess for model input
                img_array = tf.keras.preprocessing.image.img_to_array(imgWhite)
                img_array = tf.expand_dims(img_array, 0)
                img_array = preprocess_image(img_array)
                
                # Make prediction
                try:
                    if hasattr(model, 'predict'):
                        prediction = model.predict(img_array, verbose=0)
                        index = np.argmax(prediction[0])
                        confidence = float(prediction[0][index])
                        
                        # Debug output for this prediction
                        print(f"Multiple hands - Top prediction: {labels[index]} (index {index}) with confidence {confidence:.2f}")
                        
                        # Display top predictions
                        top_indices = np.argsort(prediction[0])[-3:][::-1]
                        
                        # Add top 3 predictions to debug image
                        for i, idx in enumerate(top_indices):
                            if idx < len(labels):
                                pred_text = f"{labels[idx]}: {prediction[0][idx]:.4f}"
                                cv2.putText(debug_img, pred_text, (20, 90 + i*30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        # Using TFLite
                        input_details = model.get_input_details()
                        output_details = model.get_output_details()
                        model.set_tensor(input_details[0]['index'], img_array)
                        model.invoke()
                        prediction = model.get_tensor(output_details[0]['index'])
                        index = np.argmax(prediction[0])
                        confidence = float(prediction[0][index])
                    
                    # Validate index is within range
                    if index < 0 or index >= len(labels):
                        print(f"Error: Prediction index {index} out of range (0-{len(labels)-1})")
                        index = 0
                        confidence = 0
                except Exception as e:
                    print(f"Prediction error: {e}")
                    index = 0
                    confidence = 0
                
                # Get current prediction
                current_prediction = labels[index] if index < len(labels) else "Error"
                
                # Skip if error or low confidence
                if current_prediction == "Error" or confidence < 0.3:
                    return jsonify({
                        'prediction': 'Low confidence',
                        'confidence': round(confidence * 100, 1) if confidence > 0 else 0,
                        'debug_image': base64.b64encode(cv2.imencode('.jpg', debug_img)[1]).decode('utf-8')
                    })
                
                # Add the current prediction to history for smoothing
                prediction_history.append((current_prediction, confidence))
                if len(prediction_history) > FRAMES_TO_KEEP:
                    prediction_history.pop(0)
                
                # Count occurrences of each prediction for stability
                prediction_counts = {}
                for pred, conf in prediction_history:
                    if pred in prediction_counts:
                        prediction_counts[pred] = (prediction_counts[pred][0] + 1, prediction_counts[pred][1] + conf)
                    else:
                        prediction_counts[pred] = (1, conf)
                
                # Get the most common prediction
                if prediction_counts:
                    sorted_predictions = sorted(prediction_counts.items(), 
                                               key=lambda x: (x[1][0], x[1][1]), 
                                               reverse=True)
                    
                    # Get top prediction
                    top_pred = sorted_predictions[0]
                    smoothed_prediction = top_pred[0]
                    frequency = top_pred[1][0]
                    avg_confidence = top_pred[1][1] / frequency
                    
                    # Only return a stable prediction
                    if frequency >= FRAMES_TO_KEEP * 0.5 and avg_confidence > 0.5:
                        # Display the prediction with large font in debug image
                        img_height, img_width = debug_img.shape[:2]
                        
                        # Draw the main prediction at the bottom
                        font_scale = 7
                        text_size = cv2.getTextSize(smoothed_prediction, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)[0]
                        text_x = (img_width - text_size[0]) // 2
                        text_y = img_height - 80
                        
                        # Draw semi-transparent dark rectangle at the bottom
                        overlay = debug_img.copy()
                        panel_height = 120
                        panel_y_start = img_height - panel_height
                        cv2.rectangle(overlay, (0, panel_y_start), (img_width, img_height), 
                                    (0, 0, 0), -1)
                        debug_img = cv2.addWeighted(overlay, 0.6, debug_img, 0.4, 0)
                        
                        # Draw main prediction with color based on confidence
                        if avg_confidence > 0.9:
                            color = (0, 255, 0)  # Green for high confidence
                        elif avg_confidence > 0.7:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (0, 165, 255)  # Orange for lower confidence
                            
                        cv2.putText(debug_img, smoothed_prediction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_scale, color, 7)
                        
                        # Draw confidence percentage below prediction
                        conf_text = f"{avg_confidence*100:.1f}%"
                        conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
                        conf_x = (img_width - conf_size[0]) // 2
                        conf_y = img_height - 30
                        
                        cv2.putText(debug_img, conf_text, (conf_x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1.5, (255, 255, 255), 3)
                        
                        # Convert debug image to base64 for sending to frontend
                        _, debug_buffer = cv2.imencode('.jpg', debug_img)
                        debug_image_base64 = base64.b64encode(debug_buffer).decode('utf-8')
                        
                        # Return the prediction with landmarks for all hands
                        all_landmarks = [hand['lmList'] for hand in hands]
                        all_bboxes = [hand['bbox'] for hand in hands]
                        
                        return jsonify({
                            'prediction': smoothed_prediction,
                            'confidence': round(avg_confidence * 100, 1),
                            'num_hands': len(hands),
                            'all_landmarks': all_landmarks,
                            'all_bboxes': all_bboxes,
                            'debug_image': debug_image_base64
                        })
                    else:
                        # Still collecting/analyzing
                        return jsonify({
                            'prediction': 'Analyzing...',
                            'confidence': round(avg_confidence * 100, 1),
                            'debug_image': base64.b64encode(cv2.imencode('.jpg', debug_img)[1]).decode('utf-8')
                        })
                else:
                    return jsonify({
                        'prediction': 'Low confidence',
                        'confidence': 0,
                        'debug_image': base64.b64encode(cv2.imencode('.jpg', debug_img)[1]).decode('utf-8')
                    })
            else:
                # No hands detected
                img_height, img_width = debug_img.shape[:2]
                cv2.putText(debug_img, "No hands detected", (img_width//2 - 200, img_height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Convert debug image to base64 for sending to frontend
                _, debug_buffer = cv2.imencode('.jpg', debug_img)
                debug_image_base64 = base64.b64encode(debug_buffer).decode('utf-8')
                
                return jsonify({
                    'prediction': 'No hand detected', 
                    'confidence': 0,
                    'debug_image': debug_image_base64
                })
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/get_quiz_data')
def get_quiz_data():
    # You can expand this with real data from your model
    alphabet_quiz = [
        {
            "question": "What is the sign for 'A'?",
            "options": ["/static/images/A.jpg", "/static/images/B.jpg", "/static/images/C.jpg", "/static/images/D.jpg"],
            "answer": 0
        },
        {
            "question": "What is the sign for 'B'?",
            "options": ["/static/images/D.jpg", "/static/images/B.jpg", "/static/images/F.jpg", "/static/images/G.jpg"],
            "answer": 1
        },
        # Add more questions
    ]
    
    numbers_quiz = [
        {
            "question": "What is the sign for '1'?",
            "options": ["/static/images/1.jpg", "/static/images/2.jpg", "/static/images/3.jpg", "/static/images/4.jpg"],
            "answer": 0
        },
        {
            "question": "What is the sign for '5'?",
            "options": ["/static/images/2.jpg", "/static/images/3.jpg", "/static/images/5.jpg", "/static/images/7.jpg"],
            "answer": 2
        },
        # Add more questions
    ]
    
    return jsonify({
        "alphabet": alphabet_quiz,
        "numbers": numbers_quiz
    })

@app.route('/get_learn_data')
def get_learn_data():
    # Return data for the learn page
    alphabet_data = [
        {"letter": "A", "image": "/static/images/A.jpg", "description": "Make a fist with your thumb resting on the side."},
        {"letter": "B", "image": "/static/images/B.jpg", "description": "Hold your hand up with your palm facing forward and your fingers straight up."},
        # Add more letters
    ]
    
    numbers_data = [
        {"number": "1", "image": "/static/images/1.jpg", "description": "Point your index finger up with your palm facing forward."},
        {"number": "2", "image": "/static/images/2.jpg", "description": "Extend your index and middle fingers with your palm facing forward."},
        # Add more numbers
    ]
    
    return jsonify({
        "alphabet": alphabet_data,
        "numbers": numbers_data
    })

if __name__ == '__main__':
    # Load the model before starting the app
    success = load_model()
    if not success:
        print("Warning: Running without model functionality. Some features may be limited.")
    
    # Create directories if they don't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 