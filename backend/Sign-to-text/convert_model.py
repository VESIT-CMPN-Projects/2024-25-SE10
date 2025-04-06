import tensorflowjs as tfjs
import tensorflow as tf
import os

# Create the output directory if it doesn't exist
os.makedirs('web/model', exist_ok=True)

# Load the model
model = tf.keras.models.load_model('Model/retrained_model.h5')

# Save as TensorFlow.js model
tfjs.converters.save_keras_model(model, 'web/model')
print("Model converted successfully for web use!") 