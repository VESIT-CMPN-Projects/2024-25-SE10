import math
import cv2
import tensorflow as tf
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Starting ISL model training...")

# Define the labels
data_dir = "Data"
labels = ['1']
num_classes = len(labels)

# Create image paths and labels lists
print("Loading dataset...")
image_paths = []
image_labels = []

for label_index, label in enumerate(labels):
    label_dir = os.path.join(data_dir, label)
    if not os.path.exists(label_dir):
        print(f"Warning: Directory {label_dir} does not exist")
        continue
        
    files = os.listdir(label_dir)
    print(f"Found {len(files)} images for label {label}")
    
    for filename in files:
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(label_dir, filename)
            image_paths.append(img_path)
            image_labels.append(label_index)

print(f"Total images found: {len(image_paths)}")

# Convert labels to numpy array
image_labels = np.array(image_labels)

# Create train-validation split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, image_labels, test_size=0.2, random_state=42, stratify=image_labels
)

print(f"Training set: {len(train_paths)} images")
print(f"Validation set: {len(val_paths)} images")

# Create data generators with augmentation for training
batch_size = 32

# Enhanced data augmentation techniques

# Function to apply 3D-like perspective transformation
def apply_perspective_transform(image):
    height, width = image.shape[:2]
    
    # Set the range of the distortion (lower = more severe)
    distortion_range = 0.1
    
    # Get random points for perspective transform
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # Create random offsets within the distortion range
    x_offsets = np.random.uniform(-distortion_range*width, distortion_range*width, size=4)
    y_offsets = np.random.uniform(-distortion_range*height, distortion_range*height, size=4)
    
    # Apply offsets to create destination points
    dst_points = np.float32([
        [0 + x_offsets[0], 0 + y_offsets[0]],
        [width + x_offsets[1], 0 + y_offsets[1]],
        [0 + x_offsets[2], height + y_offsets[2]],
        [width + x_offsets[3], height + y_offsets[3]]
    ])
    
    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the transform
    transformed = cv2.warpPerspective(image, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    return transformed

# Function to apply noise and blur
def apply_noise_and_blur(image):
    # Randomly decide which effects to apply
    effects = random.sample(['gaussian_noise', 'salt_pepper', 'blur', 'none'], k=1)[0]
    
    if effects == 'gaussian_noise':
        # Add Gaussian noise
        row, col, ch = image.shape
        mean = 0
        sigma = random.uniform(1, 25)  # Standard deviation of noise
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    elif effects == 'salt_pepper':
        # Add salt and pepper noise
        s_vs_p = 0.5
        amount = random.uniform(0.004, 0.01)
        noisy = np.copy(image)
        
        # Salt (white) noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255
        
        # Pepper (black) noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0
        
        return noisy
    
    elif effects == 'blur':
        # Apply blur
        blur_type = random.choice(['gaussian', 'median', 'average'])
        if blur_type == 'gaussian':
            ksize = random.choice([3, 5])
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif blur_type == 'median':
            ksize = random.choice([3, 5])
            return cv2.medianBlur(image, ksize)
        else:  # average blur
            ksize = random.choice([3, 5])
            return cv2.blur(image, (ksize, ksize))
            
    else:  # 'none'
        return image

# Function to adjust skin tone
def adjust_skin_tone(image):
    # Randomly adjust hue and saturation to simulate different skin tones
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Adjustments for skin tone variation
    hue_shift = random.uniform(-10, 10)  # Hue adjustment
    sat_scale = random.uniform(0.7, 1.3)  # Saturation adjustment
    
    # Apply adjustments
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)  # Hue is 0-179 in OpenCV
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)  # Saturation
    
    # Convert back to RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Function to apply enhanced augmentation to a batch of images
def apply_enhanced_augmentation(image):
    # Apply with certain probabilities to create variety
    if random.random() < 0.3:  # 30% chance for perspective
        image = apply_perspective_transform(image)
    
    if random.random() < 0.4:  # 40% chance for skin tone adjustment
        image = adjust_skin_tone(image)
    
    if random.random() < 0.3:  # 30% chance for noise/blur
        image = apply_noise_and_blur(image)
        
    return image

# Standard data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    preprocessing_function=apply_enhanced_augmentation
)

# Only basic preprocessing for validation
val_datagen = ImageDataGenerator()

# Function to create generators from file paths
def create_generator(datagen, paths, labels, batch_size):
    # Load and preprocess images
    preprocessed_images = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (224, 224))
        preprocessed_images.append(img)
    
    images = np.array(preprocessed_images, dtype=np.uint8)  # Ensure uint8 type
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Create generator
    return datagen.flow(images, labels_onehot, batch_size=batch_size)

# Create generators
print("Creating data generators with enhanced augmentation...")
train_generator = create_generator(train_datagen, train_paths, train_labels, batch_size)
val_generator = create_generator(val_datagen, val_paths, val_labels, batch_size)

# Define an improved model
print("Building model architecture...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Fine-tune the top layers
base_model.trainable = True
# Freeze the bottom layers (first 100 layers)
for layer in base_model.layers[:100]:
    layer.trainable = False

model = tf.keras.Sequential([
    # Add preprocessing layers at the start
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./255),  # Move rescaling to the model
    
    # Base model
    base_model,
    
    # Top layers
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile with optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Set up callbacks for better training
checkpoint_filepath = 'Model/checkpoint.h5'
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,  # Increased from 10
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Calculate steps per epoch
steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(val_paths) // batch_size

# Train model
print("Starting training with enhanced augmentation...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # Increased from 30
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[model_checkpoint, early_stopping, reduce_lr]
)

# Save the final model
print("Saving model...")
model.save("Model/retrained_model.h5")

# Create labels file
with open("Model/labels.txt", "w") as f:
    for i, label in enumerate(labels):
        f.write(f"{i} {label}\n")

print("Training complete! Enhanced model saved to Model/retrained_model.h5")

# Plot training history if matplotlib is available
try:
    import matplotlib.pyplot as plt
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('Model/training_history.png')
    print("Training history plot saved to Model/training_history.png")
except ImportError:
    print("Matplotlib not available. Skipping training history plot.") 