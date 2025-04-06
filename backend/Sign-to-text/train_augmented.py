import math
import cv2
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Starting ISL model training using pre-augmented dataset...")

# Define the labels
augmented_data_dir = "AugmentedData"
labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
num_classes = len(labels)

# Create image paths and labels lists
print("Loading augmented dataset...")
image_paths = []
image_labels = []

for label_index, label in enumerate(labels):
    label_dir = os.path.join(augmented_data_dir, label)
    if not os.path.exists(label_dir):
        print(f"Warning: Directory {label_dir} does not exist")
        continue
        
    files = os.listdir(label_dir)
    print(f"Found {len(files)} images for label {label}")
    
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
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

# Data generators - minimal augmentation since data is already augmented
batch_size = 32

# Minimal augmentation for training (just normalization)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Function to create generators from file paths
def create_generator(datagen, paths, labels, batch_size):
    # Load and preprocess images
    preprocessed_images = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (224, 224))
        preprocessed_images.append(img)
    
    images = np.array(preprocessed_images)
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Create generator
    return datagen.flow(images, labels_onehot, batch_size=batch_size)

# Create generators
print("Creating data generators...")
train_generator = create_generator(train_datagen, train_paths, train_labels, batch_size)
val_generator = create_generator(val_datagen, val_paths, val_labels, batch_size)

# Define an improved model with MobileNetV2
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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
    patience=15,
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
print("Starting training with pre-augmented dataset...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=40,
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