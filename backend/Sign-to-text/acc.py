import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
model = tf.keras.models.load_model("Model/retrained_model.h5")

# Define the path to your data directory
data_dir = "Data"

# Define your labels by reading the subdirectories in the data directory
all_possible_labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
label_to_index_all = {label: i for i, label in enumerate(all_possible_labels)}

# Set the limit for the number of test images per class
limit_per_class = 50

# Lists to store test images and labels
test_images = []
test_labels_indices = []
present_labels = set()

print("Loading limited test data...")
for label in all_possible_labels:
    label_path = os.path.join(data_dir, label)
    image_count = 0
    for filename in os.listdir(label_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                test_images.append(img)
                test_labels_indices.append(label_to_index_all[label])
                present_labels.add(label)
                image_count += 1
                if image_count >= limit_per_class:
                    break
    print(f"Loaded {image_count} images for class: {label}")

test_images = np.array(test_images)
test_labels_indices = np.array(test_labels_indices)
test_labels_categorical = tf.keras.utils.to_categorical(test_labels_indices, num_classes=len(all_possible_labels))

# Make predictions on the test set
print("Making predictions on the limited test data...")
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Get the unique labels that are actually present in the test data
unique_test_labels_indices = sorted(list(set(test_labels_indices)))
actual_labels_in_test = [all_possible_labels[i] for i in unique_test_labels_indices]

# Get the true labels in the same format as predicted labels
true_labels = test_labels_indices

# Evaluate the model
print("\n--- Evaluation on Limited Test Data ---")

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=actual_labels_in_test))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels, labels=unique_test_labels_indices) # Use labels parameter
print(cm)

# Calculate overall accuracy
overall_accuracy = np.mean(predicted_labels == true_labels)
print(f"\nOverall Test Accuracy: {overall_accuracy * 100:.2f}%")

# --- Plotting Confusion Matrix ---
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=actual_labels_in_test, yticklabels=actual_labels_in_test)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('Model/test_confusion_matrix.png')
print("Confusion matrix plot saved to Model/test_confusion_matrix.png")

# --- Save Classification Report to a Text File ---
report = classification_report(true_labels, predicted_labels, target_names=actual_labels_in_test, output_dict=True)

with open("Model/test_classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report(true_labels, predicted_labels, target_names=actual_labels_in_test))
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array_str(cm))
    f.write(f"\n\nOverall Test Accuracy: {overall_accuracy * 100:.2f}%\n")
    f.write("\nDetailed metrics per class (from classification_report dictionary):\n")
    for label_index in unique_test_labels_indices:
        label = all_possible_labels[label_index]
        if label in report:
            f.write(f"\nClass: {label}\n")
            for metric, value in report[label].items():
                f.write(f"  {metric}: {value:.4f}\n")

print("Classification report and confusion matrix details saved to Model/test_classification_report.txt")

print("\n--- Test Evaluation Complete ---")