import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm

# Number of augmented images to generate per original image
AUGMENTATIONS_PER_IMAGE = 1

print("Starting data augmentation process...")

# Define source and destination directories
data_dir = "Data"
augmented_dir = "AugmentedData"

# Create augmented data directory if it doesn't exist
if os.path.exists(augmented_dir):
    print(f"Directory {augmented_dir} already exists. Removing and recreating.")
    shutil.rmtree(augmented_dir)

os.makedirs(augmented_dir)
    
# Define the labels
labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

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

# Function to apply rotation
def apply_rotation(image):
    angle = random.uniform(-30, 30)
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Function to apply brightness adjustment
def apply_brightness(image):
    brightness_factor = random.uniform(0.7, 1.3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Function to apply shifting
def apply_shift(image):
    height, width = image.shape[:2]
    shift_x = random.uniform(-0.2, 0.2) * width
    shift_y = random.uniform(-0.2, 0.2) * height
    
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(image, translation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return shifted

# Function to apply all augmentations
def augment_image(image):
    augmented = image.copy()
    
    # Apply various augmentations with different probabilities
    if random.random() < 0.3:
        augmented = apply_perspective_transform(augmented)
    
    if random.random() < 0.4:
        augmented = adjust_skin_tone(augmented)
    
    if random.random() < 0.3:
        augmented = apply_noise_and_blur(augmented)
    
    if random.random() < 0.5:
        augmented = apply_rotation(augmented)
    
    if random.random() < 0.5:
        augmented = apply_brightness(augmented)
    
    if random.random() < 0.5:
        augmented = apply_shift(augmented)
    
    # Additional processing (resize to 224x224)
    augmented = cv2.resize(augmented, (224, 224))
    
    return augmented

# Process each label
total_original = 0
total_augmented = 0

for label in labels:
    print(f"Processing label '{label}'...")
    
    # Create directory for this label in augmented data
    label_dir = os.path.join(augmented_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Copy and augment images
    source_dir = os.path.join(data_dir, label)
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} does not exist, skipping.")
        continue
    
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Count original images
    orig_count = len(image_files)
    total_original += orig_count
    
    print(f"Found {orig_count} original images for label '{label}'")
    
    # Process each image with progress bar
    for img_file in tqdm(image_files, desc=f"Augmenting {label}"):
        img_path = os.path.join(source_dir, img_file)
        
        # Read and convert image
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Skip copying original - we'll only use augmented versions
            # filename_orig = f"orig_{img_file}"
            # save_path = os.path.join(label_dir, filename_orig)
            # cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Generate augmented versions
            for i in range(AUGMENTATIONS_PER_IMAGE):
                augmented = augment_image(img)
                
                # Save augmented image
                filename_aug = f"aug{i}_{img_file}"
                save_path = os.path.join(label_dir, filename_aug)
                cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                total_augmented += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"\nAugmentation completed!")
print(f"Original images: {total_original}")
print(f"Augmented images created: {total_augmented}")
print(f"Total images in augmented dataset: {total_original + total_augmented}")
print(f"\nNext steps:")
print(f"1. Run 'python train_augmented.py' to train with the augmented dataset")
print(f"You can view the augmented images in the '{augmented_dir}' directory") 