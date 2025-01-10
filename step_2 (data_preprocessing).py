import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# STEP 1 IMPLEMENTATION

def load_lfw_from_directory(lfw_dir, min_faces_per_person=1):
    """
    Loads the LFW dataset from the local directory (lwf)
    Parameters:
        lfw_dir: Path to the LFW directory
        min_faces_per_person: Minimum number of images required per person
    Returns:
        images, labels, names, and dataset info
    """
    images = []
    labels = []
    person_names = []

    # Get all person directories
    person_dirs = [d for d in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, d))]

    # Filter people based on minimum faces criterion
    valid_persons = []
    for person_dir in person_dirs:
        n_images = len(os.listdir(os.path.join(lfw_dir, person_dir)))
        if n_images >= min_faces_per_person:
            valid_persons.append(person_dir)

    # Create a mapping of person name to label
    name_to_label = {name: idx for idx, name in enumerate(sorted(valid_persons))}

    # Load images for valid persons
    for person_dir in valid_persons:
        person_path = os.path.join(lfw_dir, person_dir)
        image_files = os.listdir(person_path)

        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            try:
                # Read image using OpenCV
                img = cv2.imread(image_path)
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize image to a standard size (e.g., 100x100)
                    img = cv2.resize(img, (100, 100))

                    images.append(img)
                    labels.append(name_to_label[person_dir])
                    if person_dir not in person_names:
                        person_names.append(person_dir)
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Create dataset info dictionary
    dataset_info = {
        'n_samples': len(X),
        'image_shape': X[0].shape if len(X) > 0 else None,
        'n_classes': len(person_names),
        'classes': person_names,
        'samples_per_class': np.bincount(y)
    }

    return X, y, person_names, dataset_info


def preprocess_face_data(images):
    """
    Preprocess facial images for clustering

    Parameters:
        images: Array of images (n_samples, height, width, channels)

    Returns:
        preprocessed_data: Flattened and standardized image data
        scaler: Fitted StandardScaler object
    """
    # Step 1: Flatten the images
    n_samples = images.shape[0]
    flattened_data = images.reshape(n_samples, -1)
    print(f"Flattened shape: {flattened_data.shape}")

    # Step 2: Scale pixel values to [0,1]
    scaled_data = flattened_data.astype(float) / 255.0

    # Step 3: Standardize features
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(scaled_data)

    return standardized_data, scaler

def visualize_preprocessing_steps(original_image, flattened_image, standardized_image):
    """
    Visualize the effects of each preprocessing step
    """
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(131)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Scaled Image (0-1)
    plt.subplot(132)
    scaled_image = flattened_image.reshape(original_image.shape)
    plt.imshow(scaled_image)
    plt.title('Scaled Image (0-1)')
    plt.axis('off')

    # Standardized Image
    plt.subplot(133)
    std_image = standardized_image.reshape(original_image.shape)
    plt.imshow(std_image)
    plt.title('Standardized Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def analyze_feature_statistics(data, title):
    """
    Analyze and display statistics of the data
    """
    print(f"\n{title} Statistics:")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")

# Main execution
if __name__ == "__main__":
    # Step 1: Load the dataset
    print("Step 1: Loading the dataset...")
    lfw_dir = "lfw"  #Loads the dataset from the "lfw" folder
    X, y, target_names, dataset_info = load_lfw_from_directory(lfw_dir, min_faces_per_person=1)

    # Print initial dataset information
    print("\nInitial Dataset Information:")
    print(f"Number of samples: {dataset_info['n_samples']}")
    print(f"Image shape: {dataset_info['image_shape']}")
    print(f"Number of people: {dataset_info['n_classes']}")

    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing the data...")
    preprocessed_data, scaler = preprocess_face_data(X)

    # Visualize preprocessing effects on first image
    print("\nVisualizing preprocessing effects...")
    visualize_preprocessing_steps(
        X[0],
        X[0].reshape(-1).astype(float) / 255.0,
        preprocessed_data[0].reshape(X[0].shape)
    )

    # Analyze statistics
    analyze_feature_statistics(X[0], "Original Image")
    analyze_feature_statistics(preprocessed_data[0], "Preprocessed Image")

    # Save the preprocessed data
    print("\nSaving preprocessed data...")
    np.save('preprocessed_features.npy', preprocessed_data)
    np.save('labels.npy', y)
    print("Preprocessing complete! Data saved to 'preprocessed_features.npy' and 'labels.npy'")