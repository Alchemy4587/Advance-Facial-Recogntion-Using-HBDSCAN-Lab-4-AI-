import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler

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


def visualize_sample_images(X, y, target_names, n_samples=5):
    """
    Visualize random sample images from the dataset
    """
    if len(X) == 0:
        print("No images to visualize!")
        return

    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))

    # Generate random indices
    random_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)

    for i, idx in enumerate(random_indices):
        axes[i].imshow(X[idx])
        axes[i].axis('off')
        axes[i].set_title(f'{target_names[y[idx]]}')

    plt.tight_layout()
    plt.show()


def analyze_class_distribution(y, target_names):
    """
    Analyze and visualize the distribution of samples across classes
    """
    # Create distribution DataFrame
    dist_df = pd.DataFrame({
        'Person': target_names,
        'Number of Images': np.bincount(y)
    })

    # Sort by number of images
    dist_df = dist_df.sort_values('Number of Images', ascending=False)

    # Plot distribution
    plt.figure(figsize=(15, 5))
    sns.barplot(data=dist_df.head(30), x='Person', y='Number of Images')  # Show top 30 people
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Images per Person (Top 30)')
    plt.tight_layout()
    plt.show()

    return dist_df


# Load the dataset from the directory
lfw_dir = "lfw"
X, y, target_names, dataset_info = load_lfw_from_directory(lfw_dir, min_faces_per_person=1)

# Print dataset information
print("\nDataset Information:")
print(f"Number of samples: {dataset_info['n_samples']}")
print(f"Image shape: {dataset_info['image_shape']}")
print(f"Number of people: {dataset_info['n_classes']}")
print(f"Number of people with images: {len(target_names)}")

# Visualize sample images
visualize_sample_images(X, y, target_names)

# Analyze class distribution
distribution_df = analyze_class_distribution(y, target_names)
print("\nClass Distribution Summary (Top 10 people by number of images):")
print(distribution_df.head(10))
