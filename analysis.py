import numpy as np

# Load the files
preprocessed_features = np.load('preprocessed_features.npy')
labels = np.load('labels.npy')

# Print basic information
print("Preprocessed Features:")
print(f"Shape: {preprocessed_features.shape}")
print(f"Number of images: {len(preprocessed_features)}")
print(f"Features per image: {preprocessed_features.shape[1]}")
print("\nExample of first preprocessed image:")
print(preprocessed_features[0][:10])  # Show first 10 features

print("\nLabels:")
print(f"Number of labels: {len(labels)}")
print(f"Unique people: {len(np.unique(labels))}")
print("First 10 labels:", labels[:10])