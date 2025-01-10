import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import norm
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score


def load_lfw_from_directory(lfw_dir, min_faces_per_person=1, max_people=50, max_images_per_person=10):
    """
    Load limited subset of LFW dataset
    """
    images = []
    labels = []
    person_names = []

    # Get first max_people directories
    person_dirs = sorted([d for d in os.listdir(lfw_dir)
                          if os.path.isdir(os.path.join(lfw_dir, d))])[:max_people]

    name_to_label = {name: idx for idx, name in enumerate(person_dirs)}

    for person_dir in person_dirs:
        person_path = os.path.join(lfw_dir, person_dir)
        image_files = sorted(os.listdir(person_path))[:max_images_per_person]

        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (100, 100))
                    images.append(img)
                    labels.append(name_to_label[person_dir])
                    if person_dir not in person_names:
                        person_names.append(person_dir)
            except Exception as e:
                print(f"Error loading {image_path}: {str(e)}")

    X = np.array(images)
    y = np.array(labels)

    dataset_info = {
        'n_samples': len(X),
        'image_shape': X[0].shape if len(X) > 0 else None,
        'n_classes': len(person_names),
        'classes': person_names,
        'samples_per_class': np.bincount(y)
    }

    return X, y, person_names, dataset_info

# STEP 3
def preprocess_and_reduce_dimensions(images, n_components=None, variance_threshold=0.35):
    """
    Preprocess facial images and reduce dimensions using PCA

    Parameters:
        images: Array of images (n_samples, height, width, channels)
        n_components: Number of PCA components (if None, use variance_threshold)
        variance_threshold: Minimum explained variance to maintain (default 0.95)

    Returns:
        reduced_data: PCA-reduced data
        pca: Fitted PCA object
        scaler: Fitted StandardScaler object
    """
    # Step 1: Flatten the images
    n_samples = images.shape[0]
    flattened_data = images.reshape(n_samples, -1)

    # Step 2: Scale pixel values to [0,1]
    scaled_data = flattened_data.astype(float) / 255.0

    # Step 3: Standardize features
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(scaled_data)

    # Step 4: Apply PCA
    if n_components is None:
        # First fit PCA with all components
        pca_all = PCA()
        pca_all.fit(standardized_data)

        # Find number of components needed for desired variance
        cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Fit PCA with determined number of components
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(standardized_data)

    return reduced_data, pca, scaler

def analyze_pca_components(pca):
    """
    Analyze and visualize PCA components
    """
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))

    # Scree plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'bo-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')

    # Cumulative variance plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 'ro-')
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% Threshold')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nPCA Summary:")
    print(f"Number of components: {pca.n_components_}")
    print(f"Total explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Show variance explained by top components
    print("\nTop 10 components explained variance:")
    for i, var in enumerate(pca.explained_variance_ratio_[:10], 1):
        print(f"Component {i}: {var:.4f} ({var * 100:.2f}%)")

#STEP 4
def apply_hdbscan_clustering(data, min_cluster_size=30, min_samples=5, metric='euclidean'):
    """
    Apply HDBSCAN clustering to the reduced data

    Parameters:
        data: PCA-reduced feature data
        min_cluster_size: Minimum size of clusters
        min_samples: Number of samples in neighborhood for core points
        metric: Distance metric to use

    Returns:
        clusterer: Fitted HDBSCAN object
        labels: Cluster labels for each data point
    """
    # Initialize and fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        gen_min_span_tree=True  # Required for condensed tree visualization
    )

    # Fit the clusterer
    clusterer.fit(data)

    return clusterer, clusterer.labels_

def analyze_clustering_results(clusterer, labels, true_labels=None):
    """
    Analyze and print clustering statistics
    """
    # Count number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("\nClustering Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Noise percentage: {n_noise / len(labels) * 100:.2f}%")

    # Print cluster sizes
    cluster_sizes = Counter(labels)
    print("\nCluster sizes:")
    for cluster in sorted(cluster_sizes.keys()):
        if cluster != -1:
            print(f"Cluster {cluster}: {cluster_sizes[cluster]} points")

    if true_labels is not None:
        # Calculate cluster purity (how well clusters match true labels)
        purity_scores = calculate_cluster_purity(labels, true_labels)
        print("\nCluster Purity Scores:")
        for cluster, purity in purity_scores.items():
            if cluster != -1:
                print(f"Cluster {cluster}: {purity:.2f}")

def visualize_clusters_2d(data, labels, plot_title="Cluster Visualization"):
    """
    Visualize clusters using t-SNE for dimensionality reduction to 2D
    """
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    # Create scatter plot
    plt.figure(figsize=(12, 8))

    # Plot noise points first
    noise_mask = labels == -1
    plt.scatter(data_2d[noise_mask, 0], data_2d[noise_mask, 1],
                c='gray', label='Noise', alpha=0.5, s=50)

    # Plot clustered points
    clustered = data_2d[~noise_mask]
    cluster_labels = labels[~noise_mask]

    scatter = plt.scatter(clustered[:, 0], clustered[:, 1],
                          c=cluster_labels, cmap='viridis',
                          s=50, alpha=0.8)

    plt.title(plot_title)
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.show()

def calculate_cluster_purity(cluster_labels, true_labels):
    """
    Calculate purity score for each cluster
    """
    purity_scores = {}

    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue

        # Get indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]

        # Find most common true label in this cluster
        if len(cluster_true_labels) > 0:
            label_counts = Counter(cluster_true_labels)
            most_common_count = label_counts.most_common(1)[0][1]
            purity = most_common_count / len(cluster_true_labels)
            purity_scores[cluster_id] = purity

    return purity_scores

def visualize_cluster_exemplars(data, labels, original_images, n_exemplars=5):
    """
    Visualize exemplar images from each cluster
    """
    unique_clusters = sorted(list(set(labels)))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)  # Remove noise cluster

    n_clusters = len(unique_clusters)
    fig, axes = plt.subplots(n_clusters, n_exemplars, figsize=(15, 3 * n_clusters))

    for i, cluster in enumerate(unique_clusters):
        # Get indices of points in this cluster
        cluster_indices = np.where(labels == cluster)[0]

        # Randomly select exemplars
        exemplar_indices = np.random.choice(cluster_indices,
                                            size=min(n_exemplars, len(cluster_indices)),
                                            replace=False)

        for j, idx in enumerate(exemplar_indices):
            if n_clusters == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            ax.imshow(original_images[idx])
            ax.axis('off')
            if j == 0:
                ax.set_title(f'Cluster {cluster}')

    plt.tight_layout()
    plt.show()

#STEP 5
def add_noise_to_images(images, noise_type='gaussian', intensity=0.1):
    """
    Add noise to images
    """
    noisy_images = images.copy()

    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, images.shape)
        noisy_images = np.clip(images + noise, 0, 1)
    elif noise_type == 'salt_and_pepper':
        mask = np.random.random(images.shape) < intensity
        noisy_images[mask] = np.random.choice([0, 1], mask.sum())

    return noisy_images

def compare_distance_metrics(data, metrics=['euclidean', 'manhattan', 'cosine']):
    """
    Compare HDBSCAN with different distance metrics
    """
    results = {}

    for metric in metrics:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=30,
            min_samples=5,
            metric=metric
        )

        cluster_labels = clusterer.fit_predict(data)

        # Calculate metrics excluding noise points
        valid_points = cluster_labels != -1
        if valid_points.sum() > 0:
            sil_score = silhouette_score(
                data[valid_points],
                cluster_labels[valid_points]
            ) if len(np.unique(cluster_labels[valid_points])) > 1 else 0
        else:
            sil_score = 0

        results[metric] = {
            'labels': cluster_labels,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'noise_ratio': (cluster_labels == -1).sum() / len(cluster_labels),
            'silhouette': sil_score
        }

    return results

def visualize_noise_effect(original, noisy):
    """
    Visualize original vs noisy images
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(noisy)
    ax2.set_title('Noisy Image')
    ax2.axis('off')

    plt.show()

# STEP 6
def analyze_cluster_distribution(labels):
    """
    Analyze the distribution of points across clusters

    Parameters:
        labels: Array of cluster labels including noise points (-1)

    Returns:
        dict: Statistical information about cluster distribution
    """
    # Count points in each cluster
    cluster_counts = Counter(labels)

    # Calculate distribution metrics
    n_clusters = len(cluster_counts) - (1 if -1 in cluster_counts else 0)
    n_noise = cluster_counts.get(-1, 0)
    noise_ratio = n_noise / len(labels)

    # Calculate cluster size statistics
    cluster_sizes = [count for label, count in cluster_counts.items() if label != -1]

    if cluster_sizes:
        avg_cluster_size = np.mean(cluster_sizes)
        std_cluster_size = np.std(cluster_sizes)
        min_cluster_size = np.min(cluster_sizes)
        max_cluster_size = np.max(cluster_sizes)

        # Calculate cluster size entropy (measure of balance)
        cluster_probabilities = np.array(cluster_sizes) / sum(cluster_sizes)
        size_entropy = entropy(cluster_probabilities)
    else:
        avg_cluster_size = std_cluster_size = min_cluster_size = max_cluster_size = size_entropy = 0

    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': noise_ratio,
        'avg_cluster_size': avg_cluster_size,
        'std_cluster_size': std_cluster_size,
        'min_cluster_size': min_cluster_size,
        'max_cluster_size': max_cluster_size,
        'size_entropy': size_entropy,
        'cluster_counts': cluster_counts
    }

def visualize_cluster_sizes(labels):
    """
    Create visualizations of cluster size distribution

    Parameters:
        labels: Array of cluster labels
    """
    # Create distribution plot
    plt.figure(figsize=(15, 5))

    # Bar plot of cluster sizes
    plt.subplot(1, 2, 1)
    cluster_counts = Counter(labels)
    clusters = sorted([c for c in cluster_counts.keys() if c != -1])
    sizes = [cluster_counts[c] for c in clusters]

    plt.bar(clusters, sizes)
    if -1 in cluster_counts:
        plt.bar(['Noise'], [cluster_counts[-1]], color='gray')

    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Points')

    # Pie chart of cluster proportions
    plt.subplot(1, 2, 2)
    plt.pie([cluster_counts[c] for c in clusters],
            labels=[f'Cluster {c}' for c in clusters],
            autopct='%1.1f%%')
    if -1 in cluster_counts:
        plt.pie([cluster_counts[-1]], labels=['Noise'],
                colors=['gray'], autopct='%1.1f%%')

    plt.title('Cluster Proportions')

    plt.tight_layout()
    plt.show()

def evaluate_clustering_quality(data, labels):
    """
    Calculate various clustering quality metrics

    Parameters:
        data: Feature matrix
        labels: Cluster labels

    Returns:
        dict: Clustering quality metrics
    """
    # Filter out noise points for metric calculation
    valid_mask = labels != -1
    valid_data = data[valid_mask]
    valid_labels = labels[valid_mask]

    metrics = {}

    # Only calculate metrics if we have valid clusters
    if len(np.unique(valid_labels)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(valid_data, valid_labels)
        except:
            metrics['silhouette'] = 0

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(valid_data, valid_labels)
        except:
            metrics['calinski_harabasz'] = 0
    else:
        metrics['silhouette'] = 0
        metrics['calinski_harabasz'] = 0

    return metrics

def visualize_cluster_separation(data, labels, method='tsne'):
    """
    Visualize cluster separation using dimensionality reduction

    Parameters:
        data: Feature matrix
        labels: Cluster labels
        method: 'tsne' or 'pca' for dimensionality reduction
    """


    # Reduce dimensions for visualization
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    data_2d = reducer.fit_transform(data)

    # Create scatter plot
    plt.figure(figsize=(12, 8))

    # Plot noise points first
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(data_2d[noise_mask, 0], data_2d[noise_mask, 1],
                    c='gray', label='Noise', alpha=0.5, s=50)

    # Plot clustered points
    clustered = data_2d[~noise_mask]
    cluster_labels = labels[~noise_mask]

    scatter = plt.scatter(clustered[:, 0], clustered[:, 1],
                          c=cluster_labels, cmap='viridis',
                          s=50, alpha=0.8)

    plt.title(f'Cluster Separation ({method.upper()})')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.show()

def analyze_cluster_stability(data, labels, n_iterations=10, sample_ratio=0.8):
    """
    Analyze clustering stability through subsampling

    Parameters:
        data: Feature matrix
        labels: Original cluster labels
        n_iterations: Number of subsampling iterations
        sample_ratio: Ratio of data to sample in each iteration

    Returns:
        float: Average agreement score between subsample clusterings
    """

    stability_scores = []
    n_samples = int(len(data) * sample_ratio)

    # Get original HDBSCAN parameters from best clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=5
    )

    for _ in range(n_iterations):
        # Randomly sample data points
        indices = np.random.choice(len(data), n_samples, replace=False)
        subsample_data = data[indices]

        # Perform clustering on subsample
        subsample_labels = clusterer.fit_predict(subsample_data)

        # Compare with original clustering (for overlapping points)
        original_labels_subset = labels[indices]

        # Calculate agreement score (excluding noise points)
        valid_mask = (subsample_labels != -1) & (original_labels_subset != -1)
        if np.any(valid_mask):
            score = adjusted_rand_score(
                original_labels_subset[valid_mask],
                subsample_labels[valid_mask]
            )
            stability_scores.append(score)

    return np.mean(stability_scores) if stability_scores else 0


# Main execution
if __name__ == "__main__":
    # Step 1: Load the dataset
    print("Step 1: Loading the dataset...")
    lfw_dir = "lfw" 
    X, y, target_names, dataset_info = load_lfw_from_directory(lfw_dir, min_faces_per_person=1)

    print("\nInitial Dataset Information:")
    print(f"Number of samples: {dataset_info['n_samples']}")
    print(f"Image shape: {dataset_info['image_shape']}")
    print(f"Number of people: {dataset_info['n_classes']}")

    # Step 2 & 3: Preprocess and reduce dimensions
    print("\nPreprocessing and reducing dimensions...")

    # First run to analyze optimal number of components
    reduced_data, pca, scaler = preprocess_and_reduce_dimensions(X, variance_threshold=0.95)

    # Analyze PCA components
    analyze_pca_components(pca)

    # Print shape information
    print("\nData shapes:")
    print(f"Original data shape: {X.shape}")
    print(f"Reduced data shape: {reduced_data.shape}")

    # Save the reduced data
    print("\nSaving reduced data...")
    np.save('reduced_features.npy', reduced_data)
    np.save('labels.npy', y)
    print("Dimensionality reduction complete! Data saved to 'reduced_features.npy' and 'labels.npy'")

    # Load the reduced data
    print("Loading reduced data...")
    reduced_data = np.load('reduced_features.npy')
    labels = np.load('labels.npy')  # True labels

    # Load original images for visualization
    lfw_dir = "lfw"
    X, y, target_names, dataset_info = load_lfw_from_directory(lfw_dir)

    print("\nApplying HDBSCAN clustering...")

    # Try different parameter combinations
    min_cluster_sizes = [30, 50, 70]
    min_samples_values = [5, 10, 15]

    best_labels = None
    best_score = -1
    best_params = None

    # Simple parameter tuning
    for mcs in min_cluster_sizes:
        for ms in min_samples_values:
            print(f"\nTrying min_cluster_size={mcs}, min_samples={ms}")

            clusterer, cluster_labels = apply_hdbscan_clustering(
                reduced_data,
                min_cluster_size=mcs,
                min_samples=ms
            )

            # Analyze results
            analyze_clustering_results(clusterer, cluster_labels, labels)

            # Keep track of best parameters based on number of clusters and noise points
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            noise_ratio = list(cluster_labels).count(-1) / len(cluster_labels)

            # Simple scoring metric (you might want to adjust this)
            score = n_clusters * (1 - noise_ratio)

            if score > best_score:
                best_score = score
                best_labels = cluster_labels
                best_params = (mcs, ms)

    print(f"\nBest parameters found: min_cluster_size={best_params[0]}, min_samples={best_params[1]}")

    # Visualize results with best parameters
    print("\nGenerating visualizations...")
    print(reduced_data.shape, best_labels.shape, X.shape)
    visualize_clusters_2d(reduced_data, best_labels, "HDBSCAN Clustering Results")
    visualize_cluster_exemplars(reduced_data, best_labels, X)

    # Save clustering results
    np.save('cluster_labels.npy', best_labels)
    print("\nClustering complete! Results saved to 'cluster_labels.npy'")

    # Load reduced data
    reduced_data = np.load('reduced_features.npy')

    # Add noise to data
    noise_intensities = [0.1, 0.2, 0.3]
    noise_types = ['gaussian', 'salt_and_pepper']

    for noise_type in noise_types:
        for intensity in noise_intensities:
            print(f"\nTesting {noise_type} noise with intensity {intensity}")
            noisy_data = add_noise_to_images(reduced_data, noise_type, intensity)

            # Compare different distance metrics on noisy data
            metric_results = compare_distance_metrics(noisy_data)

            # Print results
            print("\nResults for different metrics:")
            for metric, results in metric_results.items():
                print(f"\n{metric}:")
                print(f"Number of clusters: {results['n_clusters']}")
                print(f"Noise ratio: {results['noise_ratio']:.2f}")
                print(f"Silhouette score: {results['silhouette']:.3f}")

            # Save best results
            best_metric = max(metric_results.items(),
                              key=lambda x: x[1]['silhouette'])
            np.save(f'cluster_labels_{noise_type}_{intensity}.npy',
                    best_metric[1]['labels'])

    print("\nAnalysis complete! Results saved to separate .npy files")


def generate_clustering_report(data, labels):
    """
    Generate a comprehensive clustering analysis report

    Parameters:
        data: Feature matrix
        labels: Cluster labels
    """
    # Distribution analysis
    dist_stats = analyze_cluster_distribution(labels)

    # Quality metrics
    quality_metrics = evaluate_clustering_quality(data, labels)

    # Stability analysis
    stability_score = analyze_cluster_stability(data, labels)

    # Create summary report
    print("=== Clustering Analysis Report ===\n")

    print("Cluster Distribution:")
    print(f"Number of clusters: {dist_stats['n_clusters']}")
    print(f"Number of noise points: {dist_stats['n_noise']}")
    print(f"Noise ratio: {dist_stats['noise_ratio']:.2%}")
    print(f"Average cluster size: {dist_stats['avg_cluster_size']:.1f}")
    print(f"Cluster size standard deviation: {dist_stats['std_cluster_size']:.1f}")
    print(f"Cluster size entropy: {dist_stats['size_entropy']:.3f}\n")

    print("Quality Metrics:")
    print(f"Silhouette score: {quality_metrics['silhouette']:.3f}")
    print(f"Calinski-Harabasz score: {quality_metrics['calinski_harabasz']:.3f}")
    print(f"Stability score: {stability_score:.3f}\n")

    # Generate visualizations
    print("Generating visualizations...")
    visualize_cluster_sizes(labels)
    visualize_cluster_separation(data, labels, method='tsne')
    visualize_cluster_separation(data, labels, method='pca')