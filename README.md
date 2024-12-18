import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
import argparse

def main(args):
    # Load the features array from a .npy file
    # These are the embeddings extracted from the feature extractor
    # Shape: [N, C], where N is number of images, C is feature dimension
    features = np.load(args.feature_path)
    print("Loaded features with shape:", features.shape)

    # Optional: Dimensionality reduction for better clustering
    # PCA to reduce to, say, 50 dimensions for faster clustering and less noise
    if args.use_pca:
        pca = PCA(n_components=args.pca_components, random_state=0)
        features_reduced = pca.fit_transform(features)
        print(f"Features reduced to shape: {features_reduced.shape}")
    else:
        features_reduced = features

    # Clustering
    if args.cluster_method == 'hdbscan':
        # Use HDBSCAN if you don't know the number of clusters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(features_reduced)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"HDBSCAN found {num_clusters} clusters (label -1 means noise/outliers).")

    elif args.cluster_method == 'kmeans':
        # Use KMeans if you have a guess for the number of clusters
        if args.num_clusters is None:
            raise ValueError("num_clusters must be provided when using K-Means.")
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0)
        labels = kmeans.fit_predict(features_reduced)
        print(f"K-Means formed {args.num_clusters} clusters.")

    else:
        raise ValueError("Invalid cluster_method. Choose 'hdbscan' or 'kmeans'.")

    # Save the cluster labels
    np.save(args.output_labels, labels)
    print(f"Cluster labels saved to {args.output_labels}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster extracted image features.")
    parser.add_argument('--feature_path', type=str, default='image_features.npy', help='Path to the saved features .npy file.')
    parser.add_argument('--cluster_method', type=str, choices=['hdbscan', 'kmeans'], default='hdbscan', help='Clustering method to use.')
    parser.add_argument('--num_clusters', type=int, default=None, help='Number of clusters for K-Means.')
    parser.add_argument('--use_pca', action='store_true', help='Use PCA for dimensionality reduction before clustering.')
    parser.add_argument('--pca_components', type=int, default=50, help='Number of PCA components if using PCA.')
    parser.add_argument('--min_cluster_size', type=int, default=5, help='Minimum cluster size for HDBSCAN.')
    parser.add_argument('--output_labels', type=str, default='cluster_labels.npy', help='Where to save the cluster labels.')

    args = parser.parse_args()
    main(args)
