import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def calcul(root_folder):
    aneu_mean = []
    aneu_std = []
    aneu_max = []
    aneu_min = []
    filenames = []

    for files in os.listdir(root_folder):
        if files.endswith(".npz"):
            file_path = os.path.join(root_folder, files)
            data_npz = np.load(file_path)
            feats = data_npz['feats']
            coords = data_npz['coords']

            filenames.append(files)

            mask = coords[:, 1] >= 8
            feats_selected = feats[mask]

            if feats_selected.shape[0] > 0:
                mean = np.mean(feats_selected, axis=0)
                std = np.std(feats_selected, axis=0)
                maxi = np.max(feats_selected, axis=0)
                mini = np.min(feats_selected, axis=0)
            else:
                feat_dim = feats.shape[1]
                mean = np.full(feat_dim, np.nan)
                std = np.full(feat_dim, np.nan)
                maxi = np.full(feat_dim, np.nan)
                mini = np.full(feat_dim, np.nan)

            aneu_mean.append(mean)
            aneu_std.append(std)
            aneu_max.append(maxi)
            aneu_min.append(mini)

    return np.array(aneu_mean), np.array(aneu_std), np.array(aneu_max), np.array(aneu_min), filenames

def calc_pca(*arrays):
    return [PCA(n_components=2).fit_transform(arr) for arr in arrays]

def calc_tsne(*arrays):
    return [TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200).fit_transform(arr) for arr in arrays]

def show_clusters(points, labels, centers, title="Clusters"):
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def cluster_2d_points(points, n_clusters=10, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(points)
    centers = kmeans.cluster_centers_
    return labels, centers

def save_cluster_results(filenames, points, labels, method_name="pca", stat_name="mean", output_dir="./clusters"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({
        "filename": filenames,
        "x": points[:, 0],
        "y": points[:, 1],
        "cluster": labels,
        "method": method_name,
        "stat": stat_name
    })

    filename = f"{output_dir}/clusters_{method_name}_{stat_name}.csv"
    df.to_csv(filename, index=False)
    print(f"[INFO] Cluster results saved to: {filename}")

if __name__ == "__main__":
    root = '/content/dataset_output/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16'

    aneu_mean, aneu_std, aneu_max, aneu_min, filenames = calcul(root)

    # PCA
    pca_results = calc_pca(aneu_mean, aneu_std, aneu_max, aneu_min)
    stat_names = ["mean", "std", "max", "min"]

    for stat_name, pca_result in zip(stat_names, pca_results):
        labels, centers = cluster_2d_points(pca_result)
        show_clusters(pca_result, labels, centers, title=f"PCA {stat_name.capitalize()} Clusters")
        save_cluster_results(filenames, pca_result, labels, method_name="pca", stat_name=stat_name)

    # t-SNE
    tsne_results = calc_tsne(aneu_mean, aneu_std, aneu_max, aneu_min)

    for stat_name, tsne_result in zip(stat_names, tsne_results):
        labels, centers = cluster_2d_points(tsne_result)
        show_clusters(tsne_result, labels, centers, title=f"t-SNE {stat_name.capitalize()} Clusters")
        save_cluster_results(filenames, tsne_result, labels, method_name="tsne", stat_name=stat_name)
