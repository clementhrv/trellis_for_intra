import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from segmentation_latents.segmentation_dataset import SegmentationDataset


def import_data():
    """
    Import data from the given file paths and labels.
    """
    dataset = SegmentationDataset(
        root_folder='dataset_segmentation_intra_5fold',
        meta_path='segmentation_latents/segmentation_meta.json',
        processing=[0, 0, 1],
        number_of_samples=2048
    )
    return dataset


def calcul_dataset(dataset):
    """
    Split features by class (aneurysm vs vessel) over the whole dataset.
    """
    aneu, vessel = [], []
    aneu_mean, vessel_mean = [], []
    aneu_std, vessel_std = [], []
    aneu_max, vessel_max = [], []
    aneu_min, vessel_min = [], []

    for i in range(len(dataset)):
        data = dataset[i]
        feats = data.x.numpy()
        label = data.y

        for j in range(len(label)):
            if label[j] == 0:
                aneu.append(feats[j])
                aneu_mean.append(np.mean(feats[j]))
                aneu_std.append(np.std(feats[j]))
                aneu_max.append(np.max(feats[j]))
                aneu_min.append(np.min(feats[j]))
            else:
                vessel.append(feats[j])
                vessel_mean.append(np.mean(feats[j]))
                vessel_std.append(np.std(feats[j]))
                vessel_max.append(np.max(feats[j]))
                vessel_min.append(np.min(feats[j]))


    return np.array(aneu), np.array(vessel), np.array(aneu_mean), np.array(aneu_std), np.array(aneu_max), np.array(aneu_min), np.array(vessel_mean), np.array(vessel_std), np.array(vessel_max), np.array(vessel_min)


def calcul_item(dataset, idx):
    """
    Split features by class for a single item.
    """
    aneu, vessel = [], []
    aneu_mean, vessel_mean = [], []
    aneu_std, vessel_std = [], []
    aneu_max, vessel_max = [], []
    aneu_min, vessel_min = [], []

    data = dataset[idx]
    feats = data.x.numpy()
    label = data.y

    for j in range(len(label)):
        if label[j] == 0:
            aneu.append(feats[j])
            aneu_mean.append(np.mean(feats[j]))
            aneu_std.append(np.std(feats[j]))
            aneu_max.append(np.max(feats[j]))
            aneu_min.append(np.min(feats[j]))
        else:
            vessel.append(feats[j])
            vessel_mean.append(np.mean(feats[j]))
            vessel_std.append(np.std(feats[j]))
            vessel_max.append(np.max(feats[j]))
            vessel_min.append(np.min(feats[j]))

    return np.array(aneu), np.array(vessel), np.array(aneu_mean), np.array(aneu_std), np.array(aneu_max), np.array(aneu_min), np.array(vessel_mean), np.array(vessel_std), np.array(vessel_max), np.array(vessel_min)



def calcul_mean_std_max_min_item(dataset,idx):
    """
    Calculate mean, std, max, and min for a single item.
    """
    data = dataset[idx]
    feats = data.x.numpy()
    label = data.y

    aneu_mean = np.mean(feats[label == 0], axis=0)
    aneu_std = np.std(feats[label == 0], axis=0)
    aneu_max = np.max(feats[label == 0], axis=0)
    aneu_min = np.min(feats[label == 0], axis=0)

    vessel_mean = np.mean(feats[label == 1], axis=0)
    vessel_std = np.std(feats[label == 1], axis=0)
    vessel_max = np.max(feats[label == 1], axis=0)
    vessel_min = np.min(feats[label == 1], axis=0)

    return aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min

def calcul_mean_std_max_min(dataset):
    """
    Calculate mean, std, max, and min for the entire dataset.
    """
    aneu_mean = []
    aneu_std = []
    aneu_max = []
    aneu_min = []

    vessel_mean = []
    vessel_std = []
    vessel_max = []
    vessel_min = []

    for i in range(len(dataset)):
        data = dataset[i]
        feats = data.x.numpy()
        label = data.y

        aneu_mean.append(np.mean(feats[label == 0], axis=0))
        aneu_std.append(np.std(feats[label == 0], axis=0))
        aneu_max.append(np.max(feats[label == 0], axis=0))
        aneu_min.append(np.min(feats[label == 0], axis=0))

        vessel_mean.append(np.mean(feats[label == 1], axis=0))
        vessel_std.append(np.std(feats[label == 1], axis=0))
        vessel_max.append(np.max(feats[label == 1], axis=0))
        vessel_min.append(np.min(feats[label == 1], axis=0))

    return np.array(aneu_mean), np.array(aneu_std), np.array(aneu_max), np.array(aneu_min), np.array(vessel_mean), np.array(vessel_std), np.array(vessel_max), np.array(vessel_min)

# Scatter plot for mean, std, min, and max
def show_item(aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min):
    """
    Create scatter plots for mean, std, min, and max for aneurysms and vessels.
    """
    plt.figure(figsize=(12, 12))

    # Scatter plot for mean
    plt.subplot(2, 2, 1)
    plt.scatter(range(len(aneu_mean)), aneu_mean, label='Aneurysm Mean', color='red', alpha=0.6)
    plt.scatter(range(len(vessel_mean)), vessel_mean, label='Vessel Mean', color='blue', alpha=0.6)
    plt.title('Mean for Aneurysms and Vessels')
    plt.xlabel('Index')
    plt.ylabel('Mean')
    plt.legend()
    plt.grid(True)

    # Scatter plot for std
    plt.subplot(2, 2, 2)
    plt.scatter(range(len(aneu_std)), aneu_std, label='Aneurysm Std', color='orange', alpha=0.6)
    plt.scatter(range(len(vessel_std)), vessel_std, label='Vessel Std', color='green', alpha=0.6)
    plt.title('Standard Deviation for Aneurysms and Vessels')
    plt.xlabel('Index')
    plt.ylabel('Std')
    plt.legend()
    plt.grid(True)

    # Scatter plot for max
    plt.subplot(2, 2, 3)
    plt.scatter(range(len(aneu_max)), aneu_max, label='Aneurysm Max', color='purple', alpha=0.6)
    plt.scatter(range(len(vessel_max)), vessel_max, label='Vessel Max', color='brown', alpha=0.6)
    plt.title('Max for Aneurysms and Vessels')
    plt.xlabel('Index')
    plt.ylabel('Max')
    plt.legend()
    plt.grid(True)

    # Scatter plot for min
    plt.subplot(2, 2, 4)
    plt.scatter(range(len(aneu_min)), aneu_min, label='Aneurysm Min', color='pink', alpha=0.6)
    plt.scatter(range(len(vessel_min)), vessel_min, label='Vessel Min', color='cyan', alpha=0.6)
    plt.title('Min for Aneurysms and Vessels')
    plt.xlabel('Index')
    plt.ylabel('Min')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def scatter_mean(dataset):
    aneu_mean, vessel_mean, aneu_std, vessel_std, aneu_max, vessel_max, aneu_min, vessel_min = calcul_mean_std_max_min(dataset)

    pca_1 = PCA(n_components=2)
    pca_2 = PCA(n_components=2)
    pca_3 = PCA(n_components=2)
    pca_4 = PCA(n_components=2)

    mean = np.vstack([aneu_mean, vessel_mean])
    std = np.vstack([aneu_std, vessel_std])
    max_ = np.vstack([aneu_max, vessel_max])
    min_ = np.vstack([aneu_min, vessel_min])

    pca_result_1 = pca_1.fit_transform(mean)
    pca_result_2 = pca_2.fit_transform(std)
    pca_result_3 = pca_3.fit_transform(max_)
    pca_result_4 = pca_4.fit_transform(min_)
    labels = np.array([0] * len(aneu_mean) + [1] * len(vessel_mean))  # 0 for aneurysms, 1 for vessels

    for pca_result, title in zip(
        [pca_result_1, pca_result_2, pca_result_3, pca_result_4],
        ['PCA of Mean Features', 'PCA of Std Features', 'PCA of Max Features', 'PCA of Min Features']
    ):
        fig = plt.figure()
        ax = fig.add_subplot(211)

        for label, color, name, alpha in zip([0, 1], ['red', 'blue'], ['Aneurysms', 'Vessels'], [1.0, 0.4]):
            ax.scatter(
                pca_result[labels == label, 0],  # X
                pca_result[labels == label, 1],  # Y
                c=color, label=name, alpha=alpha
            )

        ax.set_title(title)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend()
        plt.show()

    
  

def show_pca(aneu, vessel):
    """
    Show 2D PCA plot for both classes.
    """
    # Combine data for a shared PCA transformation
    combined = np.vstack([aneu, vessel])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)

    aneu_pca = pca_result[:len(aneu)]
    vessel_pca = pca_result[len(aneu):]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(aneu_pca[:, 0], aneu_pca[:, 1], c='red', label='Aneurysms', alpha=1.0)
    plt.scatter(vessel_pca[:, 0], vessel_pca[:, 1], c='blue', label='Vessels', alpha=0.4)
    plt.title('2D PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    dataset = import_data()
    aneu, vessel, aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min = calcul_dataset(dataset)
    show_item(aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min)
    show_pca(aneu, vessel)
    # scatter_mean(dataset)
    # # Example usage of calcul_item for a specific index
    # idx = 0  # Replace with the desired index
    # aneu, vessel, aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min = calcul_item(dataset, idx)
    # show_item(aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min)
    # show_pca(aneu, vessel)

    
