import numpy as np 
from segmentation_latents.segmentation_dataset import SegmentationDataset


def import_data():
    """
    Import data from the given file paths and labels.
    """
    dataset = SegmentationDataset(root_folder='dataset_intra_5fold',meta_path='segmentation_latents/segmentation_meta.json',processing=[0,0,1], number_of_samples=2048)
    return dataset

def calcul_dataset(dataset):
    """
    Calculate mean, std, max, and min for each class in the dataset.
    """
    aneu = []
    vessel = []

    for i in range(len(dataset)):
        data = dataset[i]
        feats = data.x.numpy()
        label = data.y

        for j in range(len(label)):
            if label[j] == 0:
                aneu.append(feats[j])
            else:
                vessel.append(feats[j])
    aneu = np.array(aneu)
    vessel = np.array(vessel)

    return aneu, vessel

def calcul_item(dataset, idx):
    aneu = []
    vessel = []
    data = dataset[idx]
    feats = data.x.numpy()
    label = data.y

    for j in range(len(label)):
        if label[j] == 0:
            aneu.append(feats[j])
        else:
            vessel.append(feats[j])

    aneu = np.array(aneu)
    vessel = np.array(vessel)

    return aneu, vessel


def show_pca(aneu, vessel):
    """
    Show PCA plot for the given data.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    aneu_pca = pca.fit_transform(aneu)
    vessel_pca = pca.fit_transform(vessel)

    plt.scatter(aneu_pca[:, 0], aneu_pca[:, 1], label='Aneurysm', alpha=0.5)
    plt.scatter(vessel_pca[:, 0], vessel_pca[:, 1], label='Vessel', alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset = import_data()
    aneu, vessel = calcul_dataset(dataset)
    show_pca(aneu, vessel)
    aneu, vessel = calcul_item(dataset, 0)
    show_pca(aneu, vessel)