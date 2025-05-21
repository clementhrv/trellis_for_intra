import numpy as np
import matplotlib.pyplot as plt
import os

def setup_intra():
    root_folder_1 = 'datasets_train_final/training/'
    root_folder_2 = 'datasets_train_final/validation/'
    classes = [
            d
            for d in os.listdir(root_folder_1)
            if os.path.isdir(os.path.join(root_folder_1, d))
        ]
    classes_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    file_paths = []
    file_labels = []
    for class_name in classes:
        class_folder_1 = os.path.join(root_folder_1, class_name)
        class_folder_2 = os.path.join(root_folder_2, class_name)
        for file_name in os.listdir(class_folder_1):
            if file_name.endswith('.npz'):
                file_paths.append(os.path.join(class_folder_1, file_name))
                file_labels.append(classes_to_idx[class_name])
        for file_name in os.listdir(class_folder_2):
            if file_name.endswith('.npz'):
                file_paths.append(os.path.join(class_folder_2, file_name))
                file_labels.append(classes_to_idx[class_name])

    return file_paths, file_labels, classes_to_idx, classes

def calcul_intra(file_paths, file_labels):
    aneu_mean = []
    aneu_std = []
    aneu_max = []
    aneu_min = []
    aneu_label = []

    vessel_mean = []
    vessel_std = []
    vessel_max = []
    vessel_min = []
    vessel_label = []

    for file_path, file_label in zip(file_paths, file_labels):
        data_npz = np.load(file_path)
        feats = data_npz['feats']
        coords = data_npz['coords']

        mean = np.mean(feats, axis=0)
        std = np.std(feats, axis=0)
        maxi = np.max(feats, axis=0)
        mini = np.min(feats, axis=0)

        if file_label == 1:
            aneu_mean.append(mean)
            aneu_std.append(std)
            aneu_max.append(maxi)
            aneu_min.append(mini)
            aneu_label.append(file_label)
        else:
            vessel_mean.append(mean)
            vessel_std.append(std)
            vessel_max.append(maxi)
            vessel_min.append(mini)
            vessel_label.append(file_label)

    return (
        np.array(aneu_mean), np.array(aneu_std), np.array(aneu_max), np.array(aneu_min), np.array(aneu_label),
        np.array(vessel_mean), np.array(vessel_std), np.array(vessel_max), np.array(vessel_min), np.array(vessel_label)
    )

def show_pca(aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min):
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    # Combine the mean features for aneurysms and vessels
    mean_features = aneu_mean + vessel_mean
    std_features = aneu_std + vessel_std
    max_features = aneu_max + vessel_max
    min_features = aneu_min + vessel_min
    labels = [0] * len(aneu_mean) + [1] * len(vessel_mean)  # 0 for aneurysms, 1 for vessels

    # Perform PCA
    pca_1 = PCA(n_components=2)
    pca_2 = PCA(n_components=2)
    pca_3 = PCA(n_components=2)
    pca_4 = PCA(n_components=2)
    pca_result_1 = pca_1.fit_transform(mean_features)
    pca_result_2 = pca_2.fit_transform(std_features)
    pca_result_3 = pca_3.fit_transform(max_features)
    pca_result_4 = pca_4.fit_transform(min_features)

    labels = np.array(labels)  # Convertir en array numpy

    for pca_result, title in zip(
        [pca_result_1, pca_result_2, pca_result_3, pca_result_4],
        ['PCA of Mean Features', 'PCA of Std Features', 'PCA of Max Features', 'PCA of Min Features']
    ):
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)

        for label, color, name, alpha in zip([0, 1], ['red', 'blue'], ['Aneurysms', 'Vessels'], [0.4, 0.4]):
            ax.scatter(
                pca_result[labels == label, 0],  # X
                pca_result[labels == label, 1],  # Y
                # pca_result[labels == label, 2],  # Z
                c=color, label=name, alpha=alpha
            )

        ax.set_title(title)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        # ax.set_zlabel('Principal Component 3')
        ax.legend()
        plt.show()


def calcul_anxplore(root_folder):

    aneu = []
    aneu_mean = []
    aneu_std = []
    aneu_max = []
    aneu_min = []

    for files in os.listdir(root_folder):
        if files.endswith(".npz"):
            file_path = os.path.join(root_folder, files)
            data_npz = np.load(file_path)
            feats = data_npz['feats']
            coords = data_npz['coords']

            mean = np.mean(feats, axis=0)
            std = np.std(feats, axis=0)
            maxi = np.max(feats, axis=0)
            mini = np.min(feats, axis=0)

            aneu_mean.append(mean)
            aneu_std.append(std)
            aneu_max.append(maxi)
            aneu_min.append(mini)


    return np.array(aneu_mean), np.array(aneu_std), np.array(aneu_max), np.array(aneu_min)


def show_full_pca(aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min, aneu_mean_anxplore, aneu_std_anxplore, aneu_max_anxplore, aneu_min_anxplore):
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    # Combine the mean features for aneurysms and vessels
    mean_features = np.concatenate([aneu_mean, vessel_mean, aneu_mean_anxplore], axis=0)
    std_features = np.concatenate([aneu_std, vessel_std, aneu_std_anxplore], axis=0)
    max_features = np.concatenate([aneu_max, vessel_max, aneu_max_anxplore], axis=0)
    min_features = np.concatenate([aneu_min, vessel_min, aneu_min_anxplore], axis=0)
    labels = [0] * aneu_mean.shape[0] + [1] * vessel_mean.shape[0] + [2] * aneu_mean_anxplore.shape[0]  # 0 for aneurysms, 1 for vessels, 2 for anxplore

    # Perform PCA
    pca_1 = PCA(n_components=2)
    pca_2 = PCA(n_components=2)
    pca_3 = PCA(n_components=2)
    pca_4 = PCA(n_components=2)
    pca_result_1 = pca_1.fit_transform(mean_features)
    pca_result_2 = pca_2.fit_transform(std_features)
    pca_result_3 = pca_3.fit_transform(max_features)
    pca_result_4 = pca_4.fit_transform(min_features)

    labels = np.array(labels)  # Convertir en array numpy

    for pca_result, title in zip(
        [pca_result_1, pca_result_2, pca_result_3, pca_result_4],
        ['PCA of Mean Features', 'PCA of Std Features', 'PCA of Max Features', 'PCA of Min Features']
    ):
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)

        for label, color, name, alpha in zip([0, 1, 2], ['red', 'blue', 'green'], ['Aneurysms', 'Vessels', 'Anxplore'], [0.4, 0.4, 0.4]):
            ax.scatter(
                pca_result[labels == label, 0],  # X
                pca_result[labels == label, 1],  # Y
                # pca_result[labels == label, 2],  # Z
                c=color, label=name, alpha=alpha
            )
        ax.set_title(title)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        # ax.set_zlabel('Principal Component 3')
        ax.legend()
        plt.show()
    

def show_full_tsne(aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min, aneu_mean_anxplore, aneu_std_anxplore, aneu_max_anxplore, aneu_min_anxplore, perplexity=30, learning_rate=200, n_iter=1000):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Combine the mean features for aneurysms and vessels
    mean_features = np.concatenate([aneu_mean, vessel_mean, aneu_mean_anxplore], axis=0)
    std_features = np.concatenate([aneu_std, vessel_std, aneu_std_anxplore], axis=0)
    max_features = np.concatenate([aneu_max, vessel_max, aneu_max_anxplore], axis=0)
    min_features = np.concatenate([aneu_min, vessel_min, aneu_min_anxplore], axis=0)
    labels = [0] * aneu_mean.shape[0] + [1] * vessel_mean.shape[0] + [2] * aneu_mean_anxplore.shape[0]  # 0 for aneurysms, 1 for vessels, 2 for anxplore

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    tsne_result_1 = tsne.fit_transform(mean_features)
    tsne_result_2 = tsne.fit_transform(std_features)
    tsne_result_3 = tsne.fit_transform(max_features)
    tsne_result_4 = tsne.fit_transform(min_features)

    labels = np.array(labels)  # Convertir en array numpy

    for tsne_result, title in zip(
        [tsne_result_1, tsne_result_2, tsne_result_3, tsne_result_4],
        ['t-SNE of Mean Features', 't-SNE of Std Features', 't-SNE of Max Features', 't-SNE of Min Features']
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for label, color, name, alpha in zip([0, 1, 2], ['red', 'blue', 'green'], ['Aneurysms', 'Vessels', 'Anxplore'], [0.4, 0.4, 0.4]):
            ax.scatter(
                tsne_result[labels == label, 0],  # X
                tsne_result[labels == label, 1],  # Y
                c=color, label=name, alpha=alpha
            )
        ax.set_title(title)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    file_paths_intra, file_labels_intra, classes_to_idx_intra, classes_intra = setup_intra()
    aneu_mean, aneu_std, aneu_max, aneu_min, aneu_label, vessel_mean, vessel_std, vessel_max, vessel_min, vessel_label = calcul_intra(file_paths_intra, file_labels_intra)
    root_folder_anxplore = '/content/dataset_output/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16'
    aneu_mean_anxplore, aneu_std_anxplore, aneu_max_anxplore, aneu_min_anxplore = calcul_anxplore(root_folder_anxplore)
    
    show_full_pca(aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min, aneu_mean_anxplore, aneu_std_anxplore, aneu_max_anxplore, aneu_min_anxplore)
    show_full_tsne(aneu_mean, aneu_std, aneu_max, aneu_min, vessel_mean, vessel_std, vessel_max, vessel_min, aneu_mean_anxplore, aneu_std_anxplore, aneu_max_anxplore, aneu_min_anxplore)
    