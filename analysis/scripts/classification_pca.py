import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.nn import MLP
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def setup():
    root_folder = 'data/processed/dataset_classification_intra_5fold'
    classes = [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
    classes_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    file_paths = []
    file_labels = []
    for class_name in classes:
        class_folder = os.path.join(root_folder, class_name)
        for file_name in os.listdir(class_folder):
            if file_name.endswith('.npz'):
                file_paths.append(os.path.join(class_folder, file_name))
                file_labels.append(classes_to_idx[class_name])

    return file_paths, file_labels, classes_to_idx, classes

def calcul(file_paths, file_labels):
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
        feats = data_npz['patchtokens']
        coords = data_npz['indices']

        mean = np.mean(feats, axis=0)
        std = np.std(feats, axis=0)
        maxi = np.max(feats, axis=0)
        mini = np.min(feats, axis=0)

        if file_label == 0:
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

    aneu_std = np.nan_to_num(np.array(aneu_std), nan=0.0, posinf=0.0, neginf=0.0)
    vessel_std = np.nan_to_num(np.array(vessel_std), nan=0.0, posinf=0.0, neginf=0.0)


    return np.array(aneu_mean), np.array(aneu_std), np.array(aneu_max), np.array(aneu_min), np.array(aneu_label), \
           np.array(vessel_mean), np.array(vessel_std), np.array(vessel_max), np.array(vessel_min), np.array(vessel_label)

def calc_tsne(*arrays):
    return [TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200).fit_transform(arr) for arr in arrays]

def calc_pca(*arrays):
    return [PCA(n_components=2).fit_transform(arr) for arr in arrays]

def show_tsne(tsne_results, labels, title="t-SNE"):
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(label='Class Label')
    plt.show()

def show_pca(pca_results, labels, title="PCA"):
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Class Label')
    plt.show()


def classify_mlp_cv(tsne_results, labels, input_size=8, epochs=100, lr=0.01, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_0_list = []
    acc_1_list = []
    f1_list = []
    preds_list = []
    X = []
    y = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(tsne_results, labels)):
        X_train, X_val = tsne_results[train_idx], tsne_results[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)

        model = MLP([input_size, 16, 64, 64, 64, 64, 64, 32, 2], act="ReLU", dropout=0.4)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(dim=1).cpu().numpy()
            for cls in np.unique(y_val.numpy()):
                cls_acc = (preds[y_val.numpy() == cls] == cls).mean()
                if cls == 0:
                    acc_0_list.append(cls_acc*100)
                else:
                    acc_1_list.append(cls_acc*100)
            f1_macro = f1_score(y_val.cpu().numpy(), preds, average="weighted")
            f1_list.append(f1_macro*100)
            preds_list.append(preds)
            X.append(X_val.cpu().numpy())
            y.append(y_val.cpu().numpy())

        # Optionally print per fold
        print(f"Fold {fold+1}: Class 0 Acc: {acc_0_list[-1]:.4f}, Class 1 Acc: {acc_1_list[-1]:.4f}, F1 Macro: {f1_list[-1]:.4f}")


    return acc_0_list, acc_1_list, f1_list, preds_list, X, y

def classify_logistic_regression_cv(results, labels, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_0_list = []
    acc_1_list = []
    f1_list = []
    preds_list = []
    X = []
    y = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(results, labels)):
        X_train, X_val = results[train_idx], results[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        for cls in np.unique(y_val):
            cls_acc = (preds[y_val == cls] == cls).mean()
            if cls == 0:
                acc_0_list.append(cls_acc*100)
            else:
                acc_1_list.append(cls_acc*100)
        f1_macro = f1_score(y_val, preds, average="weighted")
        f1_list.append(f1_macro*100)
        preds_list.append(preds)
        X.append(X_val)
        y.append(y_val)
        

        print(f"Fold {fold+1}: Class 0 Acc: {acc_0_list[-1]:.4f}, Class 1 Acc: {acc_1_list[-1]:.4f}, F1 Macro: {f1_macro:.4f}")

    return acc_0_list, acc_1_list, f1_list, preds_list, X, y

def show_results(preds, X, y, number_of_metrics=4):
    if number_of_metrics == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        metrics = ['Mean']
        axes = np.array([axes])  # Make axes 2D for consistent indexing
    else:
        fig, axes = plt.subplots(number_of_metrics, 2, figsize=(24, 24))
        metrics = ['Mean', 'Std', 'Max', 'Min'][:number_of_metrics]

    for i in range(number_of_metrics):
        # True labels
        axes[i, 0].scatter(X[:, 2*i], X[:, 2*i+1], c=y, cmap='viridis', alpha=0.6)
        axes[i, 0].set_title(f"{metrics[i]} - True Labels")
        axes[i, 0].set_xlabel("Component 1")
        axes[i, 0].set_ylabel("Component 2")
        axes[i, 0].set_aspect('auto')

        # Predicted labels
        axes[i, 1].scatter(X[:, 2*i], X[:, 2*i+1], c=preds, cmap='viridis', alpha=0.6)
        axes[i, 1].set_title(f"{metrics[i]} - Predicted Labels")
        axes[i, 1].set_xlabel("Component 1")
        axes[i, 1].set_ylabel("Component 2")
        axes[i, 1].set_aspect('auto')

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Add a bit of margin at the top
    plt.subplots_adjust(hspace=0.6, wspace=0.2)  # Increase spacing between subplots
    plt.show()

if __name__ == "__main__":
    file_paths, file_labels, classes_to_idx, classes = setup()
    aneu_mean, aneu_std, aneu_max, aneu_min, aneu_label, vessel_mean, vessel_std, vessel_max, vessel_min, vessel_label = calcul(file_paths, file_labels)

    mean = np.concatenate((aneu_mean, vessel_mean), axis=0)
    std = np.concatenate((aneu_std, vessel_std), axis=0)
    max_ = np.concatenate((aneu_max, vessel_max), axis=0)
    min_ = np.concatenate((aneu_min, vessel_min), axis=0)
    label = np.concatenate((aneu_label, vessel_label), axis=0)

    # tsne_results = calc_tsne(mean, std, max_, min_)
    # tsne_concat = np.concatenate(tsne_results, axis=1)
    # model, preds, X_val, y_val = classify_mlp(tsne_concat, label)

    pca_results = calc_pca(mean)
    pca_concat = np.concatenate(pca_results, axis=1)
    acc_0_list, acc_1_list, f1_list, preds_list, X, y = classify_mlp_cv(pca_concat, label, input_size=len(pca_results)*2)
    # acc_0_list, acc_1_list, f1_list, preds_list, X, y = classify_logistic_regression_cv(pca_concat, label)

    print(f"Class 0 Accuracy: {np.mean(acc_0_list):.4f} ± {np.std(acc_0_list):.4f}")
    print(f"Class 1 Accuracy: {np.mean(acc_1_list):.4f} ± {np.std(acc_1_list):.4f}")
    print(f"F1 Score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    show_results(np.concatenate(preds_list, axis=0), np.concatenate(X, axis=0), np.concatenate(y, axis=0), number_of_metrics=len(pca_results))
    # show_pca(pca_results[0], label, title="PCA of Mean Features")

    # show_tsne(tsne_results[0], label, title="t-SNE of Mean Features")
    # model, preds = classify_tsne_mlp(tsne_results[0], label)

    # pca_results = calc_pca(mean)
    # show_tsne(pca_results[0], label, title="PCA of Mean Features")
    