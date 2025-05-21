import os 
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import torch_geometric.transforms as T
import torch
import random


class ClassificationDataset(Dataset):
    def __init__(
        self,
        root_folder,
        meta_path: str,
        switch_to_val: bool = False,
        switch_to_test: bool = False, 
        number_of_samples: int = 512,
        number_of_connections: int = 6,
        processing: list = [1,0],
        indices = None,
    ):
        
        if switch_to_val:
            root_folder = root_folder.replace("training", "validation")
        elif switch_to_test:
            root_folder = root_folder.replace("training", "test")


        self.number_of_samples = number_of_samples
        self.number_of_connections = number_of_connections
        self.processing = processing

        self.classes = [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Lister tous les fichiers npz et leur label de classe
        all_npz = []
        all_labels = []

        for cls in self.classes:
            cls_folder = os.path.join(root_folder, cls)
            for file in os.listdir(cls_folder):
                if file.endswith(".npz"):
                    all_npz.append(os.path.join(cls_folder, file))
                    all_labels.append(self.class_to_idx[cls])

        # Trier pour assurer le bon alignement
        sorted_pairs = sorted(zip(all_npz, all_labels))
        all_npz, all_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Si indices spécifiés (KFold), ne garder que ceux-là
        if indices is not None:
            self.file_paths = [all_npz[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
        else:
            self.file_paths = list(all_npz)
            self.labels = list(all_labels)

        with open(meta_path, "r") as fp:
            self.meta = json.loads(fp.read())


    def __len__(self):
        return len(self.file_paths)
    
    def sample_points(self, feats, coords, num_points=1024):
        if feats.shape[0] > num_points:
            sample_indices = random.sample(range(feats.shape[0]), num_points)
        else:
            sample_indices = random.choice(feats.shape[0], num_points, replace=True)
        feats = feats[sample_indices]
        coords = coords[sample_indices]
        return feats, coords 
    
    def __getitem__(self, idx):
        
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        data_npz = np.load(file_path)
        feats = data_npz['feats']
        coords = data_npz['coords']

        # Sample points
        feats, coords = self.sample_points(feats, coords, num_points=self.number_of_samples)

        features = np.empty((feats.shape[0], 0))
        if self.processing[0] == 1:
            features = np.concatenate((features, coords), axis=1)
        if self.processing[1] == 1:
            features = np.concatenate((features, feats), axis=1)

        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            pos=torch.tensor(coords, dtype=torch.float32),
            y=None,
        )

        data = T.KNNGraph(k=self.number_of_connections)(data)
        
        # Set the label
        if label == 1:
            data.y = torch.tensor([0, 1], dtype=torch.float32)

        else:
            data.y = torch.tensor([1, 0], dtype=torch.float32)
       
        return data

