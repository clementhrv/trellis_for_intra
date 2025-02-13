import os 
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import torch
import random


class ClassificationDataset(Dataset):
    def __init__(
        self,
        root_folder,
        meta_path: str,
        switch_to_val: bool = False,
        switch_to_test: bool = False,        
    ):
        
        if switch_to_val:
            root_folder = root_folder.replace("training", "validation")
        elif switch_to_test:
            root_folder = root_folder.replace("training", "test")

        self.classes = [
            d
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.file_paths = []
        self.labels = []
        for cls in self.classes:
            cls_path = os.path.join(root_folder, cls)
            for file in os.listdir(cls_path):
                if file.endswith(".npz"):
                    self.file_paths.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

        with open(meta_path, "r") as fp:
            self.meta = json.loads(fp.read())


    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        data_npz = np.load(file_path)
        feats = data_npz['feats']
        coords = data_npz['coords']

        samplepoints = random.sample(range(feats.shape[0]), 1024)

        feats = feats[samplepoints]
        coords = coords[samplepoints]

        data = Data(
            x=torch.tensor(feats, dtype=torch.float32),
            pos=torch.tensor(coords, dtype=torch.float32),
            y=None,
        )
        
        # Set the label
        if label == 1:
            data.y = torch.tensor([0, 1], dtype=torch.float32)

        else:
            data.y = torch.tensor([1, 0], dtype=torch.float32)
       
        return data

