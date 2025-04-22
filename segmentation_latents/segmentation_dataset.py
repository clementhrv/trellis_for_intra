import os 
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import torch
import random
from torch_geometric.nn import knn
import meshio
import open3d as o3d


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_folder,
        meta_path: str,
        processing: str = "normal",
        switch_to_val: bool = False,
        switch_to_test: bool = False,   

    ):
        
        if switch_to_val:
            root_folder = root_folder.replace("training", "validation")

        self.npzfile_paths = []
        self.adfile_paths = []
        self.labels = []
        self.processing = processing

        for file in os.listdir(root_folder):
            if file.endswith(".npz"):
                self.npzfile_paths.append(os.path.join(root_folder, file))
            elif file.endswith(".ad"):
                self.adfile_paths.append(os.path.join(root_folder, file))

        with open(meta_path, "r") as fp:
            self.meta = json.loads(fp.read())


    def __len__(self):
        return len(self.adfile_paths)
    
    def get_encoding(self, file_name: str):
        file_name = file_name.replace("-_norm.ad", "_full.npz")

        data_npz = np.load(file_name)
        feats = data_npz['feats']
        coords = data_npz['coords']

        return feats, coords
    
    def scale_pos(self, cloud, coords):
        c_min, _ = cloud.pos.min(dim=0)
        c_max, _ = cloud.pos.max(dim=0)

        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float32)

        pc_min, _ = coords.min(dim=0)
        pc_max, _ = coords.max(dim=0)

        scale = (c_max - c_min) / (pc_max - pc_min)
        shift = c_min - pc_min * scale
        coords_scaled = coords * scale + shift

        return coords_scaled
    
    def plot_rescaled(self, cloud, coords_scaled):
        output_dir = "rescaled"
        os.makedirs(output_dir, exist_ok=True)

        cloud_pos = cloud.pos
        coords_scaled = coords_scaled.detach().cpu().tolist()
        cloud_pos = cloud_pos.detach().cpu().tolist()

        cloud_vtk_path = os.path.join(output_dir, f"{cloud.name}_cloud.vtk")
        cloud_cells = np.arange(len(cloud_pos)).reshape(-1, 1)
        cloud_mesh = meshio.Mesh(points=cloud_pos, cells={"vertex": cloud_cells})
        meshio.write(cloud_vtk_path, cloud_mesh)

        coords_vtk_path = os.path.join(output_dir, f"{cloud.name}_coords.vtk")
        coords_cells = np.arange(len(coords_scaled)).reshape(-1, 1)
        coords_mesh = meshio.Mesh(points=coords_scaled, cells={"vertex": coords_cells})
        meshio.write(coords_vtk_path, coords_mesh)

        print(f"Rescaled coordinates saved to {coords_vtk_path}")
        print(f"Cloud points saved to {cloud_vtk_path}")


    def add_encoding(self, cloud, feats, coords, K=6):
        cloud_pos = cloud.pos

        edge_index = knn(coords, cloud_pos, k=K)

        F = feats.shape[1]
        cloud_features = torch.zeros((cloud_pos.shape[0], F), dtype=torch.float32)

        for i in range(cloud_pos.shape[0]):
            neighbors = edge_index[1][edge_index[0] == i]
            if neighbors.numel() > 0:
                cloud_features[i] = feats[neighbors].mean(dim=0)
            else:
                cloud_features[i] = feats[i]
        
        return cloud_features    

        
    def get_new_features(self, cloud, index):
        file_path = self.adfile_paths[index]
        feats, coords = self.get_encoding(file_path)
        feats = torch.tensor(feats, dtype=torch.float32)

        coords[:, [1, 2]] = coords[:, [2, 1]]  # Swap Y and Z coordinates
        coords_scaled = self.scale_pos(cloud, coords)
        coords_scaled = torch.tensor(coords_scaled, dtype=torch.float32)
        coords_scaled[:, 2] = -coords_scaled[:, 2]

        pos_cloud = cloud.pos

        cloud_centroid = pos_cloud.mean(dim=0)
        coords_centroid = coords_scaled.mean(dim=0)

        translation_vector = cloud_centroid - coords_centroid
        coords_scaled[:, 2] += translation_vector[2]

        new_features = self.add_encoding(cloud, feats, coords_scaled, K=6)

        return new_features

    def add_new_features(self, cloud, index):
        features = self.get_new_features(cloud, index)
        cloud.x = torch.cat((cloud.x, features), dim=1)
        return cloud
    
    def add_labels(self, cloud, coords, K=6):
        cloud_pos = cloud.pos
        labels = cloud.y

        edge_index = knn(cloud_pos, coords, k=K)

        coords_labels = torch.zeros((coords.shape[0], 1), dtype=torch.float32)

        for i in range(cloud_pos.shape[0]):
            neighbors = edge_index[1][edge_index[0] == i]
            if neighbors.numel() > 0:
                coords_labels[i] = labels[neighbors].mean(dim=0).round().int()
            else:
                coords_labels[i] = labels[i]    

        return coords_labels        

    def get_new_labels(self, cloud, index):
        file_path = self.adfile_paths[index]
        feats, coords = self.get_encoding(file_path)
        feats = torch.tensor(feats, dtype=torch.float32)

        coords[:, [1, 2]] = coords[:, [2, 1]]  # Swap Y and Z coordinates
        coords_scaled = self.scale_pos(cloud, coords)
        coords_scaled = torch.tensor(coords_scaled, dtype=torch.float32)
        coords_scaled[:, 2] = -coords_scaled[:, 2]

        pos_cloud = cloud.pos

        cloud_centroid = pos_cloud.mean(dim=0)
        coords_centroid = coords_scaled.mean(dim=0)

        translation_vector = cloud_centroid - coords_centroid
        coords_scaled[:, 2] += translation_vector[2]

        new_labels = self.add_labels(cloud, coords_scaled, K=6)

        return coords_scaled, feats, new_labels


    def add_new_labels(self, cloud, index):
        coords, feats, new_labels = self.get_new_labels(cloud, index)
        data = Data(
            x=feats, 
            pos=coords,
            y=new_labels,
        )
        return data


    def sample_points(self, data, num_points=512):
        if data.pos.shape[0] > num_points:
            sample_indices = random.sample(range(data.pos.shape[0]), num_points)
            data.x = data.x[sample_indices]
            data.pos = data.pos[sample_indices]
            data.y = data.y[sample_indices]
        return data

    def __getitem__(self, idx):
        adfile_path = self.adfile_paths[idx]

        points = []
        normals = []
        labels = []

        with open(adfile_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue  # Skip lines without label
                x, y, z, nx, ny, nz, label = parts
                points.append([float(x), float(y), float(z)])
                normals.append([float(nx), float(ny), float(nz)])
                if label == "0":
                    labels.append(int(0))
                else:
                    labels.append(int(1))

        points = np.array(points)

        # Scale points to the range [-1, 1]
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        points = 2 * (points - min_vals) / (max_vals - min_vals) - 1
        
        data = Data(
            x=torch.tensor(normals, dtype=torch.float32),
            pos=torch.tensor(points, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.float32),
        )

        if self.processing == "normal":
            data = self.sample_points(data, num_points=512)
        elif self.processing == "add_features":
            data = self.sample_points(data, num_points=512)
            data = self.add_new_features(data, idx)
        elif self.processing == "add_labels":
            data = self.add_new_labels(data, idx)
            data = self.sample_points(data, num_points=512)

        return data
