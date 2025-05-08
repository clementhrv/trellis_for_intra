import os 
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
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
        processing: list = [1,0,0],
        switch_to_val: bool = False,
        switch_to_test: bool = False, 
        number_of_samples: int = 512,  
        number_of_connections: int = 6,
        indices = None,
        cache_dir: str = "cache",
    ):
        
        if switch_to_val:
            root_folder = root_folder.replace("training", "validation")

        self.npzfile_paths = []
        self.adfile_paths = []
        self.labels = []
        self.processing = processing
        self.number_of_samples = number_of_samples
        self.number_of_connections = number_of_connections
        self.indices = indices

        # Lister tous les fichiers
        all_npz = []
        all_ad = []

        for file in os.listdir(root_folder):
            if file.endswith(".npz"):
                all_npz.append(os.path.join(root_folder, file))
            elif file.endswith(".ad"):
                all_ad.append(os.path.join(root_folder, file))

        # Trier pour assurer le bon alignement
        all_npz.sort()
        all_ad.sort()

        # Si indices spécifiés (KFold), ne garder que ceux-là
        if self.indices is not None:
            self.adfile_paths = [all_ad[i] for i in self.indices]
            self.npzfile_paths = [all_npz[i] for i in self.indices]
        else:
            self.adfile_paths = all_ad
            self.npzfile_paths = all_npz

        with open(meta_path, "r") as fp:
            self.meta = json.loads(fp.read())

        # Initialize caching mechanism
        self.cache_dir = cache_dir
        os.remove(self.cache_dir) if os.path.exists(self.cache_dir) else None
        os.makedirs(self.cache_dir, exist_ok=True)

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
    
    
    def compute_labels_and_normals(self, cloud, coords, K=6):
        cloud_pos = cloud.pos
        cloud_labels = cloud.y
        cloud_normals = cloud.x

        edge_index = knn(cloud_pos, coords, k=K)

        coords_labels = torch.zeros((coords.shape[0]), dtype=torch.float32)
        coords_normals = torch.zeros((coords.shape[0], 3), dtype=torch.float32)

        for i in range(coords.shape[0]):
            neighbors = edge_index[1][edge_index[0] == i]
            if neighbors.numel() > 0:
                coords_labels[i] = cloud_labels[neighbors].mean(dim=0).round().int()
                coords_normals[i] = cloud_normals[neighbors].mean(dim=0)
            else:
                coords_labels[i] = cloud_labels[i]
                coords_normals[i] = cloud_normals[i]

        return coords_labels, coords_normals

    def get_normal_label(self, cloud, index):
        file_path = self.adfile_paths[index]
        feats, coords = self.get_encoding(file_path)
        feats = torch.tensor(feats, dtype=torch.float32)

        coords[:, [1, 2]] = coords[:, [2, 1]]
        coords_scaled = self.scale_pos(cloud, coords)
        coords_scaled = torch.tensor(coords_scaled, dtype=torch.float32)
        coords_scaled[:, 2] = -coords_scaled[:, 2]

        pos_cloud = cloud.pos
        cloud_centroid = pos_cloud.mean(dim=0)
        coords_centroid = coords_scaled.mean(dim=0)

        translation_vector = cloud_centroid - coords_centroid
        coords_scaled[:, 2] += translation_vector[2]

        new_labels, new_normals = self.compute_labels_and_normals(cloud, coords_scaled, K=4)

        return coords_scaled, feats, new_labels, new_normals
    
    def add_normal_label(self, cloud, index):
        cache_file = os.path.join(self.cache_dir, f"feats_{index}.pt")

        if os.path.exists(cache_file):
            try:
                cached_data = torch.load(cache_file)
                feats = cached_data['feats']
                new_labels = cached_data['new_labels']
                new_normals = cached_data['new_normals']
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}. Recomputing features.")
                os.remove(cache_file)
                feats, coords, new_labels, new_normals = self.get_normal_label(cloud, index)
                torch.save({'feats': feats, 'new_labels': new_labels, 'new_normals': new_normals}, cache_file)
        else:
            feats, coords, new_labels, new_normals = self.get_normal_label(cloud, index)
            torch.save({'feats': feats, 'new_labels': new_labels, 'new_normals': new_normals}, cache_file)

        return feats, coords, new_labels, new_normals

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
        
        data = Data(
            x=torch.tensor(normals, dtype=torch.float32),
            pos=torch.tensor(points, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.float32),
        )

        data = T.NormalizeScale()(data)

        coords, feats, new_labels, new_normals = self.add_normal_label(data, idx)

        features = torch.empty((feats.shape[0], 0), dtype=torch.float32)  # Create an empty tensor with 0 features

        if self.processing[0] == 1:
            features = torch.cat((features, coords), dim=1)
        if self.processing[1] == 1:
            features = torch.cat((features, new_normals), dim=1)
        if self.processing[2] == 1:
            features = torch.cat((features, feats), dim=1)

        new_data = Data(
            x=features,
            pos=coords,
            y=new_labels,
        )

        new_data = self.sample_points(new_data, self.number_of_samples)
        new_data = T.NormalizeScale()(new_data)
        new_data = T.KNNGraph(k=self.number_of_connections)(new_data)

        return new_data
