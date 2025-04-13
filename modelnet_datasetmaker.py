import os
from torch_geometric.datasets import ModelNet
from torch_geometric.data import Data
import torch
from tqdm import tqdm

def save_mesh_as_obj(mesh_data: Data, filepath: str):
    """
    Save the mesh data as an .obj file.

    Parameters:
    mesh_data (Data): The mesh data object containing 'pos' and 'face'.
    filepath (str): The path to save the .obj file.
    """
    vertices = mesh_data.pos
    faces = mesh_data.face

    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # Write faces
        for face in faces.t():  # Transpose to get faces in shape (3, num_faces)
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def save_modelnet_as_obj(root: str, name: str = '40', force_reload: bool = False):
    """
    Save the ModelNet dataset as .obj files.

    Parameters:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset ("10" for ModelNet10, "40" for ModelNet40).
    force_reload (bool): Whether to re-process the dataset.
    """
    # Create directories
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Load training dataset
    train_dataset = ModelNet(root=root, name=name, train=True, force_reload=force_reload)
    for i, data in enumerate(tqdm(train_dataset, desc="Saving train data")):
        class_dir = os.path.join(train_dir, data.y.item())
        os.makedirs(class_dir, exist_ok=True)
        filepath = os.path.join(class_dir, f"train_{i}.obj")
        save_mesh_as_obj(data, filepath)

    # Load test dataset
    test_dataset = ModelNet(root=root, name=name, train=False, force_reload=force_reload)
    for i, data in enumerate(tqdm(test_dataset, desc="Saving test data")):
        class_dir = os.path.join(test_dir, data.y.item())
        os.makedirs(class_dir, exist_ok=True)
        filepath = os.path.join(class_dir, f"test_{i}.obj")
        save_mesh_as_obj(data, filepath)

# Set the root directory
root_directory = 'modelnet_dataset'

# Save the ModelNet40 dataset
save_modelnet_as_obj(root=root_directory, name='40')
