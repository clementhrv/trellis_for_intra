import os
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

def download_and_extract(url, extract_to='.'):
    """
    Télécharge et extrait un fichier depuis une URL.

    Parameters:
    url (str): L'URL du fichier à télécharger.
    extract_to (str): Le répertoire où extraire les fichiers.
    """
    filename = os.path.join(extract_to, url.split('/')[-1])
    urllib.request.urlretrieve(url, filename)

    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif filename.endswith('.tar.gz'):
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)

    os.remove(filename)

def parse_off_file(filepath):
    """
    Parse un fichier .off et retourne les sommets et les faces.

    Parameters:
    filepath (str): Le chemin vers le fichier .off.

    Returns:
    vertices (list of tuples): Liste des sommets.
    faces (list of tuples): Liste des faces.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    if lines[0].strip() != 'OFF':
        raise ValueError("Le fichier n'est pas au format OFF.")

    parts = lines[1].strip().split()
    num_vertices = int(parts[0])
    num_faces = int(parts[1])

    vertices = []
    faces = []

    for line in lines[2:2 + num_vertices]:
        parts = line.strip().split()
        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))

    for line in lines[2 + num_vertices:2 + num_vertices + num_faces]:
        parts = line.strip().split()
        faces.append((int(parts[1]), int(parts[2]), int(parts[3])))

    return vertices, faces

def save_as_obj(vertices, faces, filepath):
    """
    Sauvegarde les sommets et les faces dans un fichier .obj.

    Parameters:
    vertices (list of tuples): Liste des sommets.
    faces (list of tuples): Liste des faces.
    filepath (str): Le chemin où sauvegarder le fichier .obj.
    """
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def process_modelnet_dataset(root, name='40', train=True):
    """
    Télécharge, extrait et lit les fichiers .off du jeu de données ModelNet.

    Parameters:
    root (str): Répertoire racine où stocker les fichiers.
    name (str): Nom du jeu de données ('10' pour ModelNet10, '40' pour ModelNet40).
    train (bool): Si True, télécharge le jeu de données d'entraînement, sinon le jeu de données de test.
    """
    os.makedirs(root, exist_ok=True)

    if name == '10':
        url = 'http://modelnet.cs.princeton.edu/ModelNet10.zip'
    elif name == '40':
        url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    else:
        raise ValueError("Le nom du jeu de données doit être '10' ou '40'.")

    download_and_extract(url, extract_to=root)

    dataset_dir = os.path.join(root, f'ModelNet{name}')
    subset_dir = os.path.join(dataset_dir, 'train' if train else 'test')
    obj_dir = os.path.join(root, 'obj_files')
    os.makedirs(obj_dir, exist_ok=True)

    file_count = 0

    for class_dir in tqdm(os.listdir(subset_dir), desc="Processing files"):
        class_path = os.path.join(subset_dir, class_dir)
        for off_file in os.listdir(class_path):
            if off_file.endswith('.off'):
                file_count += 1
                off_file_path = os.path.join(class_path, off_file)
                vertices, faces = parse_off_file(off_file_path)
                print(f"Processed {off_file_path}: {len(vertices)} vertices, {len(faces)} faces")

                # Convertir un fichier sur six en .obj
                if file_count % 6 == 0:
                    obj_file_path = os.path.join(obj_dir, f"{off_file.replace('.off', '.obj')}")
                    save_as_obj(vertices, faces, obj_file_path)
                    print(f"Saved {obj_file_path} as .obj")

# Exemple d'utilisation
root_directory = 'modelnet_dataset'
process_modelnet_dataset(root=root_directory, name='40', train=True)
