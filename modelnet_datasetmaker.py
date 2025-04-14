import os
import urllib.request
import zipfile
from tqdm import tqdm



def download_modelnet(dataset_name='ModelNet40', root='datasets'):
    """
    Télécharge et dézippe le dataset ModelNet.
    """
    os.makedirs(root, exist_ok=True)
    url = f"http://modelnet.cs.princeton.edu/{dataset_name}.zip"
    zip_path = os.path.join(root, f"{dataset_name}.zip")
    extract_path = os.path.join(root, dataset_name)

    if not os.path.exists(extract_path):
        print(f"Téléchargement de {dataset_name}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Décompression...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        print("Suppression de l'archive...")
        os.remove(zip_path)
    else:
        print(f"{dataset_name} déjà présent dans {root}")

    return extract_path

def read_off(file_path):
    """
    Lecture robuste d’un fichier OFF : retourne sommets et faces.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Enlever les lignes vides
    lines = [line.strip() for line in lines if line.strip()]

    if lines[0] != 'OFF':
        if lines[0].startswith('OFF'):
            # Cas où tout est collé sur la 1ère ligne : OFF 123 456 0
            parts = lines[0].replace('OFF', '').strip().split()
            if len(parts) >= 2:
                counts = list(map(int, parts[:2]))
                vertex_lines = lines[1:1 + counts[0]]
                face_lines = lines[1 + counts[0]:1 + counts[0] + counts[1]]
            else:
                raise ValueError(f"Header invalide dans le fichier {file_path}")
        else:
            raise ValueError(f"Fichier OFF invalide : {file_path}")
    else:
        # Cas normal : OFF + dimensions sur la 2ème ligne
        counts = list(map(int, lines[1].split()))
        vertex_lines = lines[2:2 + counts[0]]
        face_lines = lines[2 + counts[0]:2 + counts[0] + counts[1]]

    vertices = [list(map(float, v.split())) for v in vertex_lines]
    faces = []

    for face in face_lines:
        parts = list(map(int, face.split()))
        if parts[0] == 3:
            faces.append(parts[1:4])

    return vertices, faces

def write_obj(vertices, faces, output_path):
    """
    Écriture des données dans un fichier .obj.
    """
    with open(output_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def convert_off_to_obj(modelnet_path, output_path, sampling_ratio=6, part=0):
    """
    Conversion de fichiers .off en .obj, en ne prenant qu'1 fichier sur `sampling_ratio`.
    """
    class_names = [d for d in os.listdir(modelnet_path) if os.path.isdir(os.path.join(modelnet_path, d))]

    for class_name in tqdm(class_names, desc="Conversion des classes"):
        class_path = os.path.join(modelnet_path, class_name)

        for split in ['train', 'test']:
            split_path = os.path.join(class_path, split)
            if not os.path.isdir(split_path):
                continue

            output_class_dir = os.path.join(output_path, split, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            off_files = [f for f in os.listdir(split_path) if f.endswith('.off')]
            off_files.sort()  # pour garder un ordre stable

            for idx, filename in enumerate(off_files):
                if idx % sampling_ratio != part:
                    continue  # On garde 1 fichier sur `sampling_ratio`

                off_path = os.path.join(split_path, filename)
                obj_filename = filename.replace('.off', '.obj')
                obj_path = os.path.join(output_class_dir, obj_filename)

                vertices, faces = read_off(off_path)
                write_obj(vertices, faces, obj_path)

def main():
    dataset_name = 'ModelNet40'  # ou 'ModelNet10'
    root_dir = 'datasets'
    output_dir = 'ModelNet_OBJ'
    SAMPLING_RATIO  = 6  # 1 fichier sur 6
    PART_OF_DATASET = 0 # number to change between 0 and 5 if SAMPLING_RATIO = 6 to select the part of the dataset to download

    modelnet_path = download_modelnet(dataset_name, root=root_dir)
    convert_off_to_obj(modelnet_path, output_path=output_dir, sampling_ratio=SAMPLING_RATIO, part =PART_OF_DATASET)
    print(f"\nConversion terminée ! 1 fichier sur {SAMPLING_RATIO} a été converti dans : {output_dir}")

if __name__ == '__main__':
    main()
