import os
import numpy as np

# Parcourt tous les fichiers du dossier courant
for filename in os.listdir('dataset_classification_intra_5fold'):
    for  subdir, dirs, files in os.walk('dataset_classification_intra_5fold'):
        for file in files:
            if file.endswith('.npz'):
                filename = os.path.join(subdir, file)
                # Charger les données du fichier .npz
                data = np.load(filename)

                # Supposons que les points sont stockés dans un tableau nommé 'coords'
                if 'coords' in data:
                    coords = data['coords']
                    if 'all_lens' not in locals():
                        all_lens = []
                    all_lens.append(len(coords))
                else:
                    print(f"{filename} : clé 'coords' non trouvée")

# Après la boucle, afficher la moyenne, le max et le min de len(coords)
if 'all_lens' in locals() and all_lens:
    mean_len = np.mean(all_lens)
    max_len = np.max(all_lens)
    min_len = np.min(all_lens)
    print(f"len(coords) : mean={mean_len}, max={max_len}, min={min_len}")
else:
    print("Aucune donnée 'coords' trouvée dans les fichiers.")
