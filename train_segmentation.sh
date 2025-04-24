#!/bin/bash

# Chemin vers le dossier contenant les fichiers JSON
JSON_DIR="segmentation_latents/settings"

# Boucle pour exécuter le script deux fois pour chaque fichier JSON
for i in {1..2}; do
    for json_file in "$JSON_DIR"/*.json; do
        # Extraire le nom du fichier sans l'extension
        base_name=$(basename "$json_file" .json)

        # Construire les noms de projet et de dossier
        project_name="${base_name}_b2_do_run_${i}"
        project_folder="trellis_test_2"

        # Exécuter le script avec les paramètres appropriés
        python -m segmentation_latents.train \
            --training_parameters_path="$json_file" \
            --project_name="$project_name" \
            --project_folder="$project_folder" \
            --num_epochs=60 \
            --init_lr=0.01 \
            --batch_size=2 \
            --warmup=100 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_path="${project_folder}/model_run_${i}.ckpt"
    done
done