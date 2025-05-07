#!/bin/bash

# Chemin vers le fichier JSON spécifique
JSON_FILE="segmentation_latents/segmentation_1024_feats_test5fold.json"

# Extraire le nom du fichier sans l'extension
base_name=$(basename "$JSON_FILE" .json)

# Construire les noms de projet et de dossier
project_name="${base_name}_b2_do_run_1"
project_folder="trellis_5fold"

# Exécuter le script avec les paramètres appropriés
python -m segmentation_latents.train \
    --training_parameters_path="$JSON_FILE" \
    --project_name="$project_name" \
    --project_folder="$project_folder" \
    --num_epochs=200 \
    --init_lr=0.001 \
    --batch_size=8 \
    --warmup=100 \
    --num_workers=0 \
    --prefetch_factor=0 \
    --model_save_path="${project_folder}/model_run_1.ckpt"
