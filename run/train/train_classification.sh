#!/bin/bash

JSON_FILE="configs/training/classification/pn2_1024feats_class5f.json"
project_name=$(basename "$JSON_FILE" .json)
project_folder="trellis_5fold_classification"

# Exécuter le script avec les paramètres appropriés
python -m core.classification_latents.train \
    --training_parameters_path="$JSON_FILE" \
    --project_name="$project_name" \
    --project_folder="$project_folder" \
    --num_epochs=20 \
    --init_lr=0.001 \
    --batch_size=16 \
    --warmup=100 \
    --num_workers=0 \
    --prefetch_factor=0 \
    --model_save_path="${project_folder}/model_run_1.ckpt"
