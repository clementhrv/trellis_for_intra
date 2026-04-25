#!/bin/bash

JSON_FILE="configs/training/segmentation/pn2_feats_benchmark.json"
project_name=$(basename "$JSON_FILE" .json)
project_folder="trellis_5fold"

# Exécuter le script avec les paramètres appropriés
python -m core.segmentation_latents.train \
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
