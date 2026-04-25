set -e 
set -x 

export DATASET_NAME=MedMNISTv2
export DATASET_SOURCE=MedMNISTv2
export OUTPUT_DIR=data/artifacts/dataset_medmnist
export ROOT_FOLDER=obj_export_medmnist
export RANK=0
export WORLD_SIZE=1600
export MAX_WORKERS=6

# initially build meta data of the specified dataset
python core/dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}
echo "Init building metadata done."

# download dataset
python core/dataset_toolkits/download.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --world_size ${WORLD_SIZE}
python core/dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}
echo "Dataset Downloaded."

# render multi-view images with blender (e.g. 150 views)
python core/dataset_toolkits/render.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --max_workers ${MAX_WORKERS}
python core/dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

# voxelization (based on open3d)
python core/dataset_toolkits/voxelize.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR}
python core/dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

# extract DINOv2 features
python core/dataset_toolkits/extract_feature.py --output_dir ${OUTPUT_DIR}
# update meta data
python core/dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

# extract SparseStructure Latents
python core/dataset_toolkits/encode_ss_latent.py --output_dir ${OUTPUT_DIR}
# update meta data
python core/dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

python core/dataset_toolkits/encode_latent.py --output_dir ${OUTPUT_DIR}
python core/dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

