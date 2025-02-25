set -e 
set -x 

export DATASET_NAME=AnxPlore
export DATASET_SOURCE=AnxPlore
export OUTPUT_DIR=dataset_output
export ROOT_FOLDER=surface_dataset
export RANK=0
export WORLD_SIZE=1600
export MAX_WORKERS=6

# initially build meta data of the specified dataset
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}
echo "Init building metadata done."

# download dataset
python dataset_toolkits/download.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --world_size ${WORLD_SIZE}
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}
echo "Dataset Downloaded."

# render multi-view images with blender (e.g. 150 views)
python dataset_toolkits/render.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --max_workers ${MAX_WORKERS}
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

# voxelization (based on open3d)
python dataset_toolkits/voxelize.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR}
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

# extract DINOv2 features
python dataset_toolkits/extract_feature.py --output_dir ${OUTPUT_DIR}
# update meta data
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

# extract SparseStructure Latents
python dataset_toolkits/encode_ss_latent.py --output_dir ${OUTPUT_DIR}
# update meta data
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

python dataset_toolkits/encode_latent.py --output_dir ${OUTPUT_DIR}
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --root_folder ${ROOT_FOLDER}

