import numpy as np
import pandas as pd
import shutil
import os
import sys

def get_metadata(root_dir):
    file_path = os.path.join(root_dir, 'metadata.csv')
    metadata = pd.read_csv(file_path)
    
    return metadata


def move_latents(metadata, root_dir):
    for index, row in metadata.iterrows():
        sha256 = row['sha256']
        # Create the directory structure
        if row['split'] == 'training':
            train_val_dir = 'training'
        elif row['split'] == 'validation':
            train_val_dir = 'validation'
        else:
            train_val_dir = 'test'
        aneurysm_vessel_dir = 'aneurysm' if row['class'] == 'aneurysm' else 'vessel'
        latent_dir = os.path.join(root_dir, 'latents_sorted', train_val_dir, aneurysm_vessel_dir)
        
        if not os.path.exists(latent_dir):
            os.makedirs(latent_dir)
        
        # Move the latent file
        src_file = os.path.join(root_dir, 'latents', 'dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16', f'{sha256}.npz')
        dst_file = os.path.join(latent_dir, f'{sha256}.npz')
        
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
    


def move_ss_latents(metadata, root_dir):
    for index, row in metadata.iterrows():
        sha256 = row['sha256']
        # Create the directory structure
        if row['split'] == 'training':
            train_val_dir = 'training'
        elif row['split'] == 'validation':
            train_val_dir = 'validation'
        else:
            train_val_dir = 'test'        
        aneurysm_vessel_dir = 'aneurysm' if row['class'] == 'aneurysm' else 'vessel'
        latent_dir = os.path.join(root_dir, 'ss_latents_sorted', train_val_dir, aneurysm_vessel_dir)
        
        if not os.path.exists(latent_dir):
            os.makedirs(latent_dir)
        
        # Move the latent file
        src_file = os.path.join(root_dir, 'ss_latents', 'ss_enc_conv3d_16l8_fp16', f'{sha256}.npz')
        dst_file = os.path.join(latent_dir, f'{sha256}.npz')
        
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
    


if __name__ == '__main__':
    
    root_dir = sys.argv[1]
    
    metadata = get_metadata(root_dir)
    move_latents(metadata, root_dir)
    move_ss_latents(metadata, root_dir)