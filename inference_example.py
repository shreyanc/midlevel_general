# Import necessary libraries
from tqdm import tqdm
from inference import InferenceMidlevelModel, default_config
import numpy as np
import os
import utils
from omegaconf import OmegaConf

# Create configuration object
config = OmegaConf.create(default_config)

# Create an instance of the InferenceMidlevelModel
predictor = InferenceMidlevelModel(config=config)

# Define the directory name for saving midlevel files
midlevels_dirname = 'midlevel_2bf74_2'

# Get a list of audio file paths
audio_paths = utils.list_files_deep('path/to/audio/dir', full_paths=True, filter_ext=['.wav'])

# Iterate over each audio file path
for audio_fp in tqdm(audio_paths):

    # Get the audio file name without extension
    audio_fn = os.path.basename(audio_fp).split('.')[0]

    # Get the directory path of the audio file
    curdir = os.path.dirname(audio_fp)

    # Define the directory path for saving midlevel files
    savedir = 'path/to/save/midlevels'
    os.makedirs(savedir, exist_ok=True)

    # Define the file path for saving the midlevel file
    save_fp = os.path.join(savedir, audio_fn+'.csv')

    # Skip if the midlevel file already exists or the audio file does not exist
    if os.path.exists(save_fp) or not os.path.exists(audio_fp):
        continue

    try:
        # Predict midlevels and save to file
        mls = predictor.predict_file(audio_fp, 
                                     return_what='midlevels', 
                                     output_aggregate=None,
                                     cache_dir_dataset='audio_dir_midlevels', 
                                     cache_file_name=audio_fn,
                                     saveto=save_fp
                                     )
    except Exception as e:
        # Print error message if prediction fails
        print(f"Could not compute {audio_fp}: {e}")