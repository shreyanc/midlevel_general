import os

HOME_ROOT = '/share/home/shreyan'
DATASETS_ROOT = '/share/home/shreyan/datasets'
MAIN_RUN_DIR = './runs'

path_cache_fs = os.path.join(HOME_ROOT, 'data_caches')

path_midlevel_root = os.path.join(DATASETS_ROOT, 'MidlevelFeatures')
path_midlevel_annotations_dir = os.path.join(path_midlevel_root, 'metadata_annotations')
path_midlevel_annotations = os.path.join(path_midlevel_annotations_dir, 'annotations.csv')
path_midlevel_metadata = os.path.join(path_midlevel_annotations_dir, 'metadata.csv')
path_midlevel_audio_dir = os.path.join(path_midlevel_root, 'audio')
