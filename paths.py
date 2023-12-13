import os

HOME_ROOT = '/share/hel/home/shreyan'
HOME_DATASETS_ROOT = '/share/hel/home/shreyan/datasets'
DATASETS_ROOT = '/share/hel/cp/datasets'
MAIN_RUN_DIR = './runs'

path_cache_fs = os.path.join(HOME_ROOT, 'data_caches')

path_midlevel_root = os.path.join(HOME_DATASETS_ROOT, 'MidlevelFeatures')
path_midlevel_annotations_dir = os.path.join(path_midlevel_root, 'metadata_annotations')
path_midlevel_annotations = os.path.join(path_midlevel_annotations_dir, 'annotations.csv')
path_midlevel_metadata = os.path.join(path_midlevel_annotations_dir, 'metadata.csv')
path_midlevel_metadata_instruments = os.path.join(HOME_DATASETS_ROOT, 'MidlevelFeatures/metadata_annotations/metadata_domains.csv')
path_midlevel_metadata_piano = os.path.join(HOME_DATASETS_ROOT, 'MidlevelFeatures/metadata_annotations/metadata_piano.csv')
path_midlevel_audio_dir = os.path.join(path_midlevel_root, 'audio')


path_maestro_audio = os.path.join(DATASETS_ROOT, 'maestro-v2.0.0/audio_and_midi')
path_maestro_audio_15sec = os.path.join(DATASETS_ROOT, 'maestro-v2.0.0/audio_15sec')