import getpass
import os
import re

rk = re.compile("rechenknecht[0-8].*")

hostname = os.uname()[1]
username = getpass.getuser()

use_local = False

if bool(rk.match(hostname)):
    local_datasets_root = '/home/shreyan/shared/datasets/'
    local_home_root = '/home/shreyan/'
    local_run_dir = '/home/shreyan/RUNS/'

    fs_datasets_root = '/share/cp/datasets/'
    fs_home_root = '/share/home/shreyan/'
    fs_run_dir = '/share/home/shreyan/RUNS/'

    fs_ml_explanations_project_root = '/share/cp/projects/midlevel_explanations'

elif hostname == 'shreyan-All-Series':
    local_datasets_root = '/mnt/2tb/datasets/'
    local_home_root = ''
    local_run_dir = ''

    fs_datasets_root = '/home/shreyan/mounts/datasets@fs/'
    fs_home_root = '/home/shreyan/mounts/home@fs/'
    fs_run_dir = '/home/shreyan/mounts/home@fs/RUNS/'

    fs_ml_explanations_project_root = '/home/shreyan/mounts/cp/projects/midlevel_explanations'

elif hostname == 'shreyan-HP-EliteBook-840-G5':
    local_datasets_root = '/home/shreyan/mounts/pc/datasets/'
    local_home_root = ''
    local_run_dir = ''

    fs_datasets_root = '/home/shreyan/mounts/datasets@fs/'
    fs_home_root = '/home/shreyan/mounts/home@fs/'
    fs_run_dir = '/home/shreyan/mounts/home@fs/RUNS/'

    fs_ml_explanations_project_root = '/home/shreyan/mounts/cp/projects/midlevel_explanations'
else:
    local_datasets_root = '/home/shreyan/mounts/pc/datasets/'
    local_home_root = ''
    local_run_dir = ''

    fs_datasets_root = '/home/shreyan/mounts/fs/datasets@fs/'
    fs_home_root = ''
    fs_run_dir = '/home/shreyan/mounts/fs/home@fs/RUNS/'

if use_local:
    MAIN_RUN_DIR = local_run_dir
    DATASETS_ROOT = local_datasets_root
    HOME_ROOT = local_home_root
else:
    MAIN_RUN_DIR = fs_run_dir
    DATASETS_ROOT = fs_datasets_root
    HOME_ROOT = fs_home_root

path_cache_fs = os.path.join(HOME_ROOT, 'data_caches')

path_data_out = os.path.join(HOME_ROOT, 'data_output')

path_midlevel_root = os.path.join(DATASETS_ROOT, 'MidlevelFeatures')
path_midlevel_annotations_dir = os.path.join(path_midlevel_root, 'metadata_annotations')
path_midlevel_annotations = os.path.join(path_midlevel_annotations_dir, 'annotations.csv')
path_midlevel_metadata = os.path.join(path_midlevel_annotations_dir, 'metadata.csv')
path_midlevel_metadata_piano = os.path.join(path_midlevel_annotations_dir, 'metadata_piano.csv')
path_midlevel_metadata_instruments = os.path.join(path_midlevel_annotations_dir, 'metadata_domains.csv')
path_midlevel_audio_dir = os.path.join(path_midlevel_root, 'audio')
path_midlevel_compressed_audio_dir = os.path.join(path_midlevel_root, 'audio_compressed/audio')

path_soundtracks_root = os.path.join(DATASETS_ROOT, 'Soundtracks')
path_soundtracks_ratings = os.path.join(path_soundtracks_root, 'set1/set1/mean_ratings_set1_.csv')
path_soundtracks_audio_dir = os.path.join(path_soundtracks_root, 'set1/set1/mp3/Soundtrack360_mp3')

path_pmemo_root = os.path.join(DATASETS_ROOT, 'pmemo')
path_pmemo_metadata = os.path.join(path_pmemo_root, 'metadata.csv')
path_pmemo_audio_dir = os.path.join(path_pmemo_root, 'chorus')
path_pmemo_annotations_dir = os.path.join(path_pmemo_root, 'annotations')
path_pmemo_static_annotations = os.path.join(path_pmemo_annotations_dir, 'static_annotations.csv')

path_emotify_root = os.path.join(DATASETS_ROOT, 'emotify')
path_emotify_annotations = os.path.join(path_emotify_root, 'data.csv')
path_emotify_audio_dir = os.path.join(path_emotify_root, 'audio')
path_emotify_metadata = os.path.join(path_emotify_root, 'meta.txt')

path_deam_root = os.path.join(DATASETS_ROOT, 'deam')
path_deam_audio_dir = os.path.join(path_deam_root, 'DEAM_audio/MEMD_audio')
path_deam_metadata_dir = os.path.join(path_deam_root, 'metadata')
path_deam_annotations_static_1_2000 = os.path.join(path_deam_root, 'DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv')
path_deam_annotations_static_2001_2058 = os.path.join(path_deam_root, 'DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2001_2058.csv')

path_maestro_audio = os.path.join(DATASETS_ROOT, 'maestro-v2.0.0/audio_and_midi')
path_maestro_audio_15sec = os.path.join(DATASETS_ROOT, 'maestro-v2.0.0/audio_15sec')
path_pianoset_audio = os.path.join(DATASETS_ROOT, 'pianoset_shreyan/audio')

path_ce_audio_15sec = os.path.join(DATASETS_ROOT, 'con_espressione_game/data_audio_15sec')
path_ce_audio = os.path.join(DATASETS_ROOT, 'con_espressione_game/audio')
path_ce_metadata = os.path.join(DATASETS_ROOT, 'con_espressione_game/metadata.csv')
path_ce_root = os.path.join(DATASETS_ROOT, 'con_espressione_game')

path_ce_public_root = os.path.join(DATASETS_ROOT, 'con_espressione_game_dataset_(public)')

path_mtgjamendo_audio = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/audio')
path_mtgjamendo_raw_30s_labels = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/MTG-Jamendo_annotations/raw_30s.tsv')
path_mtgjamendo_raw_30s_labels_processed = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/MTG-Jamendo_annotations/raw_30s_processed.tsv')
path_mtgjamendo_raw_30s_labels_50artists_processed = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/raw_30s_cleantags_50artists_processed.tsv')
path_mtgjamendo_raw_30s_labels_50artists = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/raw_30s_cleantags_50artists.tsv')
path_mtgjamendo_mood_labels_split ={
    "train": os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/train_processed.tsv'),
    "validation": os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/validation_processed.tsv'),
    "test": os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/test_processed.tsv')
}

def path_sota_trained_models(dataset, modelname):
    assert dataset in ['jamendo', 'msd', 'mtat'], print(f'dataset should be in ["jamendo", "msd", "mtat"], is {dataset}')
    assert modelname in ['musicnn'], print(f'modelname should be in ["musicnn"], is {modelname}')
    return os.path.join(HOME_ROOT, 'PROJECTS/sota-music-tagging-models/models', f'{dataset}/{modelname}/best_model.pth')