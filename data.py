import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from dataset_base import DatasetBase
from dataset_utils import slice_func, normalize_spec, get_dataset_stats, slice_func_simple
from audio import MadmomAudioProcessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from helpers.specaugment import SpecAugment
from utils import *
from paths import *
import logging

logger = logging.getLogger()


def load_midlevel_exclude_piano():
    meta = pd.read_csv(path_midlevel_metadata, sep=';')
    meta_piano = pd.read_csv(path_midlevel_metadata_piano, sep=',')
    no_piano = list(set(meta['song id'].values) - set(meta_piano['song id'].values))
    return np.array(no_piano), meta_piano['song id'].values


def load_midlevel_domain_split(source):
    if source == 'piano':
        return load_midlevel_exclude_piano()
    meta = pd.read_csv(path_midlevel_metadata, sep=';')
    assert source in meta['Source'].unique(), print(f"Unknown source {source}. Available: {meta['Source'].unique()}")

    sc_ids = meta[meta['Source'] == source]['song id']
    not_sc_ids = meta[~meta['song id'].isin(sc_ids)]['song id']

    assert len(sc_ids) > 0 and len(not_sc_ids) > 0, print(f"Can't split with {source}")
    return not_sc_ids.values, sc_ids.values


def load_midlevel_extra(instrument):
    meta = pd.read_csv(path_midlevel_metadata, sep=';')
    meta_solo_instruments = pd.read_csv(path_midlevel_metadata_instruments, sep=',')
    ids_selected = meta_solo_instruments[meta_solo_instruments['Domain'] == instrument]['song id'].values
    ids_not_selected = list(set(meta['song id'].values) - set(ids_selected))
    return ids_not_selected, ids_selected


def load_midlevel_aljanaki(ids=None, exclude=None, seed=None, tsize=0.08):
    rand_state = check_random_state(seed)
    meta = pd.read_csv(path_midlevel_metadata, sep=';')
    annotations = pd.read_csv(path_midlevel_annotations)
    if ids is not None:
        meta = meta[meta['song id'].isin(ids)]
        annotations = annotations[annotations['song_id'].isin(ids)]

    assert meta['song id'].equals(annotations['song_id']), "Song IDs in metadata file does not equal those in annotations file."

    artists = meta['Artist']
    if exclude is not None:
        meta = meta.drop(meta[meta.Source == exclude].index)
    if isinstance(tsize, int):
        test_set_size = tsize
    else:
        test_set_size = int(tsize * len(meta))
    artist_value_counts = artists.value_counts()
    single_artists = artist_value_counts.index[artist_value_counts == 1]
    assert len(single_artists) >= test_set_size, "Single artist test set size is greater than number of single artists in dataset."

    single_artists = single_artists.sort_values()
    selected_artists = rand_state.choice(single_artists, test_set_size, replace=False)
    selected_tracks_for_test = meta[meta['Artist'].isin(selected_artists)]

    test_song_ids = selected_tracks_for_test['song id'].values
    train_song_ids = annotations[~annotations['song_id'].isin(test_song_ids)]['song_id'].values

    return train_song_ids, test_song_ids


aljanaki_split = load_midlevel_aljanaki


def get_label_medians():
    annotations = pd.read_csv(path_midlevel_annotations)
    return annotations.median().values[1:]


def temporal_mixup(x1, t1, x2, t2, slice_len):
    x1 = slice_func_simple(x1, slice_len)
    x2 = slice_func_simple(x2, slice_len)

    joined = np.append(x1, x2, axis=-1)
    st = np.random.randint(0, slice_len)
    et = st + slice_len
    sliced = joined[:, st:et]
    interpolated_targets = ((slice_len - st)*t1 + (et - slice_len)*t2)/slice_len

    return sliced, interpolated_targets


def get_extra_dataset(dset_name, seed=None, augment=None, test_size=0.1, labeled=False, normalizing_dset=None):
    if seed is None:
        seed = 0

    if dset_name == 'maestro':
        domain_files = list_files_deep(path_maestro_audio_15sec, filter_ext=['.wav', '.WAV', '.mp3'], full_paths=True)
        tg_train_files, tg_test_files = train_test_split(domain_files, test_size=test_size, random_state=seed)
        tg_tr_dataset = UnlabeledAudioDataset(name='maestro', audio_files=tg_train_files, augment=augment, slice_mode='start', normalizing_dset=normalizing_dset)
        tg_te_dataset = UnlabeledAudioDataset(name='maestro', audio_files=tg_test_files, augment=None, slice_mode='start', normalizing_dset=normalizing_dset)
        return tg_tr_dataset, tg_te_dataset



class MidlevelDataset(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(name='midlevel', annotations=path_midlevel_annotations, **kwargs)

    def __getitem__(self, ind):
        song_id = self.song_id_list[ind]
        audio_path = self._get_audio_path_from_id(song_id)
        x = self._get_spectrogram(audio_path, self.dataset_cache_dir)
        labels = self._get_labels(song_id)
        slice_length = self.processor.times_to_frames(self.duration)
        x_sliced, _, _ = slice_func(x, slice_length, self.processor, mode=self.slice_mode, padding=self.padding)
        return audio_path, torch.from_numpy(x_sliced), labels

    def _load_annotations(self, annotations):
        ann = pd.read_csv(annotations)
        ann.columns = ann.columns.str.replace(' ', '')
        return ann

    def _get_label_names_from_annotations_df(self):
        return self.annotations.columns[1:]

    def _get_dataset_label_stats(self):
        return self.annotations.agg(['mean', 'std', 'min', 'max'])

    def _get_id_col(self):
        return self.annotations.columns[0]

    def _get_audio_path_from_id(self, songid):
        return os.path.join(path_midlevel_audio_dir, str(songid) + '.mp3')



class UnlabeledAudioDataset(DatasetBase):
    def __init__(self, audio_files, **kwargs):
        
        super().__init__(cache_dir=path_cache_fs,
                         name='audio_dataset', 
                         duration=15,
                         annotations=path_midlevel_annotations, 
                         **kwargs)
        
        self.audio_files = audio_files

    def __getitem__(self, ind):
        audio_path = self.audio_files[ind]
        x = self._get_spectrogram(audio_path, self.dataset_cache_dir)
        slice_length = self.processor.times_to_frames(self.duration)
        x_sliced, _, _ = slice_func(x, slice_length, self.processor, mode=self.slice_mode, padding=self.padding)
        return audio_path, torch.from_numpy(x_sliced)

    def __len__(self):
        return len(self.audio_files)
