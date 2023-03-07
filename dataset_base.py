from madmom.audio import SignalProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from dataset_utils import slice_func, normalize_spec, get_dataset_stats, slice_func_waveform
from audio import MadmomAudioProcessor
import pandas as pd
import numpy as np

from utils import *
from paths import *
import logging

logger = logging.getLogger()


class DatasetBase(Dataset):
    def __init__(self, cache_dir=path_cache_fs,
                 name='',
                 duration=15,
                 select_song_ids=None,
                 aud_processor=None,
                 annotations=None,
                 normalize_labels=False,
                 scale_labels=True,
                 normalize_inputs=True,
                 normalizing_dset_name=None,
                 padding='loop',
                 return_labels=True,
                 return_waveform=False,
                 **kwargs):

        if aud_processor is not None:
            self.processor = aud_processor
        else:
            self.processor = MadmomAudioProcessor(fps=31.3)

        self.verbose = kwargs.get('verbose', True)
        self.dataset_cache_dir = os.path.join(cache_dir, name)
        self.dset_name = name
        self.duration = duration  # seconds
        assert padding in ['zero', 'loop'], print(f"Padding should be in ['zero', 'loop'], is {padding}")
        self.padding = padding  # in slice_func, determines how to handle spectrogram slice when requested length is longer than spectrogram length

        if return_labels:
            try:
                self.annotations = self._load_annotations(annotations)
            except Exception as e:
                print("Exception occured while parsing annotations: ", e)
                assert isinstance(annotations, pd.DataFrame), print("annotations should be a pd.DataFrame object")
                self.annotations = annotations

            self.label_names = self._get_label_names_from_annotations_df()
            self.id_col = self._get_id_col()

            if select_song_ids is not None:
                selected_set = self.annotations[self.annotations[self.id_col].isin([int(i) for i in select_song_ids])]
            else:
                selected_set = self.annotations
            self.song_id_list = selected_set[self.id_col].tolist()

            self.normalize_labels = normalize_labels
            self.scale_labels = scale_labels

            self.label_stats = self._get_dataset_label_stats()

        self.return_labels = return_labels

        if kwargs.get('slice_mode') is None:
            self.slice_mode = 'random'
        else:
            assert kwargs['slice_mode'] in ['random', 'start', 'end', 'middle']
            self.slice_mode = kwargs['slice_mode']

        self.specs_dir = os.path.join(path_cache_fs, self.dset_name, self.processor.get_params.get("name"))
        self.stats_dir = os.path.join(path_cache_fs, self.dset_name, self.processor.get_params.get("name") + '_stats')

        if normalizing_dset_name is None:
            self.normalizing_dset_name = self.dset_name
        else:
            self.normalizing_dset_name = normalizing_dset_name

        if normalize_inputs is None or normalize_inputs is False:
            self.normalize_inputs = False
            logger.info("Input normalization OFF")
        else:
            # Valid normalize_inputs: str: "single"/"spec", or boolean True/False
            self.normalize_inputs = True
            # Default normalization method is using single mean pixel value.
            self.normalize_method = normalize_inputs if isinstance(normalize_inputs, str) else 'single'
            logger.info(f"Normalizing inputs using method {self.normalize_method}")

        self.return_waveform = return_waveform

    def __getitem__(self, ind):
        audio_path = self._get_audio_path_from_idx(ind)
        song_id = self._get_song_id_from_idx(ind)

        if not self.return_waveform:
            x = self._get_spectrogram(audio_path, self.dataset_cache_dir)
            if self.normalize_inputs:
                x = normalize_spec(x, dset_name=self.normalizing_dset_name, aud_processor=self.processor, method=self.normalize_method)
            slice_length = self.processor.times_to_frames(self.duration)
            x_sliced, start_time, end_time = slice_func(x, slice_length, self.processor, mode=self.slice_mode, padding=self.padding)
        else:
            x = self._get_waveform(audio_path)
            x_sliced, start_time, end_time = slice_func_waveform(x, length_seconds=self.duration,
                                                                 sr=22050, mode=self.slice_mode, padding=self.padding)

        x_ret = x_sliced
        if self.return_labels:
            labels = self._get_labels(song_id)
            return audio_path, torch.from_numpy(x_ret), labels
        else:
            return audio_path, torch.from_numpy(x_ret)

    def _get_waveform(self, audio_path):
        audio_reader = SignalProcessor(num_channels=1, sample_rate=22050, norm=True)
        return audio_reader(audio_path)

    def _get_spectrogram(self, audio_path, dataset_cache_dir):
        specpath = os.path.join(dataset_cache_dir, self.processor.get_params.get("name"), str(os.path.basename(audio_path).split('.')[0]))
        specdir = os.path.split(specpath)[0]
        if not os.path.exists(specdir):
            os.makedirs(specdir)
        try:
            return np.load(specpath + '.npy')
        # except FileNotFoundError:
        #     return pickleload(specpath + '.specobj').spec  # for backward compatibility
        except Exception as e:
            if self.verbose:
                print(f"Could not load {specpath}")
                print(f"Calculating spectrogram for {audio_path} and saving to {specpath+'.npy'}")
            spec_obj = self.processor(audio_path)
            np.save(specpath+'.npy', spec_obj.spec)
            return spec_obj.spec

    def __len__(self):
        return len(self.song_id_list)

    def _load_annotations(self, annotations):
        pass

    def _get_label_names_from_annotations_df(self):
        pass

    def _get_dataset_label_stats(self):
        pass

    def _get_id_col(self):
        pass

    def _get_audio_path_from_idx(self, ind):
        raise NotImplementedError

    def _get_song_id_from_idx(self, ind):
        return ind

    def _get_labels(self, song_id):
        # labels = self.annotations.loc[self.annotations[self.id_col] == song_id][self.label_names]
        ann = self.get_labels_df()
        labels = ann.loc[ann[self.id_col] == song_id][self.label_names]

        # if self.normalize_labels:
        #     labels -= self.label_stats.loc['mean'][self.label_names].values
        #     labels /= self.label_stats.loc['std'][self.label_names].values
        #
        # if self.scale_labels:
        #     labels -= self.label_stats.loc['min'][self.label_names].values
        #     labels /= self.label_stats.loc['max'][self.label_names].values - self.label_stats.loc['min'][self.label_names].values
        #     labels *= 2
        #     labels -= 1

        return torch.from_numpy(labels.values).squeeze()

    def get_labels_df(self):
        ann = self.annotations.copy()
        if self.normalize_labels:
            ann[self.label_names] -= self.label_stats.loc['mean'][self.label_names].values
            ann[self.label_names] /= self.label_stats.loc['std'][self.label_names].values

        if self.scale_labels:
            ann[self.label_names] -= self.label_stats.loc['min'][self.label_names].values
            ann[self.label_names] /= self.label_stats.loc['max'][self.label_names].values - self.label_stats.loc['min'][self.label_names].values
            ann[self.label_names] *= 2
            ann[self.label_names] -= 1

        return ann
