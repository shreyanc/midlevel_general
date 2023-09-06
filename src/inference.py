import itertools
import os
import cProfile
import pstats
import librosa.feature

import madmom
from madmom.audio.spectrogram import Spectrogram

import numpy as np
import pandas as pd
from madmom.features import RNNPianoNoteProcessor
from madmom.io import load_audio_file
from omegaconf import OmegaConf
from tqdm import tqdm

import utils
from models.cpresnet import CPResnet, CPResnet_BackProp

import torch

from datasets.dataset_utils import normalize_spec
from audio import MadmomAudioProcessor
from models.cpresnet import config_cp_field_shallow_m2
from paths import HOME_ROOT, path_ce_personal_audio_all, path_ce_personal_root
from utils import load_model

path_saved_models_root = os.path.join(HOME_ROOT, 'SAVED_MODELS')

default_config = {
    'architecture': 'cpresnet_da',  # Model architecture. Choose from ['cpresnet', 'cpresnet_da']
    'load_from_path': './trained/CPResnet_BackProp_2bf74_2.pt', # pre-trained midlevel model path
    'prediction_window_len': 15,  # Prediction window length in seconds
    'prediction_hop_len': 5,  # Prediction hop length in seconds
    'normalize_spectrogram': False,  # Whether or not to normalize the spectrogram before prediction
    'additional_features': {
        'onset_density': False,
        'rms': False
    },
    'onset_detection_method': 'superflux'
}


class Batchifier:
    def __init__(self, arr, window_len: int, hop_len: int):
        if len(arr.shape) == 1:
            arr = arr[np.newaxis, :]
        self.arr = arr
        self.arr_len = arr.shape[1]
        self.window_len = window_len
        self.hop_len = hop_len

    def batchify(self):
        if self.arr_len < self.window_len:
            self.arr = np.pad(self.arr, ((0, 0), (0, self.window_len - self.arr_len)), mode='wrap')
            self.arr_len = self.window_len

        for ndx in range(0, self.arr_len - self.window_len + 1, self.hop_len):
            yield ndx + self.window_len, self.arr[:, ndx:ndx + self.window_len].squeeze()

    def get_ith_batch(self, get_ith_batch: int = None):
        start_idx = get_ith_batch * self.hop_len
        return self.arr[:, start_idx:min(start_idx + self.window_len, self.arr_len)].squeeze()



class InferenceMidlevelModel:

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_processor = MadmomAudioProcessor(fps=31.3)  # audio processor
        self.ml_names = ['melodiousness', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness']
        # List of names of midlevel features, including additional features
        self.audio_fp = None  # Audio file being processed currently
        self._build_model()
        self.window_len_frames = self.audio_processor.times_to_frames(self.config.prediction_window_len)
        self.hop_len_frames = self.audio_processor.times_to_frames(self.config.prediction_hop_len)

    def _build_model(self):
        if self.config.architecture == 'cpresnet':
            self.model = CPResnet(num_targets=7, config=config_cp_field_shallow_m2)
        elif self.config.architecture == 'cpresnet_da':
            self.model = CPResnet_BackProp(num_targets=7, config=config_cp_field_shallow_m2)
        else:
            raise ValueError("config.architecture should be in ['cpresnet', 'cpresnet_da']")

        load_model(self.config.load_from_path, self.model)
        self.model.to(self.device)

    def _get_model_predictions(self, input, mode='eval'):
        if mode == 'eval':
            self.model.eval()
        output = self.model(input)
        if isinstance(output, dict):
            return output['output'].cpu().detach().numpy().squeeze()
        else:
            return output.cpu().detach().numpy().squeeze()

    def _get_embeddings(self, input, mode='eval'):
        if mode == 'eval':
            self.model.eval()
        output = self.model(input)
        if isinstance(output, dict):
            return output['embedding'].cpu().detach().numpy().squeeze()
        else:
            raise TypeError("Expected model to return a dict with 'embedding' key, but got {} instead".format(type(output)))

    def _compute_frame_onset_density(self, ith_frame=None):
        import librosa.util
        frame_onset_curve = self.onset_curve.get_ith_batch(ith_frame)
        onsets = librosa.util.peak_pick(frame_onset_curve, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=0.8, wait=5)
        num_onsets = len(onsets)
        od = float(num_onsets) / float(self.config.prediction_window_len)
        return od

    def _compute_rms(self, ith_frame=None):
        frame_rms = self.rms.get_ith_batch(ith_frame)
        return np.mean(frame_rms)

    def _predict_all_features(self, spec):
        self.ml_preds_and_times_list = []
        batched_spec = Batchifier(spec, self.window_len_frames, self.hop_len_frames)
        for i, (frame_end_idx, frame) in enumerate(batched_spec.batchify()):
            frame_i_time = self.audio_processor.frames_to_times(frame_end_idx)

            if self.config.normalize_spectrogram:
                frame = normalize_spec(frame, dset_name='midlevel', aud_processor=self.audio_processor, method='single')
            inputs = torch.tensor(frame).unsqueeze(0).unsqueeze(0).to(self.device)
            frame_midlevel_preds = self._get_model_predictions(inputs)

            frame_additional_features = []
            if self.config.additional_features.onset_density:
                frame_od = self._compute_frame_onset_density(i)
                frame_additional_features.append(frame_od)
            if self.config.additional_features.rms:
                frame_rms = self._compute_rms(i)
                frame_additional_features.append(frame_rms)

            self.ml_preds_and_times_list.append(np.hstack([np.array(frame_i_time),
                                                           frame_midlevel_preds,
                                                           frame_additional_features]))  # [time, ml1, ml2, ml3, ...]

    def _predict_embeddings(self, spec):
        self.ml_preds_and_times_list = []
        batched_spec = Batchifier(spec, self.window_len_frames, self.hop_len_frames)
        for i, (frame_end_idx, frame) in enumerate(batched_spec.batchify()):
            frame_i_time = self.audio_processor.frames_to_times(frame_end_idx)

            if self.config.normalize_spectrogram:
                frame = normalize_spec(frame, dset_name='midlevel', aud_processor=self.audio_processor, method='single')
            inputs = torch.tensor(frame).unsqueeze(0).unsqueeze(0).to(self.device)
            frame_midlevel_preds = self._get_embeddings(inputs)

            self.ml_preds_and_times_list.append(np.hstack([np.array(frame_i_time),
                                                           frame_midlevel_preds]))  # [time, ml1, ml2, ml3, ...]

    def _process_audio_file(self):
        spec_cache_path = utils.get_spectrogram_cache_path(self.audio_fp, self.audio_processor)
        if os.path.exists(spec_cache_path):
            return np.load(spec_cache_path)
        else:
            spec = self.audio_processor.process(self.audio_fp).spec
            utils.ensure_parents_exist(spec_cache_path)
            np.save(spec_cache_path, spec)
            return spec

    def get_midlevels(self):
        return np.array(self.ml_preds_and_times_list)[:, 1:]

    def get_midlevels_dataframe(self):
        return pd.DataFrame(self.get_midlevels(), columns=self.ml_names)

    def get_midlevels_dataframe_with_time(self):
        return pd.DataFrame(self.ml_preds_and_times_list, columns=['frame_start_time'] + self.ml_names)

    def get_midlevels_mean(self):
        return np.mean(self.get_midlevels(), axis=0)

    def save_midlevels(self, fp: str, mode: str = 'features_and_times'):
        assert mode in ['only_features', 'features_and_times']
        if mode == 'features_and_times':
            df = self.get_midlevels_dataframe_with_time()
        else:
            df = self.get_midlevels_dataframe()
        parents, _ = os.path.split(fp)
        os.makedirs(parents, exist_ok=True)
        df.to_csv(fp, index=False)

    def _precompute_track_level_features(self):
        # Load audio file and get spectrogram
        y, sr = load_audio_file(self.audio_fp, sample_rate=None, num_channels=1)
        self.spectrogram = self.audio_processor(self.audio_fp).spec

        # Compute RMS if enabled
        if self.config.additional_features.rms:
            self.rms = Batchifier(librosa.feature.rms(y=y.astype(float)), self.window_len_frames, self.hop_len_frames)

        # Compute onset density if enabled
        if self.config.additional_features.onset_density:
            # Compute onset density using piano method
            if self.config.onset_detection_method == 'piano':
                piano_activity = RNNPianoNoteProcessor()(self.audio_fp)
                self.onset_curve = Batchifier(piano_activity.sum(axis=0), self.window_len_frames, self.hop_len_frames)
            # Compute onset density using superflux method
            else:
                self.onset_curve = Batchifier(madmom.features.onsets.superflux(self.spectrogram.T), self.window_len_frames, self.hop_len_frames)

    def predict_file(self, audio_fp: str, return_what='midlevels', output_aggregate: str = 'mean', saveto: str = None):
        """
        Predicts the mid-level features of an audio file and optionally saves them to a file.

        :param audio_fp: The filepath to the audio file.
        :param output_aggregate: The aggregation method to use when returning the mid-level features.
                                 Can be 'mean' or None. Default is 'mean'.
        :param saveto: The filepath to save the mid-level features to. Default is None (no saving).
        :return: The mid-level features of the audio file, aggregated according to `output_aggregate`.
        """
        assert output_aggregate in ['mean', None], "Invalid output_aggregate parameter. Must be 'mean' or None."

        self.audio_fp = audio_fp

        spec = self._process_audio_file()
        if return_what == 'midlevels':
            if any(self.config.additional_features.values()):
                self._precompute_track_level_features()

            self._predict_all_features(spec)
        else:
            self._predict_embeddings(spec)

        if saveto:
            self.save_midlevels(saveto, mode='features_and_times')

        if output_aggregate == 'mean':
            return self.get_midlevels_mean()
        else:
            return self.get_midlevels()


    def predict_dir(self, audio_fp: str, output_aggregate: str = 'mean', saveto: str = None):
        pass


if __name__ == '__main__':
    config = OmegaConf.create(default_config)
    np.set_printoptions(precision=2)
    predictor = InferenceMidlevelModel(config=config)
    for fp in utils.list_files_deep(path_ce_personal_audio_all, full_paths=True):
        yid = os.path.basename(fp)[:-4]
        save_path = os.path.join(path_ce_personal_root, 'features_audio', 'midlevel_2bf74_2_embeddings', yid)

        mls = predictor.predict_file(fp, return_what='embeddings')
        np.save(save_path, mls)
