import hashlib
import os
import warnings

# import librosa
from tqdm import tqdm

from audio import MadmomAudioProcessor
from paths import *
from utils import *
import torch
import torch.utils.data
import numpy as np
from sklearn.model_selection import train_test_split
import concurrent.futures

dset_stats = {}


def load_split(dset, split_type, split_id):
    assert dset in ['deam', 'pmemo']
    if dset == 'deam':
        dset_root = path_deam_root
    else:
        dset_root = path_pmemo_root
    assert split_type in ['train', 'val', 'test']
    split_files = list_files_deep(os.path.join(dset_root, 'splits'))
    selected_split = [f for f in split_files if split_type in f and str(split_id) in f]
    assert len(selected_split) == 1
    song_ids = np.load(os.path.join(dset_root, 'splits', selected_split[0]))
    return song_ids


# def load_audio(aud_path, target_sr):
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore")
#         waveform, _ = librosa.load(aud_path, mono=True, sr=target_sr)
#     return waveform


def get_dset_dirs_paths(dset_name, aud_processor):
    if aud_processor is None:
        aud_processor = MadmomAudioProcessor(fps=31.3)
    dataset_cache_dir = os.path.join(path_cache_fs, dset_name)
    all_specs_path = os.path.join(dataset_cache_dir, aud_processor.get_params.get("name"))
    stats_path = os.path.join(dataset_cache_dir, aud_processor.get_params.get("name")+'_stats')
    return all_specs_path, stats_path


def get_dataset_stats(dset_name, aud_processor=None):
    global dset_stats
    all_specs_path, stats_path = get_dset_dirs_paths(dset_name, aud_processor)
    os.makedirs(stats_path, exist_ok=True)

    if dset_stats.get(dset_name) is not None:
        try:
            mean = dset_stats[dset_name]['mean']
            std = dset_stats[dset_name]['std']
        except KeyError:
            raise Exception(f"mean or std not found for {dset_name}")
    else:
        all_specs_list = list_files_deep(all_specs_path, full_paths=True, filter_ext=['.npy'])
        all_specs_list = np.random.choice(all_specs_list, size=3000, replace=False)

        try:
            mean = np.load(os.path.join(stats_path, 'mean.npy'))
        except FileNotFoundError:
            mean = 0.0
            for specnpy in tqdm(all_specs_list, desc=f'Calculating mean for {dset_name} {aud_processor.get_params.get("name")}'):
                spec = np.load(specnpy)
                mean += np.mean(spec).item()
            mean = mean / len(all_specs_list)
            np.save(os.path.join(stats_path, 'mean.npy'), mean)

        try:
            std = np.load(os.path.join(stats_path, 'std.npy'))
        except FileNotFoundError:
            sum_of_mean_of_squared_dev = 0.0
            for specnpy in tqdm(all_specs_list, desc=f'Calculating std for {dset_name} {aud_processor.get_params.get("name")}'):
                spec = np.load(specnpy)
                sum_of_mean_of_squared_dev += np.mean(np.square(spec - mean)).item()
            std = np.sqrt(sum_of_mean_of_squared_dev / len(all_specs_list))
            np.save(os.path.join(stats_path, 'std.npy'), std)

        dset_stats[dset_name] = {'mean': mean, 'std': std}

    return mean, std


def get_dataset_mean_and_std_spec(dset_name, proc):
    stats_dir = os.path.join(path_cache_fs, dset_name, proc.get_params.get("name")+'_stats')
    mean_spec = np.load(os.path.join(stats_dir, 'mean_spec.npy'))
    std_spec = np.load(os.path.join(stats_dir, 'std_spec.npy'))
    return mean_spec, std_spec


def normalize_spec(spec, mean=None, std=None, dset_name=None, aud_processor=None, method='single'):
    """
    if method is "single", it normalizes all pixels of a spectrogram by the same value, which is the average pixel value
    across all spectrograms of a dataset. If method is "spec", it normalizes by the mean spectrogram image (each pixel
    has its own mean value).
    """
    if method == 'single':
        if mean is None and std is None:
            mean, std = get_dataset_stats(dset_name, aud_processor)
        assert (isinstance(mean, np.ndarray) and isinstance(std, np.ndarray)) or (isinstance(mean, float) and isinstance(std, float)), \
            print(f"Either mean or std is not a float: mean={mean}, std={std}")
        return (spec - mean) / std
    elif method == 'spec':
        mean_spec, std_spec = get_dataset_mean_and_std_spec(dset_name, aud_processor)
        return (spec[:, :mean_spec.shape[1]] - mean_spec) / std_spec
    else:
        return spec



def slice_func(spec, length, processor=None, mode='random', offset_seconds=0, slice_times=None, padding='zero'):
    if slice_times is not None:
        start_time, end_time = slice_times[0], slice_times[1]
        return spec[:, processor.times_to_frames(start_time): processor.times_to_frames(end_time)], start_time, end_time

    offset_frames = int(processor.times_to_frames(offset_seconds))

    length = int(length)

    if padding == 'zero':
        spec = np.hstack([spec, np.zeros((spec.shape[0], max(offset_frames+length-spec.shape[-1], 0)))])
        xlen = spec.shape[-1]
        midpoint = xlen // 2 + offset_frames
    else:
        while spec.shape[-1] < offset_frames + length:
            spec = np.append(spec, spec[:, :length - spec.shape[-1]], axis=1)
        xlen = spec.shape[-1]
        midpoint = xlen // 2 + offset_frames

    if mode == 'start':
        start_time = processor.frames_to_times(offset_frames)
        end_time = processor.frames_to_times(offset_frames + length)
        output = spec[:, offset_frames: offset_frames + length]
    elif mode == 'end':
        start_time = processor.frames_to_times(xlen - length)
        end_time = processor.frames_to_times(xlen)
        output = spec[:, -length:]
    elif mode == 'middle':
        start_time = processor.frames_to_times(xlen - length)
        end_time = processor.frames_to_times(xlen)
        output = spec[:, midpoint - length // 2: midpoint + length // 2 + 1]
    elif mode == 'random':
        k = torch.randint(offset_frames, xlen - length + 1, (1,))[0].item()
        start_time = processor.frames_to_times(k)
        end_time = processor.frames_to_times(k + length)
        output = spec[:, k: k + length]
    else:
        raise Exception(f"mode must be in ['start', 'end', 'middle', 'random'], is {mode}")

    return output, start_time, end_time


def slice_func_simple(spec, length, mode='random', padding='loop'):
    length = int(length)

    if padding == 'zero':
        spec = np.hstack([spec, np.zeros((spec.shape[0], max(length-spec.shape[-1], 0)))])
        xlen = spec.shape[-1]
        midpoint = xlen // 2
    else:
        while spec.shape[-1] < length:
            spec = np.append(spec, spec[:, :length - spec.shape[-1]], axis=1)
        xlen = spec.shape[-1]
        midpoint = xlen // 2

    if mode == 'start':
        sliced = spec[:, 0:length]
    elif mode == 'end':
        sliced = spec[:, -length:]
    elif mode == 'middle':
        sliced = spec[:, midpoint - length // 2: midpoint + length // 2 + 1]
    elif mode == 'random':
        k = torch.randint(0, xlen - length + 1, (1,))[0].item()
        sliced = spec[:, k: k + length]
    else:
        raise Exception(f"mode must be in ['start', 'end', 'middle', 'random'], is {mode}")

    return sliced



def slice_func_waveform(wf, length_seconds, sr=22050, mode='random', offset_seconds=0, slice_times=None, padding='zero'):
    # ONLY FOR MONO WF, as of now

    if slice_times is not None:
        start_time, end_time = slice_times[0], slice_times[1]
        return wf[int(start_time*sr) : int(end_time*sr)], start_time, end_time

    offset_samples = int(offset_seconds * sr)
    length_samples = int(length_seconds * sr)

    if padding == 'zero':
        wf = np.hstack([wf, np.zeros(max(offset_samples+length_samples-len(wf), 0))])
        midpoint = len(wf) // 2 + offset_samples
    else:
        while len(wf) < offset_samples + length_samples:
            wf = np.append(wf, wf[:length_samples - len(wf)])
        midpoint = len(wf) // 2 + offset_samples

    if mode == 'start':
        output = wf[offset_samples: offset_samples + length_samples]
        start_time = offset_seconds
        end_time = offset_seconds + length_seconds
    elif mode == 'end':
        output = wf[-length_samples:]
        start_time = (len(wf) - length_samples)/sr
        end_time = len(wf)
    elif mode == 'middle':
        output = wf[midpoint - length_samples // 2: midpoint + length_samples // 2 + 1]
        start_time = (midpoint - length_samples // 2)/sr
        end_time = (midpoint + length_samples // 2 + 1)/sr
    elif mode == 'random':
        k = torch.randint(offset_samples, len(wf) - length_samples + 1, (1,))[0].item()
        output = wf[k: k + length_samples]
        start_time = k/sr
        end_time = (k + length_samples)/sr
    else:
        raise Exception(f"mode must be in ['start', 'end', 'middle', 'random'], is {mode}")

    return output, start_time, end_time


class DsetNoLabel(torch.utils.data.Dataset):
    # Make sure that your dataset actually returns many elements!
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        ret_stuff = self.dset[index]
        return ret_stuff[:-1] if len(ret_stuff) > 2 else ret_stuff

    def __len__(self):
        return len(self.dset)


class DsetMultiDataSources(torch.utils.data.Dataset):
    def __init__(self, *dsets):
        self.dsets = dsets
        self.lengths = [len(d) for d in self.dsets]

    def __getitem__(self, index):
        return_triplets = []
        for ds in self.dsets:
            idx = index % len(ds)
            try:
                path, x, y = ds[idx]
                return_triplets.append((path, x, y))
            except:
                # if dataset does not return path, generate a (semi-)unique hash from the sum of the tensor, to be used as an identifier of the tensor for caching
                x, y = ds[idx]
                return_triplets.append((hashlib.md5(f"{str(torch.sum(x).item())}".encode("UTF-8")).hexdigest(), x, y))

        return tuple(return_triplets)

    def __len__(self):
        return min(self.lengths)


class DsetThreeChannels(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        image, label = self.dset[index]
        return image.repeat(3, 1, 1), label

    def __len__(self):
        return len(self.dset)


if __name__ == '__main__':
    print(get_dataset_stats('midlevel'))
