import logging
import os
import pickle
import random

import torch
import numpy as np
# from pymde.datasets import Dataset
from scipy.stats import pearsonr
from sklearn import metrics
from torch.utils.data import Subset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ml_names = ['melody', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness']

logger = logging.getLogger()

def init_logger(run_dir, run_name):
    global logger
    fh = logging.FileHandler(os.path.join(run_dir, f'{run_name}.log'))
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

def dset_to_loader(dset, bs, shuffle=False):
    num_workers = 1 if os.uname()[1] in ['shreyan-HP-EliteBook-840-G5', 'shreyan-All-Series'] else bs
    return DataLoader(dset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=False)



def get_dataset_name_from_file_path(fp):
    datasets_idx = fp.find('datasets')
    if datasets_idx == -1:
        return None

    subdirs = fp[datasets_idx:].split(os.path.sep)
    dataset_name = None
    for i in range(len(subdirs)):
        if subdirs[i] in ('datasets', 'datasets@fs') and i + 1 < len(subdirs):
            dataset_name = subdirs[i + 1]
            break

    return dataset_name


def get_spectrogram_cache_path(audio_fp, audio_processor, dataset_name=None, file_name=None, cache_dir=None):
    if dataset_name is None:
        dataset_name = get_dataset_name_from_file_path(audio_fp)
    if file_name is None:
        file_name = os.path.splitext(os.path.basename(audio_fp))[0]
    processor_dir_name = audio_processor.name
    spec_cache_path = os.path.join(cache_dir, dataset_name, processor_dir_name, file_name+'.npy')
    return spec_cache_path


def ensure_parents_exist(path):
    parents, _ = os.path.split(path)
    if not os.path.exists(parents):
        os.makedirs(parents, exist_ok=True)
        print(f"created dir {parents}")



def num_if_possible(s):
    try:
        return int(s)
    except Exception as e:
        pass

    try:
        return float(s)
    except Exception as e:
        pass

    if s in ['True', 'true']:
        return True
    if s in ['False', 'false']:
        return False

    return s

def list_files_deep(dir_path, full_paths=True, filter_ext=None):
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(dir_path, '')):
        if len(filenames) > 0:
            for f in filenames:
                if full_paths:
                    all_files.append(os.path.join(dirpath, f))
                else:
                    all_files.append(f)

    if filter_ext is not None:
        return [f for f in all_files if os.path.splitext(f)[1] in filter_ext]
    else:
        return all_files


def save(model, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    try:
        torch.save(model.module.state_dict(), path)
    except AttributeError:
        torch.save(model.state_dict(), path)


def pickledump(data, fp):
    d = os.path.dirname(fp)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def pickleload(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def dumptofile(data, fp):
    d = os.path.dirname(fp)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(fp, 'w') as f:
        print(data, file=f)


def print_dict(dict, round):
    for k, v in dict.items():
        print(f"{k}:{np.round(v, round)}")


def log_dict(logger, dict, round=None, delimiter='\n'):
    log_str = ''
    for k, v in dict.items():
        if isinstance(round, int):
            try:
                log_str += f"{k}: {np.round(v, round)}{delimiter}"
            except:
                log_str += f"{k}: {v}{delimiter}"
        else:
            log_str += f"{k}: {v}{delimiter}"
    logger.info(log_str)


def load_model(model_weights_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_weights_path))
    else:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()


def inf(dl):
    """Infinite dataloader"""
    while True:
        for x in iter(dl): yield x


def choose_rand_index(arr, num_samples):
    return np.random.choice(arr.shape[0], num_samples, replace=False)


def compute_metrics(y, y_hat, metrics_list, **kwargs):
    metrics_res = {}
    for metric in metrics_list:
        Y, Y_hat = y, y_hat
        if metric in ['rocauc-macro', 'rocauc']:
            metrics_res[metric] = metrics.roc_auc_score(Y, Y_hat, average='macro')
        if metric == 'rocauc-micro':
            metrics_res[metric] = metrics.roc_auc_score(Y, Y_hat, average='micro')
        if metric in ['prauc-macro', 'prauc']:
            metrics_res[metric] = metrics.average_precision_score(Y, Y_hat, average='macro')
        if metric == 'prauc-micro':
            metrics_res[metric] = metrics.average_precision_score(Y, Y_hat, average='micro')

        if metric == 'corr_avg':
            corr, pval = [], []
            for i in range(kwargs.get("num_cols", 7)):
                c, p = pearsonr(Y[:, i], Y_hat[:, i])
                corr.append(c)
            metrics_res['corr_avg'] = np.mean(corr)

        if metric == 'corr':
            corr, pval = [], []
            for i in range(kwargs.get("num_cols", 7)):
                c, p = pearsonr(Y[:, i], Y_hat[:, i])
                corr.append(c)
            metrics_res['corr'] = corr

        if metric == 'mae':
            metrics_res[metric] = metrics.mean_absolute_error(Y, Y_hat)

        if metric == 'r2':
            metrics_res[metric] = metrics.r2_score(Y, Y_hat)

        if metric == 'r2_raw':
            metrics_res[metric] = metrics.r2_score(Y, Y_hat, multioutput='raw_values')

        if metric == 'mse':
            metrics_res[metric] = metrics.mean_squared_error(Y, Y_hat)

        if metric == 'rmse':
            metrics_res[metric] = np.sqrt(metrics.mean_squared_error(Y, Y_hat))

        if metric == 'rmse_raw':
            metrics_res[metric] = np.sqrt(metrics.mean_squared_error(Y, Y_hat, multioutput='raw_values'))

    return metrics_res


class DataLogger():
    def __init__(self, path=None):
        if path is None:
            path = os.getcwd()
        os.makedirs(path, exist_ok=True)
        self.path = path
        self._logs = {}

    def log(self, logdict, step=None):
        for k, v in logdict.items():
            with open(os.path.join(self.path, k+'.csv'), 'a') as logfile:
                if step is not None:
                    logfile.write(f'{step}, {v}\n')
                    if k not in self._logs:
                        self._logs[k] = {step:v}
                    else:
                        self._logs[k].update({step:v})
                else:
                    # logfile.write(f'{v}\n')
                    # self.logs[k].update({step:v})
                    raise NotImplementedError

    def get_data(self, key, step):
        return self._logs[key][step]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def pytorch_random_sampler(dset, num_samples):
    assert num_samples < len(dset)
    sample_indices = np.random.choice(len(dset), num_samples)
    return Subset(dset, sample_indices)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_dir='.', saved_model_name="model_chkpt",
                 condition='minimize'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saved_model_name = saved_model_name
        self.save_path = os.path.join(self.save_dir, self.saved_model_name + '.pt')
        self.condition = condition
        assert condition in ['maximize', 'minimize']
        self.metric_best = np.Inf if condition == 'minimize' else -np.Inf

    def __call__(self, metric, model):

        score = metric if self.condition == 'maximize' else -metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Metric improved ({self.condition}) ({self.metric_best:.6f} --> {metric:.6f}).  Saving model to {os.path.join(self.save_dir, self.saved_model_name + ".pt")}')
        torch.save(model.state_dict(), self.save_path)
        self.metric_best = metric


def get_centroid(dataloader, model):
    all_outputs = []
    for batch_idx, (_, inputs, labels) in enumerate(dataloader):
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            try:
                all_outputs.append(outputs.cpu())
            except:
                all_outputs.append(outputs['output'].cpu())
    return torch.mean(torch.cat(all_outputs), dim=0)


def get_mmd(loader_1, loader_2, model):
    print("calculating mmd")
    model.eval()
    centroid_1 = get_centroid(loader_1, model)
    centroid_2 = get_centroid(loader_2, model)
    model.train()
    return torch.dist(centroid_1, centroid_2, 2).item()


def mmd_select_naive(mmd):
    return np.argmin(mmd)


def mmd_select_scale(mmd, sce):
    sce = np.asarray(sce)
    mmd = np.asarray(mmd)
    scl = np.min(sce) / np.min(mmd)
    return np.argmin(sce + mmd * scl)
