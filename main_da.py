import sys
import os
from torch.utils.data import DataLoader

from data import get_extra_dataset, load_midlevel_domain_split, load_midlevel_extra, aljanaki_split, MidlevelDataset
from utils import get_mmd
from train_funcs import test, train_da_backprop, train
from cpresnet import config_cp_field_shallow_m2

PROJECT_NAME = 'midlevel_da'
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)
SUBPROJECT_NAME = 'main_da'

import hashlib
import time
import logging
from torch import nn, optim
from sklearn.model_selection import train_test_split
from cpresnet import CPResnet_BackProp, CPResnet

from datetime import datetime as dt
from utils import *
from paths import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
SEED = int(int(time.time() * 10e3) - int(time.time()) * 10e3)
rState = np.random.RandomState(seed=SEED)
train_valid_split = train_test_split
ml_names = ['melody', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness']
NUM_WORKERS = 8 if torch.cuda.is_available() else 0

dtstr = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
hash = hashlib.sha1()
hash.update(str(time.time()).encode('utf-8'))
run_hash = hash.hexdigest()[:5]
run_name = f'{run_hash}_{dtstr}'

if os.uname()[1] in ['shreyan-HP', 'shreyan-All-Series']:
    PROJECT_RUN_DIR = os.path.join(MAIN_RUN_DIR, '_debug_runs')
else:
    PROJECT_RUN_DIR = os.path.join(MAIN_RUN_DIR, SUBPROJECT_NAME)
if not os.path.exists(os.path.join(PROJECT_RUN_DIR, run_name)):
    os.makedirs(os.path.join(PROJECT_RUN_DIR, run_name))

curr_run_dir = os.path.join(PROJECT_RUN_DIR, run_name)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(curr_run_dir, f'{run_name}.log'))
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)


def main():
    # HPARAMS
    h = dict(domain_split='piano',
             batch_size=8,
             learning_rate=1e-3,
             patience=14,
             input_noise='none',
             label_noise=0.0,
             da=False,
             tg_in_tr=1.0
             )
    log_dict(logger, h, delimiter=', ')
    logger.info(f"SEED = {SEED}")
    # -------------------------------------------------

    # INIT MODEL
    if h['da']:
        net = CPResnet_BackProp(config=config_cp_field_shallow_m2, num_targets=7).to(device)
    else:
        net = CPResnet(config=config_cp_field_shallow_m2, num_targets=7).to(device)

    # net = audio2feature2_model(num_targets=7).to(device)

    # INIT DATA
    sc_ids, tg_ids = load_midlevel_domain_split('piano')
    _, extra_piano_pieces = load_midlevel_extra('piano')
    extra_piano_pieces = extra_piano_pieces[np.isin(extra_piano_pieces, tg_ids, invert=True)]
    sc_ids = sc_ids[np.isin(sc_ids, extra_piano_pieces, invert=True)]

    leak_ids = np.random.choice(extra_piano_pieces, int(h['tg_in_tr'] * len(extra_piano_pieces)), replace=False)
    sc_ids = np.hstack([sc_ids, leak_ids])

    # 10% of data as evaluation set. Aljanaki split: eval split contains tracks by artists not in the training set
    sc_tr_ids, sc_te_ids = aljanaki_split(sc_ids, tsize=int(round(0.1 * len(sc_ids))), seed=SEED)
    # 20% of eval data as validation set, 80% as test set
    sc_te_ids, sc_va_ids = train_valid_split(sc_te_ids, test_size=int(round(0.2 * len(sc_te_ids))), random_state=SEED)

    sc_tr_dataset = MidlevelDataset(select_song_ids=sc_tr_ids, duration=15, normalize_inputs='single')
    sc_va_dataset = MidlevelDataset(select_song_ids=sc_va_ids, duration=15, normalize_inputs='single')
    sc_te_dataset = MidlevelDataset(select_song_ids=sc_te_ids, duration=15, normalize_inputs='single')
    tg_te_dataset = MidlevelDataset(select_song_ids=tg_ids, duration=15, normalize_inputs='single')

    logger.info(f"LENGTHS: sc_tr={len(sc_tr_dataset)}, sc_va={len(sc_va_dataset)}, tg_te={len(tg_te_dataset)}")

    sc_tr_dataloader = DataLoader(sc_tr_dataset, batch_size=h['batch_size'], shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)
    sc_va_dataloader = DataLoader(sc_va_dataset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)
    sc_te_dataloader = DataLoader(sc_te_dataset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)

    tg_tr_dataset = get_extra_dataset('maestro', test_size=0.1, augment=None, normalizing_dset='midlevel')[1]
    tg_tr_dataloader = DataLoader(tg_tr_dataset, batch_size=h['batch_size'], shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)
    tg_te_dataloader = DataLoader(tg_te_dataset, batch_size=h['batch_size'], shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)

    # INIT TRAINER
    optimizer = optim.Adam(net.parameters(), lr=h['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 10, 15, 20], gamma=0.5)
    criterion = nn.MSELoss().to(device)
    es = EarlyStopping(patience=h['patience'], condition='maximize', verbose=True,
                       save_dir=os.path.join(curr_run_dir, 'saved_models'),
                       saved_model_name=net.name + '_' + run_name[:5] + '_teacher')

    test_metric = 'corr_avg'

    for epoch in range(1, 100):
        sc_tg_tr_dataloader = zip(sc_tr_dataloader, tg_tr_dataloader)
        if h['da']:
            train_da_backprop(net, sc_tg_tr_dataloader, optimizer, criterion, epoch, run_name + f'_epoch-{epoch}', dataloader_len=min(len(sc_tr_dataloader), len(tg_tr_dataloader)))
        else:
            train(net, sc_tr_dataloader, optimizer, criterion, epoch, run_name + f'_epoch-{epoch}')

        scheduler.step()
        sc_val_corr = test(net, sc_va_dataloader, criterion, epoch, mets=[test_metric])[0][test_metric]
        sc_test_corr = test(net, sc_te_dataloader, criterion, epoch, mets=[test_metric])[0]
        tg_test_corr = test(net, tg_te_dataloader, criterion, epoch, mets=[test_metric])[0]

        discrepancy = get_mmd(sc_te_dataloader, tg_te_dataloader, net)

        logger.info(
            f"Epoch {epoch} sc val {test_metric} = {round(sc_val_corr, 4)} | sc test {test_metric} = {round(sc_test_corr[test_metric], 4)} | "
            f"tg test {test_metric} = {round(tg_test_corr[test_metric], 4)} | discrepancy = {round(discrepancy, 4)}")

        es(sc_val_corr, net)
        if es.early_stop:
            logger.info(f"Early stop - trained for {epoch - es.counter} epochs - best metric {es.best_score}")
            break

    load_model(es.save_path, net)
    tg_test_corr = test(net, tg_te_dataloader, criterion, mets=[test_metric])[0][test_metric]

    logger.info(f"DA={h['da']} | tg_in_tr={h['tg_in_tr']}")
    logger.info(f"Final target {test_metric} = {round(tg_test_corr, 4)}")


if __name__ == '__main__':
    main()
