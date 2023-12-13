import argparse
import hashlib
import logging
import os
import time

import torch
from torch import optim, nn
import torch.utils.data as torchdata

from cpresnet import CPResnet, config_cp_field_shallow_m2
from data import load_midlevel_aljanaki, MidlevelDataset
from paths import MAIN_RUN_DIR
from utils import dset_to_loader, EarlyStopping
from train_funcs import train, test
from datetime import datetime as dt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtstr = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
name_hash = hashlib.sha1()
name_hash.update(str(time.time()).encode('utf-8'))
run_hash = name_hash.hexdigest()[:5]
run_name = f'{run_hash}_{dtstr}'
RUN_DIR = os.path.join(MAIN_RUN_DIR, 'midlevel_general', run_name)
os.makedirs(RUN_DIR)

logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(RUN_DIR, f'{run_name}.log'))
sh = logging.StreamHandler()  # for printing to terminal or console
formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

ml_tr_ids, ml_te_ids = load_midlevel_aljanaki(tsize=0.1)
ml_tr_dataset = MidlevelDataset(select_song_ids=ml_tr_ids, duration=15, normalize_inputs='single')
ml_te_dataset = MidlevelDataset(select_song_ids=ml_te_ids, duration=15, normalize_inputs='single')

train_set_size = int(len(ml_tr_dataset) * 0.8)
valid_set_size = len(ml_tr_dataset) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, valid_set = torchdata.random_split(ml_tr_dataset, [train_set_size, valid_set_size], generator=seed)

ml_tr_dataloader = dset_to_loader(train_set, bs=8, shuffle=True)
ml_va_dataloader = dset_to_loader(valid_set, bs=8)
ml_te_dataloader = dset_to_loader(ml_te_dataset, bs=8)

es = EarlyStopping(patience=13, condition="maximize", verbose=True,
                   save_dir=os.path.join(RUN_DIR, 'saved_models'),
                   saved_model_name=f"midlevel_general_cpresnetm2_{run_hash}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-devices', type=int, nargs='+')
    args = parser.parse_args()

    netz = CPResnet(num_targets=7, config=config_cp_field_shallow_m2).to(device)

    optimizer = optim.Adam(netz.parameters(), lr=0.001, weight_decay=1e-5, betas=(0.75, 0.90))
    criterion = nn.MSELoss().to(device)

    for epoch in range(1, 200):
        tr_losses = train(netz, ml_tr_dataloader, optimizer, criterion, epoch, run_hash + f'_epoch-{epoch}', dataloader_len=len(ml_tr_dataloader))
        logger.info(f"tr_loss: {tr_losses['avg_loss']}")
        val_metrics = test(netz, ml_va_dataloader, criterion, epoch, mets=['r2', 'corr_avg', 'rmse'])[0]
        logger.info(f"metrics: {val_metrics['corr_avg']}")

        es(val_metrics['corr_avg'], netz)
        if es.early_stop:
            logger.info(f"Early stop - trained for {epoch - es.counter} epochs - best metric {es.best_score}")
            break
