import pandas as pd
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging

from utils import compute_metrics

logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, criterion, epoch, run_name, **kwargs):
    model.train()
    loss_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=len(dataloader), desc=run_name):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        output = model(inputs)
        try:
            loss = criterion(output['output'].float(), labels.float())
        except Exception as e:
            print(e)
            loss = criterion(output.float(), labels.float())

        if kwargs.get('loss_batch_average'):
            loss = torch.mean(torch.sum(loss, axis=-1))

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss = np.mean(loss_list)

    return {'avg_loss': epoch_loss}


def test(model, dataloader, criterion, epoch=-1, **kwargs):
    if kwargs.get('mets') is None:
        mets = ['corr_avg']
    else:
        mets = kwargs['mets']

    if kwargs.get('test_output') is not None:
        test_output = kwargs['test_output']
    else:
        test_output = 'output'

    model.eval()
    loss_list = []
    preds_list = []
    labels_list = []
    return_dict = {}

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc=f"Testing {test_output} ... "):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(len(inputs.shape) - 2)
        output = model(inputs)[test_output]

        if (idxs := kwargs.get('idxs')) is not None:
            output = output[:, idxs[0]: idxs[1]]

        loss = criterion(output.float(), labels.float())
        preds_list.append(output.cpu().detach().numpy())

        if kwargs.get('loss_batch_average'):
            loss = torch.mean(torch.sum(loss, axis=-1))
        loss_list.append(loss.item())
        labels_list.append(labels.cpu().detach().numpy())

    epoch_test_loss = np.mean(loss_list)
    return_dict['avg_loss'] = epoch_test_loss
    if kwargs.get('compute_metrics', True) is True:
        return_dict.update(compute_metrics(np.vstack(labels_list), np.vstack(preds_list), metrics_list=mets))

    return return_dict, np.vstack(labels_list), np.vstack(preds_list)


def predict_mls(model, dataloader, mls):
    model.eval()
    preds_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc='Predicting...'):
        song_ids, inputs = batch
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(len(inputs.shape) - 2)

        output = model(inputs)

        ml_preds = output['output'].cpu().detach().numpy()
        preds_list.append(np.hstack([np.array(song_ids).reshape((len(song_ids), 1)), ml_preds]))

    preds_np = np.vstack(preds_list)
    preds = pd.DataFrame(preds_np, columns=['path'] + mls)

    return preds


class ReduceOnPlateau(ReduceLROnPlateau):
    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logger.info('Epoch {:5d}: reducing learning rate'
                                ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
