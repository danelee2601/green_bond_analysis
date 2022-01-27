"""
Pytorch lightning module (plm) for training
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import wandb
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

from ssl_methods.vibcreg import VIbCReg
from utils import root_dir
from utils.types import *


def average_epoch_end_outs(outs):
    mean_outs = {k: 0. for k in outs[0].keys()}
    for k in mean_outs.keys():
        for i in range(len(outs)):
            mean_outs[k] += outs[i][k]
        mean_outs[k] /= len(outs)
    return mean_outs


class TrainPLM(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 classifier: nn.Module,
                 config: dict,
                 n_train_samples: int,
                 GB_size: int,
                 VB_size: int,
                 ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.model = nn.Sequential(self.encoder, self.classifier)
        self.config = config

        self.vibcreg = VIbCReg(self.encoder, **self.config['enc_param'])
        n_classes = 2
        class_weight = torch.Tensor([(GB_size + VB_size) / (n_classes * VB_size),
                                     (GB_size + VB_size) / (n_classes * GB_size)]) if self.config['loss_weight']['use_class_weight'] else None
        print('class_weight:', class_weight)
        self.criterion_clf = nn.CrossEntropyLoss(weight=class_weight)  # nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.T_max = config['trainer_params']['max_epochs'] * np.ceil(
            n_train_samples / config['data']['batch_size'] + 1)  # Maximum number of iterations

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        self.encoder.train()
        self.classifier.train()
        self.vibcreg.projector.train()
        loss_hist = {}

        # fetch data
        (GB_ts, closest_VB_ts), (MR_ts, label) = batch
        GB_ts = torch.nan_to_num(GB_ts)
        closest_VB_ts = torch.nan_to_num(closest_VB_ts)
        MR_ts = torch.nan_to_num(MR_ts)
        label = label.reshape(-1)

        # pull between `GB_ts` and `closest_VB_ts` (regression)
        # y_gb, z_gb = self.vibcreg(GB_ts)
        # y_vb, z_vb = self.vibcreg(closest_VB_ts)
        # vibcreg_loss = self.vibcreg.loss_function(z_gb, z_vb, self.config['vibcreg'], loss_hist)

        # classify MR_ts (classification)
        out = self.model(MR_ts)
        label_smoothing = self.config['loss_weight']['label_smoothing']
        label_rev = (label - 1).abs()
        clf_loss = (1 - label_smoothing) * self.criterion_clf(out, label) + label_smoothing * self.criterion_clf(out, label_rev)
        loss_hist['clf_loss'] = clf_loss

        # total loss
        # loss = (self.config['loss_weight']['w_vibcreg'] * vibcreg_loss) + \
        #        (self.config['loss_weight']['w_clf'] * clf_loss)
        loss = (self.config['loss_weight']['w_clf'] * clf_loss)
        loss_hist['loss'] = loss

        # metrics
        pred = torch.argmax(out.detach().cpu(), dim=1).numpy().astype(int)
        label = label.cpu().numpy().astype(int)
        loss_hist['f1_score'] = f1_score(label, pred)
        loss_hist['acc'] = accuracy_score(label, pred)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        for k in loss_hist.keys():
            if k not in ['loss', 'f1_score', 'acc']:
                loss_hist[k] = loss_hist[k].detach()
        return loss_hist

    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        self.classifier.eval()
        self.vibcreg.projector.eval()
        loss_hist = {}

        # fetch data
        (GB_ts, closest_VB_ts), (MR_ts, label) = batch
        GB_ts = torch.nan_to_num(GB_ts)
        closest_VB_ts = torch.nan_to_num(closest_VB_ts)
        MR_ts = torch.nan_to_num(MR_ts)
        label = label.reshape(-1)

        # pull between `GB_ts` and `closest_VB_ts` (regression)
        # y_gb, z_gb = self.vibcreg(GB_ts)
        # y_vb, z_vb = self.vibcreg(closest_VB_ts)
        # vibcreg_loss = self.vibcreg.loss_function(z_gb, z_vb, self.config['vibcreg'], loss_hist)

        # classify MR_ts (classification)
        out = self.model(MR_ts)
        clf_loss = self.criterion_clf(out, label)
        loss_hist['clf_loss'] = clf_loss

        # total loss
        # loss = (self.config['loss_weight']['w_vibcreg'] * vibcreg_loss) + \
        #        (self.config['loss_weight']['w_clf'] * clf_loss)
        loss = (self.config['loss_weight']['w_clf'] * clf_loss)
        loss_hist['loss'] = loss

        # metrics
        pred = torch.argmax(out.detach().cpu(), dim=1).numpy().astype(int)
        label = label.cpu().numpy().astype(int)
        loss_hist['f1_score'] = f1_score(label, pred)
        loss_hist['acc'] = accuracy_score(label, pred)

        for k in loss_hist.keys():
            if k not in ['loss', 'f1_score', 'acc']:
                loss_hist[k] = loss_hist[k].detach()
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.encoder.parameters()},
                                {'params': self.classifier.parameters()},
                                {'params': self.vibcreg.projector.parameters()}],
                                lr=self.config['exp_params']['LR'],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt,
                'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def training_epoch_end(self, outs) -> None:
        mean_outs = average_epoch_end_outs(outs)
        log_items = {'epoch': self.current_epoch}
        for k, v in mean_outs.items():
            if v is not None:
                log_items[f'train/{k}'] = mean_outs[k]
        wandb.log(log_items)

        # save model
        if (self.current_epoch + 1) % self.config['exp_params']['model_save_ep_period'] == 0:
            if not os.path.isdir(root_dir.joinpath('checkpoints')):
                os.mkdir(root_dir.joinpath('checkpoints'))
            torch.save({'epoch': self.current_epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        }, root_dir.joinpath('checkpoints', f'model-ep_{self.current_epoch + 1}.ckpt'))

    def validation_epoch_end(self, outs) -> None:
        mean_outs = average_epoch_end_outs(outs)
        log_items = {'epoch': self.current_epoch}
        for k, v in mean_outs.items():
            if v is not None:
                log_items[f'validate/{k}'] = mean_outs[k]
        wandb.log(log_items)
