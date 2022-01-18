"""
Pytorch lightning module (plm) for training
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from utils.types import *


class TrainPLM(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 classifier: nn.Module,
                 config: dict,
                 n_train_samples: int,
                 ):
        super().__init__()
        self.model = nn.Sequential(encoder, classifier)
        self.config = config

        self.criterion = nn.BCEWithLogitsLoss(weight=None)
        # `weight` is probably needed given the imbalanced dataset.

        self.T_max = config['trainer_params']['max_epochs'] * np.ceil(n_train_samples / config['data']['batch_size'])  # Maximum number of iterations

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        return out

    def training_step_end(self, batch, batch_idx):
        self.model.train()

        # fetch data
        _ = batch

        # forward

        # loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        return {'loss': None, }

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        # fetch data
        _ = batch

        # forward

        # loss

        return {'loss': None, }

    def configure_optimizers(self):
        opt = torch.optim.Adam([{'params': self.encoder.parameters()},
                                {'params': self.classifier.parameters()}],
                               lr=self.config['exp_params']['LR'],
                               weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt,
                'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def training_epoch_end(self, outs) -> None:
        # average 'train outs'

        # log

        # save model
        pass

    def validation_epoch_end(self, outs) -> None:
        # average 'validate outs'

        # log
        pass
