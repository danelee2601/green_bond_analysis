"""
Train a model for the classification task.

input: entire time series
output: 0 or 1
"""
from argparse import ArgumentParser
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from utils import load_yaml_param_settings
from preprocessing.build_data_pipeline import build_data_pipeline
from models import encoders, Classifier
from pl_modules.train_plm import TrainPLM
from utils import get_root_dir


def load_args():
    parser = ArgumentParser()
    root_dir = get_root_dir()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=root_dir.joinpath('configs', 'config.yaml'))
    return parser.parse_args()


if __name__ == '__main__':

    # load config(s)
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    train_data_loader, test_data_loader = build_data_pipeline(config)

    # build model (encoder + classifier)
    encoder = encoders[config['enc_param']['name']]()
    classifier = Classifier(**config['clf_param'])

    # fit
    train_plm = TrainPLM(encoder, classifier, config, n_train_samples=train_data_loader.dataset.__len__())
    wandb_logger = WandbLogger(project='ML_green_bond', name=None, config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         checkpoint_callback=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         **config['trainer_params'],)
    trainer.fit(train_plm, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)
    wandb.finish()
