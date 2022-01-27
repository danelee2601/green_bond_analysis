from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import numpy as np

from preprocessing.augmentations import augmentations
# from preprocessing.preprocess import GBVBDataset
from preprocessing.preprocess2 import GBVBDataset
from utils.types import *


def build_data_pipeline(cf: dict) -> Tuple[DataLoader, DataLoader]:
    cf_data = cf['data']
    batch_size = cf_data['batch_size']
    num_workers = cf_data['num_workers']

    # build transforms
    augs = []
    for aug_name in cf_data['augmentations']:
        if aug_name in augmentations.keys():
            augs.append(augmentations[aug_name]())
    transforms = Compose(augs)

    # build datasets
    np.random.seed(cf_data['rand_seed'])
    train_dataset = GBVBDataset(transform=transforms,
                                kind='train',
                                **cf_data)
    np.random.seed(cf_data['rand_seed'])
    test_dataset = GBVBDataset(transform=transforms,
                               kind='test',
                               **cf_data)

    # build data_loaders
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,  # we don't have many GB samples
                                   num_workers=num_workers,
                                   pin_memory=True if num_workers > 0 else False)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=num_workers,
                                  pin_memory=True if num_workers > 0 else False)

    return train_data_loader, test_data_loader
