import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

from argparse import ArgumentParser
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from models import encoders, Classifier
from preprocessing.preprocess2 import GBVBDataset
from utils import root_dir, load_yaml_param_settings


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=root_dir.joinpath('configs', 'config.yaml'))
    return parser.parse_args()


dataset = None


def get_GB_VB_pairs(GB_ticker: str = None):
    if not GB_ticker:
        idx_pull = np.random.randint(0, len(dataset.GB_tickers))
        GB_ticker = dataset.GB_tickers[idx_pull]
    GB_ts = pd.read_csv(os.path.join('dataset', 'dataset_GB', f'{GB_ticker}-ts.csv'))

    closest_VB_ticker = dataset._get_closest_VB_ticker(GB_ticker)
    closest_VB_ts = pd.read_csv(os.path.join('dataset', 'dataset_VB', f'{closest_VB_ticker}-ts.csv'))

    GB_ts = (GB_ts['High'] - GB_ts['Low']).values
    closest_VB_ts = (closest_VB_ts['High'] - closest_VB_ts['Low']).values

    GB_ts = dataset._clip_timeseries(GB_ts)
    closest_VB_ts = dataset._clip_timeseries(closest_VB_ts)

    GB_ts = dataset._scale_ts(GB_ts)
    closest_VB_ts = dataset._scale_ts(closest_VB_ts)
    return (GB_ts, closest_VB_ts), (GB_ticker, closest_VB_ticker)


if __name__ == '__main__':
    import os
    import pandas as pd
    import torch
    import numpy as np

    # load config(s)
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # load model
    encoder = encoders['resnet18']()
    classifier = Classifier(**config['clf_param'])
    model = nn.Sequential(encoder, classifier)
    ckpt_path = 'checkpoints/clf_ssl-ep_200.ckpt'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state_dict'])
    model.eval()
    # print('model:\n', model)

    target_layers = [model[0].layer4[-1]]

    input_tensor = torch.rand(1, 1, 300)  # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # get data
    ts_len = 757
    dataset = GBVBDataset(ts_len, 0.1)
    (GB_ts, closest_VB_ts), (GB_ticker, closest_VB_ticker) = \
        get_GB_VB_pairs(GB_ticker=None)  # GB_ticker='DG'
    GB_ts = torch.from_numpy(GB_ts[None, None, :]).float()
    closest_VB_ts = torch.from_numpy(closest_VB_ts[None, None, :]).float()
    print('GB_ts.shape:', GB_ts.shape)  # (B, 1, ts_len)
    print('closest_VB_ts.shape:', closest_VB_ts.shape)  # (B, 1, ts_len)

    # compute CAM with a moving window
    targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1)]
    i = 0
    ts_crop_len = 300
    shift_rate = 0.5
    grayscale_cam = np.zeros((2, ts_len))
    count_cam = np.zeros((2, ts_len))
    while True:
        print(f'index: {i} - {i+ts_crop_len}')
        closest_VB_ts_ = closest_VB_ts[:, :, i:i+ts_crop_len]  # (1, 1, ts_cro_len)
        GB_ts_ = GB_ts[:, :, i:i+ts_crop_len]  # (1, 1, ts_cro_len)

        closest_VB_ts_ = torch.from_numpy(dataset._scale_ts(closest_VB_ts_.numpy()))
        GB_ts_ = torch.from_numpy(dataset._scale_ts(GB_ts_.numpy()))

        input = torch.concat((closest_VB_ts_, GB_ts_), dim=0)
        grayscale_cam_ = cam(input_tensor=input, targets=targets)

        grayscale_cam[:, i:i+ts_crop_len] = grayscale_cam_
        count_cam[:, i:i + ts_crop_len] += np.ones(grayscale_cam_.shape)

        i = i + np.floor(ts_crop_len * shift_rate).astype(int)
        if i >= ts_len:
            break
    grayscale_cam = grayscale_cam / count_cam
    print('grayscale_cam:\n', grayscale_cam)
    print('grayscale_cam.shape:', grayscale_cam.shape)

    # targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1)]
    # input = torch.concat((closest_VB_ts, GB_ts), dim=0)
    # grayscale_cam = cam(input_tensor=input, targets=targets)
    # print('grayscale_cam:\n', grayscale_cam)
    # print('grayscale_cam.shape:', grayscale_cam.shape)

    # get a sample
    batch_idx = 0
    channel_idx = 0
    VB_ts_ = closest_VB_ts[batch_idx][channel_idx].numpy()
    GB_ts_ = GB_ts[batch_idx][channel_idx].numpy()

    # visualization
    ts_len = GB_ts_.shape[-1]
    x = np.arange(ts_len)
    plt.figure(figsize=(10, 4))
    for i, ts in enumerate([VB_ts_, GB_ts_]):
        plt.subplot(2, 1, i+1)
        plt.title(f'VB: {closest_VB_ticker}' if i == 0 else f'GB: {GB_ticker}')
        plt.plot(x, np.sinh(ts))
        plt.scatter(x, np.sinh(ts),
                    s=grayscale_cam[i]*30,
                    c=grayscale_cam[i],
                    cmap='coolwarm')
        plt.xticks([])
        plt.tight_layout()
    plt.show()



