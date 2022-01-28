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


if __name__ == '__main__':
    # load config(s)
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # load model
    encoder = encoders['resnet18']()
    classifier = Classifier(**config['clf_param'])
    model = nn.Sequential(encoder, classifier)
    ckpt_path = 'checkpoints/resnet18_ssl-ep_200.ckpt'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state_dict'])
    model.eval()
    # print('model:\n', model)

    target_layers = [model[0].layer4[-1]]

    input_tensor = torch.rand(1, 1, 300)  # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    # get data
    dataset = GBVBDataset(757, 0.2, kind='train')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    (GB_ts, closest_VB_ts), (MR_ts, label) = next(iter(data_loader))
    print('GB_ts.shape:', GB_ts.shape)
    print('closest_VB_ts.shape:', closest_VB_ts.shape)
    print('MR_ts.shape:', MR_ts.shape)
    print('label.shape:', label.shape)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # targets = [ClassifierOutputTarget(l) for l in label.reshape(-1)]
    targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1)]
    input = torch.concat((closest_VB_ts, GB_ts), dim=0)
    grayscale_cam = cam(input_tensor=input, targets=targets)
    print('grayscale_cam.shape:', grayscale_cam.shape)
    print('np.sum(grayscale_cam):', np.sum(grayscale_cam))

    # get a sample
    batch_idx = 0
    channel_idx = 0
    VB_ts_ = closest_VB_ts[batch_idx][channel_idx].numpy()
    GB_ts_ = GB_ts[batch_idx][channel_idx].numpy()
    # label_ = label[batch_idx].numpy()
    # grayscale_cam_ = grayscale_cam[batch_idx]
    # print('grayscale_cam_:', grayscale_cam_)
    # print('label_:', label_)
    # str_label_ = 'VB' if label_ == 0 else 'GB'
    # print('str_label_:', str_label_)

    # visualization
    ts_len = GB_ts_.shape[-1]
    x = np.arange(ts_len)
    plt.figure(figsize=(10, 3))
    for i, ts in enumerate([VB_ts_, GB_ts_]):
        plt.subplot(2, 1, i+1)
        plt.title('VB' if i == 0 else 'GB')
        plt.plot(x, ts)
        plt.scatter(x, ts,
                    s=grayscale_cam[i]*30,
                    c=grayscale_cam[i],
                    cmap='coolwarm')
        plt.tight_layout()
    plt.show()
