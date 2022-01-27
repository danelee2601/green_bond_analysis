from .resnet import resnet_small, resnet18, resnet34
from .simple import simplenet
from .classifier import Classifier
from .vit import ViT

encoders = {'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet_small': resnet_small,
            'simplenet': simplenet,
            'ViT': ViT,
            }
