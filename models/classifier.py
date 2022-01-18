import torch.nn as nn
from utils.types import *


class Classifier(nn.Module):
    """
    Linear classifier
    """
    def __init__(self, in_size: int, out_size: int):
        super(Classifier, self).__init__()
        self.clf = nn.Linear(in_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.clf(x)
