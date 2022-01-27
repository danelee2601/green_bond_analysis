import torch.nn as nn


# def simplenet():
#     return nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, stride=2),
#                          nn.GELU(),
#                          nn.Conv1d(64, 128, kernel_size=3, stride=2),
#                          nn.GELU(),
#                          nn.Conv1d(128, 256, kernel_size=3, stride=2),
#                          nn.GELU(),
#                          nn.Conv1d(256, 512, kernel_size=3, stride=2),
#                          nn.GELU(),
#                          nn.AdaptiveAvgPool1d(1),
#                          nn.Flatten(start_dim=-2),
#                          )

def simplenet():
    return nn.Sequential(nn.Conv1d(1, 64, kernel_size=7, stride=2),
                         nn.MaxPool1d(4, stride=2),
                         nn.AdaptiveAvgPool1d(1),
                         nn.Flatten(start_dim=-2),
                         )


if __name__ == '__main__':
    import torch

    # toy dataset
    batch_size = 32
    in_channels = 1
    L = 300  # length
    x = torch.rand(batch_size, in_channels, L)

    simplenet = simplenet()

    # forward
    out = simplenet(x)
    print(out.shape)
