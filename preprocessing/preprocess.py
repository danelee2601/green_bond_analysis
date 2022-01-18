from torch.utils.data import Dataset
from torchvision.transforms import Compose


class GBVBDataset(Dataset):
    def __init__(self, transform: Compose, kind: str, **kwargs):
        super().__init__()

        self.transform = transform
        self.kind = kind

        # TODO_: need to decide whether to make an index csv file and load the data or load the entire dataset.
        # TODO_: that'd depend on size of the dataset.

        self._len = None

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self._len