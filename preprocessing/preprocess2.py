import os
from typing import List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from sklearn.preprocessing import StandardScaler


class GBVBDataset(Dataset):
    def __init__(self,
                 ts_len: int,
                 test_ratio: float,
                 transform: Compose = None,
                 kind: str = 'train',
                 ts_len_threshold: int = 10,
                 **kwargs):
        super().__init__()
        self.transform = transform
        self.ts_len = ts_len

        self.GB_tickers = self._get_tickers('GB')  # obtained based on fnames in the corresponding directory
        self.VB_tickers = self._get_tickers('VB')  # obtained based on fnames in the corresponding directory
        self.GB_size = len(self.GB_tickers)
        self.VB_size = len(self.VB_tickers)

        # keep tickers with zero(or near-zero)-long timeseries
        VB_tickers_ = []
        for vb_ticker in self.VB_tickers:
            ts_len = pd.read_csv(os.path.join('dataset', 'dataset_VB', f'{vb_ticker}-ts.csv')).shape[0]
            if ts_len > ts_len_threshold:
                VB_tickers_.append(vb_ticker)
        self.VB_tickers = VB_tickers_

        # split into train / test
        if kind == 'train':
            GB_idx = np.random.rand(len(self.GB_tickers)) > test_ratio
            VB_idx = np.random.rand(len(self.VB_tickers)) > test_ratio
        elif kind == 'test':
            GB_idx = np.random.rand(len(self.GB_tickers)) <= test_ratio
            VB_idx = np.random.rand(len(self.VB_tickers)) <= test_ratio
        else:
            raise ValueError
        self.GB_tickers = np.array(self.GB_tickers)[:, None][GB_idx].reshape(-1)
        self.VB_tickers = np.array(self.VB_tickers)[:, None][VB_idx].reshape(-1)

        self.merged_tickers = np.concatenate((self.GB_tickers, self.VB_tickers), axis=-1)
        np.random.shuffle(self.merged_tickers)

        self.df_info_GBVB_clean = pd.read_csv(os.path.join('dataset', 'info_GBVB_clean.csv'), index_col=0)  # cleaned
        self.df_info_GB_clean = self.df_info_GBVB_clean[self.df_info_GBVB_clean['bond_type'] == 'GB']
        self.df_info_VB_clean = self.df_info_GBVB_clean[self.df_info_GBVB_clean['bond_type'] == 'VB']
        for i in list(self.df_info_VB_clean.index):
            if i not in self.VB_tickers:
                self.df_info_VB_clean = self.df_info_VB_clean.drop(index=i)

        self._remove_bond_type_column()

        self.scaler = StandardScaler()
        self.scaler.fit(self.df_info_GBVB_clean.values)

        # self._len = len(self.GB_tickers)
        self._len = len(self.merged_tickers)

    def _get_tickers(self, kind: str) -> List[str]:
        if kind == 'GB':
            tickers = os.listdir(os.path.join('dataset', 'dataset_GB'))
        elif kind == 'VB':
            tickers = os.listdir(os.path.join('dataset', 'dataset_VB'))
        else:
            raise ValueError
        tickers = list(set([ticker.split('-')[0] for ticker in tickers]))
        return tickers

    def _remove_bond_type_column(self, ):
        self.df_info_GBVB_clean = self.df_info_GBVB_clean.drop(['bond_type'], axis=1)
        self.df_info_GB_clean = self.df_info_GB_clean.drop(['bond_type'], axis=1)
        self.df_info_VB_clean = self.df_info_VB_clean.drop(['bond_type'], axis=1)

    def _get_closest_VB_ticker(self, GB_ticker: str):
        """
        get the closest VB sample to the target GB sample w.r.t info variables.
        - criterion: Euclidean distance
        :return:
        """
        gb_var = self.df_info_GBVB_clean.loc[GB_ticker].values.reshape(1, -1)
        gb_var = self.scaler.transform(gb_var)

        # compute distances with VB samples
        df_info_VB_clean_ = self.scaler.transform(self.df_info_VB_clean.values)
        dist = (df_info_VB_clean_ - gb_var) ** 2
        dist = np.sqrt(np.sum(dist, axis=1))
        idx = np.argmin(dist)

        # select the closest
        VB_ticker = self.df_info_VB_clean.iloc[idx, :].name
        return VB_ticker

    def _scale_ts(self, ts):
        ts = np.arcsinh(ts)
        min_, max_ = np.min(ts), np.max(ts)

        if (max_ - min_) < 1e-1:
            pass
        else:
            ts = (ts - min_) / (max_ - min_)
            # ts = (ts - np.mean(ts)) / np.std(ts + 1e-4)
            # ts = np.arcsinh(ts)
        return ts

    def _clip_timeseries(self, ts):
        if self.ts_len >= len(ts):
            # ts_ = np.random.rand(self.ts_len) * 0.1
            ts_ = np.zeros(self.ts_len)
            ts_[:len(ts)] = ts
            ts = ts_.copy()
        else:
            rand_ts_idx = np.random.randint(0, len(ts) - self.ts_len)
            ts = ts[rand_ts_idx:rand_ts_idx + self.ts_len]
        return ts

    def __getitem__(self, idx):
        # select one GB timeseries sample
        idx_pull = np.random.randint(0, len(self.GB_tickers))
        GB_ticker = self.GB_tickers[idx_pull]
        GB_ts = pd.read_csv(os.path.join('dataset', 'dataset_GB', f'{GB_ticker}-ts.csv'))

        # select the closest VB timeseries sample to the selected GB sample
        # closest w.r.t info-vars
        closest_VB_ticker = self._get_closest_VB_ticker(GB_ticker)
        closest_VB_ts = pd.read_csv(os.path.join('dataset', 'dataset_VB', f'{closest_VB_ticker}-ts.csv'))

        # randomly select a sample regardless of GB and VB; `class weight` must be considered in this case.
        # mr_idx = np.random.randint(0, len(self.merged_tickers))  # mr: merged random
        MR_ticker = self.merged_tickers[idx]
        dirname = 'dataset_GB' if (MR_ticker in self.GB_tickers) else 'dataset_VB'
        MR_ts = pd.read_csv(os.path.join('dataset', dirname, f'{MR_ticker}-ts.csv'))
        label = 0 if (dirname == 'dataset_VB') else 1  # {0: VB, 1: GB}

        # convert dtypes
        GB_ts = (GB_ts['High'] - GB_ts['Low']).values
        closest_VB_ts = (closest_VB_ts['High'] - closest_VB_ts['Low']).values
        MR_ts = (MR_ts['High'] - MR_ts['Low']).values
        label = torch.Tensor([label]).long()

        # clip timeseries
        GB_ts = self._clip_timeseries(GB_ts)
        closest_VB_ts = self._clip_timeseries(closest_VB_ts)
        MR_ts = self._clip_timeseries(MR_ts)

        # scale timeseries (should be located after clip)
        GB_ts = self._scale_ts(GB_ts)
        closest_VB_ts = self._scale_ts(closest_VB_ts)
        MR_ts = self._scale_ts(MR_ts)

        # add channel dim on timeseries
        GB_ts = np.expand_dims(GB_ts, axis=0)
        closest_VB_ts = np.expand_dims(closest_VB_ts, axis=0)
        MR_ts = np.expand_dims(MR_ts, axis=0)

        # convert timeseries dtype to Tensor
        GB_ts = torch.from_numpy(GB_ts).float()
        closest_VB_ts = torch.from_numpy(closest_VB_ts).float()
        MR_ts = torch.from_numpy(MR_ts).float()
        label = torch.Tensor([label]).long()

        return (GB_ts, closest_VB_ts), (MR_ts, label)

    def __len__(self):
        return self._len


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn as nn

    from utils import root_dir

    np.random.seed(0)

    os.chdir(root_dir)
    print(os.getcwd())

    dataset = GBVBDataset(100, 0.2, kind='train')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # fetch a batch
    batch = next(iter(data_loader))
    (GB_ts, closest_VB_ts), (MR_ts, label) = batch
    print('GB_ts.shape:', GB_ts.shape)
    print('closest_VB_ts.shape', closest_VB_ts.shape)
    print('MR_ts.shape:', MR_ts.shape)
    print('label.shape:', label.shape)
    print('np.unique(label):', np.unique(label))
    print()

    # plot
    # n_samples = 10
    # for i in range(n_samples):
    #     batch_idx = i
    #     channel_idx = 0
    #     plt.figure(figsize=(2 * 3, 8))
    #     plt.subplot(3, 1, 1)
    #     plt.plot(GB_ts[batch_idx, channel_idx, :])
    #     plt.subplot(3, 1, 2)
    #     plt.plot(closest_VB_ts[batch_idx, channel_idx, :])
    #     plt.subplot(3, 1, 3)
    #     plt.plot(MR_ts[batch_idx, channel_idx, :])
    #     plt.show()

    # validity check
    model = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, stride=2),
                          nn.GELU(),
                          nn.Conv1d(64, 64, kernel_size=3, stride=2),
                          nn.GELU(),
                          nn.AdaptiveAvgPool1d(1),
                          nn.Flatten(start_dim=-2),
                          nn.Linear(64, 2)
                          )
    n_epochs = 20
    criterion = nn.CrossEntropyLoss()
    for batch in data_loader:
        (GB_ts, closest_VB_ts), (MR_ts, label) = batch
        label = label.reshape(-1)
        out = model(MR_ts)
        loss = criterion(out, label)
        print(loss)
