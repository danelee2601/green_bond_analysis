import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import root_dir


def run(target_GB_ticker):
    # fetch green bond info with issue dates
    df = pd.read_excel(root_dir.joinpath('dataset', 'GBs_n_2255.xlsx'))
    df = df[['Exchange Ticker', 'Issue Date']]

    # search for a target ticker
    issue_date = df[df['Exchange Ticker'] == target_GB_ticker]['Issue Date']
    print('issue_date:\n', issue_date, '\n')
    if issue_date.shape[0] > 1:
        issue_date = issue_date.iloc[[0]]
        print('corrected) issue_date:\n', issue_date, '\n')

    # fetch corresponding timeseries
    ts = pd.read_csv(root_dir.joinpath('dataset', 'dataset_GB', f'{target_GB_ticker}-ts.csv'))
    ts_volat = ts['High'] - ts['Low']
    ts_date = ts['Date']
    print('ts_date:', ts_date)
    target_ts_date = ts_date.apply(lambda x: x == issue_date).values.astype(int)
    print(np.sum(target_ts_date))
    plt.figure(figsize=(10, 3))
    plt.title(f'{target_GB_ticker}')
    plt.plot(ts_date, ts_volat)
    plt.scatter(ts_date, ts_volat,
                c=target_ts_date,
                s=target_ts_date * 100,
                cmap='coolwarm',
                )
    plt.xticks(ts_date[::20], rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run(target_GB_ticker='ENPH')
