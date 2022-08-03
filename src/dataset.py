import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Generator
from tqdm import tqdm
from collections import defaultdict


def flatten(li: List[Any]) -> Generator:
    """flatten nested list
    ```python
    x = [[[1], 2], [[[[3]], 4, 5], 6], 7, [[8]], [9], 10]
    print(type(flatten(x)))
    # <generator object flatten at 0x00000212BF603CC8>
    print(list(flatten(x)))
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ```
    Args:
        li (List[Any]): any kinds of list
    Yields:
        Generator: flattened list generator
    """
    for ele in li:
        if isinstance(ele, list) or isinstance(ele, tuple):
            yield from flatten(ele)
        else:
            yield ele

class MetaStockDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            meta_type: str ='train', 
            data_dir: Path | str ='', 
            dtype: str ='kdd17', 
            stock_universe: int =0, 
            n_sample: int =5,
            n_support: int =6,  # n_support
            n_lag: int =1
        ):
        """
        dataset ref: https://arxiv.org/abs/1810.09936
        In this meta learning setting, we have 3 meta-test and 1 meta-train
        vertical = stocks, horizontal = time
                train      |    test
           A               |
           B   meta-train  |   meta-test
           C               |      (1)
           ----------------|-------------
           D   meta-test   |   meta-test
           E     (2)       |      (3)

        meta-test (1) same stock, different time
        meta-test (2) different stock, same time
        meta-test (3) different stock, different time
        use `valid_date` to split the train / test set

        the number of training stock was splitted with number of total stocks * 0.8
        we have 5 stock universe
        """
        super().__init__()
        # for debugging purpose
        self.labels_dict = {
            'fall': 0, 'rise': 1, 'unchange': 2 
        }
        # data config
        self.data_dir = Path(data_dir).resolve()
        ds_info = {
            # train: (Jan-01-2007 to Jan-01-2015)
            # val: (Jan-01-2015 to Jan-01-2016)
            # test: (Jan-01-2016 to Jan-01-2017)
            'kdd17': {
                'path': self.data_dir / 'kdd17/price_long_50',
                'date': self.data_dir / 'kdd17/trading_dates.csv',
                'universe': self.data_dir / 'kdd17/stock_universe.json', 
                'start_date': '2007-01-01',
                'train_date': '2015-01-01', 
                'val_date': '2016-01-01', 
                'test_date': '2017-01-01',
            },
            # train: (Jan-01-2014 to Aug-01-2015)
            # val: (Aug-01-2015 to Oct-01-2015)
            # test: (Oct-01-2015 to Jan-01-2016)
            'acl18': {
                'path': self.data_dir / 'stocknet-dataset/price/raw',
                'date': self.data_dir / 'stocknet-dataset/price/trading_dates.csv',
                'universe': self.data_dir / 'stocknet-dataset/stock_universe.json',
                'start_date': '2014-01-01',
                'train_date': '2015-08-01', 
                'val_date': '2015-10-01', 
                'test_date': '2016-01-01',
            }
        }
        ds_config = ds_info[dtype]
        
        self.meta_type = meta_type
        self.window_sizes = [5] # [5, 10, 15, 20]
        self.n_sample = n_sample
        assert n_support % 2 == 0, '`n_support must be a even number'
        self.n_support = n_support
        self.n_lag = n_lag
        self.stock_universe = str(stock_universe)

        # get data
        self.data = {}
        self.candidates = {}
        ps = list((ds_config['path']).glob('*.csv'))
        with ds_config['universe'].open('r') as file:
            universe_dict = json.load(file)
        
        universe_key = 'known' if (meta_type == 'train') or (meta_type == 'test1') else 'unknown'
        universe = universe_dict[self.stock_universe][universe_key]
        iterator = [p for p in ps if p.name.strip('.csv') in universe]
        for p in tqdm(iterator, total=len(iterator), desc=f'Processing data and candidates for {self.meta_type}'):    
            stock_symbol = p.name.rstrip('.csv')
            df_single = self.load_single_stock(p)
            if meta_type == 'train':
                cond = df_single['date'].between(ds_config['start_date'], ds_config['train_date'])
                df_single = df_single.loc[cond].reset_index(drop=True)
                labels_indices = self.get_candidates(df_single)
            else:
                if meta_type == 'test1':
                    df_single = df_single.loc[df_single['date'] > ds_config['val_date']].reset_index(drop=True)
                    labels_indices = self.get_candidates(df_single)
                elif meta_type == 'test2':
                    df_single = df_single.loc[df_single['date'] <= ds_config['train_date']].reset_index(drop=True)
                    labels_indices = self.get_candidates(df_single)
                elif meta_type == 'test3':
                    df_single = df_single.loc[df_single['date'] > ds_config['val_date']].reset_index(drop=True)
                    labels_indices = self.get_candidates(df_single)
                else:
                    raise KeyError('Error argument `meta_type`, should be in (train, test1, test2, test3)')

            self.data[stock_symbol] = df_single
            self.candidates[stock_symbol] = labels_indices

        self.n_stocks = len(universe)
        # self.reset_tensor_data()

    def load_single_stock(self, p: Path | str):
        def longterm_trend(x: pd.Series, k:int):
            return (x.rolling(k).sum().div(k*x) - 1) * 100

        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        if 'Unnamed' in df.columns:
            df.drop(columns=df.columns[7], inplace=True)
        if 'Original_Open' in df.columns:
            df.rename(columns={'Original_Open': 'Open', 'Open': 'Adj Open'}, inplace=True)

        # Open, High, Low
        z1 = (df.loc[:, ['Open', 'High', 'Low']].div(df['Close'], axis=0) - 1).rename(
            columns={'Open': 'open', 'High': 'high', 'Low': 'low'}) * 100
        # Close
        z2 = df[['Close']].pct_change().rename(columns={'Close': 'close'}) * 100
        # Adj Close
        z3 = df[['Adj Close']].pct_change().rename(columns={'Adj Close': 'adj_close'}) * 100

        z4 = []
        for k in [5, 10, 15, 20, 25, 30]:
            z4.append(df[['Adj Close']].apply(longterm_trend, k=k).rename(columns={'Adj Close': f'zd{k}'}))

        df_pct = pd.concat([df['Date'], z1, z2, z3] + z4, axis=1).rename(columns={'Date': 'date'})
        cols_max = df_pct.columns[df_pct.isnull().sum() == df_pct.isnull().sum().max()]
        df_pct = df_pct.loc[~df_pct[cols_max].isnull().values, :]

        # from https://arxiv.org/abs/1810.09936
        # Examples with movement percent ≥ 0.55% and ≤ −0.5% are 
        # identified as positive and negative examples, respectively
        df_pct['label'] = self.labels_dict['unchange']
        df_pct.loc[(df_pct['close'] >= 0.55), 'label'] = self.labels_dict['rise']
        df_pct.loc[(df_pct['close'] <= -0.5), 'label'] = self.labels_dict['fall']
        
        return df_pct

    def get_candidates(self, df):
        """support candidates"""
        checks = [self.labels_dict['fall'], self.labels_dict['rise']]
        condition = np.isin(df['label'], checks)
        labels_indices = df.index[condition].to_numpy()
        return labels_indices

    # def get_candidates(self, df):
    #     """support candidates"""
    #     condition = df['label'].rolling(2).apply(self.check_func).shift(-self.n_lag).fillna(0.0).astype(bool)
    #     labels_indices = df.index[condition].to_numpy()
    #     return labels_indices

    # def check_func(self, x):
    #     checks = [self.labels_dict['fall'], self.labels_dict['rise']]
    #     return np.isin(x.values[0], checks) and np.isin(x.values[1], checks)

    @property
    def symbols(self):
        return list(self.data.keys())

    # Normal Generator
    # def init_data(self, tasks, device: None | str=None):
    #     self.tensor_data = self.map_to_tensor(tasks, device=device)
        
    # def generate_all(self):
    #     all_tasks = defaultdict()
    #     for window_size in self.window_sizes:
    #         tasks = defaultdict(list)
    #         for symbol in self.symbols:
    #             data = self.generate_support_query(symbol, window_size)
    #             for k, v in data.items():
    #                 tasks[k].extend(v)
    #         all_tasks[window_size] = tasks
        
    #     self.all_tasks = all_tasks

    # def generate_support_query(self, symbol: str, window_size: int):
    #     df_stock = self.data[symbol]
    #     labels_indices = self.candidates[symbol]
    #     y_s = labels_indices[labels_indices >= window_size]
    #     y_ss = y_s-window_size
    #     support, support_labels = self.generate_data(df_stock, y_start=y_ss, y_end=y_s)

    #     y_q = y_s + self.n_lag
    #     y_qs = y_q - window_size
    #     query, query_labels = self.generate_data(df_stock, y_start=y_qs, y_end=y_q)

    #     return {
    #         'support': support, 'support_labels': support_labels,
    #         'query': query, 'query_labels': query_labels
    #     }


    # Meta Generator
    def generate_tasks(self):
        all_tasks = defaultdict()
        for window_size in self.window_sizes:
            tasks = self.generate_tasks_per_window_size(window_size)
            all_tasks[window_size] = tasks
        return all_tasks

    def generate_tasks_per_window_size(self, window_size: int):
        # Query: (n_sample, 1, T, I) / Support: (n_sample, n_support, T, I)
        # Query Labels: (n_sample,) / Support Labels: (n_sample, n_support)
        tasks = defaultdict(list)
        for symbol in self.symbols:
            data = self.generate_task_per_window_size_and_single_stock(symbol, window_size)
            for k, v in data.items():
                tasks[k].append(v)

        return tasks

    def generate_task_per_window_size_and_single_stock(self, symbol: str, window_size: int):
        df_stock = self.data[symbol]
        # filter out unpossible candidates
        labels_indices = self.candidates[symbol]
        labels_candidates = labels_indices[labels_indices >= window_size]
        idx = self.get_possible_idx(df_stock, labels_candidates)
        labels_candidates = labels_candidates[idx:]

        # sample query
        # Query: (n_sample, 1, T, I)
        # Query Labels: (n_sample, 1)
        y_q = np.array(np.random.choice(labels_candidates, size=(self.n_sample,), replace=False))
        print(symbol, y_q)
        y_qs = y_q - window_size
        query, query_labels = self.generate_data(df_stock, y_start=y_qs, y_end=y_q)
        query = np.expand_dims(query, 1)
        query_labels = np.expand_dims(query_labels, 1)
        # sample support
        # decided by n_support
        # Support: (n_sample, n_support, T, I)
        # Support Labels: (n_sample, n_support)
        support = []
        support_labels = []
        for q in y_q:
            q_idx = np.arange(len(labels_candidates))[labels_candidates == q][0]
            rise, fall = self.get_rise_fall(df_stock, labels_candidates, idx=q_idx)
            y_s = np.concatenate([fall, rise])
            y_ss = y_s - window_size
            data_s, label_s = self.generate_data(df_stock, y_start=y_ss, y_end=y_s)
            support.append(data_s)
            support_labels.append(label_s)
        support = np.array(support)
        support_labels = np.array(support_labels)

        return {
            'support': support, 'support_labels': support_labels,
            'query': query, 'query_labels': query_labels
        }

    def generate_data(self, df: pd.DataFrame, y_start: np.ndarray, y_end: np.ndarray):
        # generate mini task
        inputs = []
        labels = []
        for i, j in zip(y_start, y_end):
            inputs.append(df.loc[i:j-1].to_numpy()[:, 1:-1].astype(np.float64))
            labels.append(df.loc[j].iloc[-1].astype(np.uint8))

        # inputs: (n_sample, y_end-y_start, n_in), labels: (n_sample,)
        return np.array(inputs), np.array(labels)

    def get_possible_idx(self, df: pd.DataFrame, labels_candidates: np.ndarray):
        i = 0
        while i < len(labels_candidates):
            rise, fall = self.get_rise_fall(df, labels_candidates, idx=i)
            if len(rise) + len(fall) == 4:
                break
            else:
                i += 1
        return i

    def get_rise_fall(self, df: pd.DataFrame, labels_candidates: np.ndarray, idx: int):
        df_check = df.loc[labels_candidates[:idx], 'label'].sort_index(ascending=False)
        rise = df_check.index[df_check == self.labels_dict['rise']][:(self.n_support // 2)].to_numpy()
        fall = df_check.index[df_check == self.labels_dict['fall']][:(self.n_support // 2)].to_numpy()
        return rise, fall

    def map_to_tensor(self, tasks, device: None | str=None):
        if device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(device)
        tensor_tasks = {}
        for k, v in tasks.items():
            tensor_fn = torch.LongTensor if 'labels' in k else torch.FloatTensor
            tensor = tensor_fn(np.array(v))
            if ('labels' not in k) and tensor.ndim == 1:
                tensor = tensor.view(1, -1)
            tensor_tasks[k] = tensor.to(device)
        return tensor_tasks
    
    def iter_json(self, task, n_iter):
        for i in range(n_iter):
            data = {}
            for k in task.keys():
                data[k] = task[k][i]
            yield data

    # def __len__(self):
    #     if self.tensor_data is None:
    #         raise ValueError('You Need to generate data first, please call the function')
    #     else:
    #         return self.n_stocks

    # def __getitem__(self, index):
    #     t = {}
    #     for k, v in self.tensor_data.items():
    #         t[k] = v[index]        
    #     return t

    # def set_tensor_data(self, data):
    #     self.tensor_data = data

    # def reset_tensor_data(self):
    #     self.tensor_data = None