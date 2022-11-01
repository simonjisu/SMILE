import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Generator
from tqdm import tqdm
from collections import Counter


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

class StockDataDict(dict):
    def __init__(self, data, window_size):
        self.window_size = window_size
        self._set_state(f'numpy')
        for k, v in data.items():
            data[k] = np.array(v)
        
        self.n_stocks = len(v)
        super().__init__(data)
    
    def tensor_fn(self, value, key):
        if '_' in key:
            return torch.LongTensor(value)
        else:
            return torch.FloatTensor(value)

    def _set_state(self, state: str):
        self.state = state

    def to(self, device: None | str=None):
        if device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(device)
        self._set_state(f'tensor.{device}')
        for key in self.keys():
            value = self.__getitem__(key)
            tvalue = self.tensor_fn(value, key)
            self.__setitem__(key, tvalue.to(device)) 
        
    def numpy(self):
        self._set_state('numpy')
        for key in self.keys():
            tvalue = self.__getitem__(key)
            if not isinstance(tvalue, np.ndarray): 
                self.__setitem__(key, tvalue.detach().numpy())

    def __str__(self):
        s = f'StockDataDict(T={self.window_size}, {self.state})\n'
        for i, key in enumerate(self.keys()):
            value = self.__getitem__(key)
            s += f'- {key}: {value.shape}'
            s += '' if i == len(self.keys())-1 else '\n'
        return s

    def __repr__(self):
        return self.__str__()

    def get_data(self):
        return dict(self.items())

    def __len__(self):
        return self.n_stocks

class MetaStockDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            meta_type: str ='train', 
            data_dir: Path | str ='', 
            dtype: str ='kdd17', 
            batch_size: int =64,
            n_support: int =4, 
            n_lag: int =1,
            n_classes: int =2,
            window_sizes: List[int] =[5, 10, 15, 20]
        ):    
        """dataset ref: https://arxiv.org/abs/1810.09936

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

        the number of training stock was splitted with number of total stocks * 0.8.
        we have 5 stock universe

        Args:
            meta_type (str, optional): _description_. Defaults to 'train'.
            data_dir (Path | str, optional): _description_. Defaults to ''.
            dtype (str, optional): _description_. Defaults to 'kdd17'.
            stock_universe (int, optional): _description_. Defaults to 0.
            batch_size (int, optional): Batch size. Number of stock x Number of timestamp that is aviable for each window size. Defaults to 64.
            n_support (int, optional): Number of support. Defaults to 4.
            n_classes (int, optional): _description_. Defaults to 2.
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
                'valid_date': '2016-01-01', 
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
                'valid_date': '2015-10-01', 
                'test_date': '2016-01-01',
            }
        }
        ds_config = ds_info[dtype]
        
        self.meta_type = meta_type
        self.window_sizes = window_sizes
        self.batch_size = batch_size
        self.n_support = n_support
        self.n_lag = n_lag
        self.n_classes = n_classes

        # get data
        self.data = {}
        self.candidates = {}
        ps = list((ds_config['path']).glob('*.csv'))
        with ds_config['universe'].open('r') as file:
            universe_dict = json.load(file)
        
        # meta_type: train / valid1: valid-time, valid2: valid-stock, valid3: valid-mix / test1, test2, test3
        if meta_type in ['train', 'valid-time', 'test-time']:
            universe = universe_dict['train']
        elif meta_type in ['valid-stock', 'valid-mix']:
            universe = universe_dict['valid']
        elif meta_type in ['test-stock', 'test-mix']:
            universe = universe_dict['test']
        else:
            raise KeyError('Error argument `meta_type`, should be in (train, valid-time, valid-stock, valid-mix, test-time, test-stock, test-mix)')

        if meta_type in ['train', 'valid-stock', 'test-stock']:
            date1 = ds_config['start_date']
            date2 = ds_config['train_date']
        elif meta_type in ['valid-time', 'valid-mix']:
            date1 = ds_config['train_date']
            date2 = ds_config['valid_date']
        elif meta_type in ['test-time', 'test-mix']:
            date1 = ds_config['valid_date']
            date2 = ds_config['test_date']
        else:
            raise KeyError('Error argument `meta_type`, should be in (train, valid-time, valid-stock, valid-mix, test-time, test-stock, test-mix)')

        iterator = [p for p in ps if p.name.strip('.csv') in universe]
        for p in tqdm(iterator, total=len(iterator), desc=f'Processing data and candidates for {self.meta_type}'):    
            stock_symbol = p.name.rstrip('.csv')
            df_single = self.load_single_stock(p)
            cond = df_single['date'].between(date1, date2)
            df_single = df_single.loc[cond].reset_index(drop=True)
            labels_indices = self.get_candidates(df_single)
            
            self.data[stock_symbol] = df_single
            self.candidates[stock_symbol] = labels_indices

        self.n_stocks = len(universe)
        self.reset_q_idx_dist()

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
        checks = [self.labels_dict['fall'], self.labels_dict['rise']]
        condition = np.isin(df['label'], checks)
        labels_indices = df.index[condition].to_numpy()
        return labels_indices

    @property
    def symbols(self):
        return list(self.data.keys())

    # Meta Generator
    def generate_tasks(self):
        """
        Data: n_classes=2. It chooses the data by latest rise & fall 
            in `t_start`:`t_end-1`, to guess `t_end` step
        - Query: (batch_size, 1, T, I)
        - Support: (batch_size, n_support*n_classes, T, I)
        
        Labels: both with `t_e` data as label
        - Query Labels: (batch_size,)
        - Support Labels: (batch_size*n_support*n_classes,) 
        """
        all_tasks = dict(
            query = [],
            query_labels = [],
            support = [],
            support_labels = [],
        )
        window_size = np.random.choice(self.window_sizes)  # window_sizes: list
        for i in range(self.batch_size):
            symbol = np.random.choice(self.symbols)
            data = self.generate_task_per_window_size_and_single_stock(symbol, window_size)
            for k, v in data.items():
                if 'labels' in k:
                    all_tasks[k].extend(v.tolist())
                else:
                    all_tasks[k].append(v)
            
        return StockDataDict(all_tasks, window_size=window_size)

    def check_condition(self, array):
        cond1 = array.sum() >= self.n_classes
        cond2 = np.isin(array, self.labels_dict['fall']).sum() >= self.n_support
        cond3 = np.isin(array, self.labels_dict['rise']).sum() >= self.n_support
        return cond1 & cond2 & cond3

    def generate_task_per_window_size_and_single_stock(self, symbol: str, window_size: int):
        """Generate tasks per single stock and window size
        
        For each single stock and single window size `T`, 
        first, choose target data in `t_start`:`t_end-1`

        Args:
            symbol (str): stock symbol
            window_size (int): window size

        Returns:
            data: dictionary with 
            keys=['query', 'query_labels', 'support', 'support_labels']
        """
        df_stock = self.data[symbol]
        # filter out unpossible candidates
        labels_indices = self.candidates[symbol] 
        labels_indices = labels_indices[labels_indices >= window_size]

        for i in range(len(labels_indices)):
            array = df_stock.loc[labels_indices, 'label'].loc[:(labels_indices[i])].to_numpy()
            if self.check_condition(array):
                break

        # satisfied condition label index 
        candidates = labels_indices[(i+1):]  # query candidates

        data = dict(
            query = None,
            query_labels = None,
            support = None,
            support_labels = None,
        )

        q_target = np.random.choice(candidates)   # index in the dataframe
        # for q_target in y_q:
            # Queries
        q_idx = np.arange(len(labels_indices))[labels_indices == q_target][0]  # get the index of label data
        self.update_q_idx_dist(q_target)
        q_end = np.array([q_target]) 
        q_start = q_end - window_size
        q_data, q_labels = self.generate_data(df_stock, y_start=q_start, y_end=q_end)
        
        data['query'] = q_data
        data['query_labels'] = q_labels  # np.array: (1,)

        # Supports
        s_fall, s_rise = self.get_rise_fall(df_stock, labels_indices, idx=q_idx, n_select=self.n_support)
        s_end = np.concatenate([s_fall, s_rise])
        s_start = s_end - window_size
        s_data, s_labels = self.generate_data(df_stock, y_start=s_start, y_end=s_end)
        
        data['support'] = s_data
        data['support_labels'] = s_labels  # np.array: (N*K)

        return data

    def generate_data(self, df: pd.DataFrame, y_start: np.ndarray, y_end: np.ndarray):
        """
        n_data = n_support*n_classes or 1
        inputs: (n_data, win_size, n_in)
        labels: (n_data,)

        Args:
            df (pd.DataFrame): stock dataframe
            y_start (np.ndarray): start date index
            y_end (np.ndarray): end date index

        """
        # generate mini task
        inputs = []
        labels = []
        for i, j in zip(y_start, y_end):
            inputs.append(df.iloc[i:j].to_numpy()[:, 1:-1].astype(np.float64))
            labels.append(df.iloc[j, -1].astype(np.uint8))

        return np.array(inputs), np.array(labels)

    def get_rise_fall(self, df: pd.DataFrame, labels_indices: np.ndarray, idx: int, n_select: int):
        df_check = df.loc[labels_indices[:idx], 'label'].sort_index(ascending=False)
        rise = df_check.index[df_check == self.labels_dict['rise']][:n_select].to_numpy()
        fall = df_check.index[df_check == self.labels_dict['fall']][:n_select].to_numpy()
        return fall, rise

    def update_q_idx_dist(self, q_target):
        self.q_dist[q_target] += 1

    def reset_q_idx_dist(self):
        self.q_dist = Counter()
    