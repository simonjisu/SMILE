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

    def get_single_stock_instance(self, idx):
        instance = {}
        for key, value in self.items():
            instance[key] = value[idx]
        return instance

    def __len__(self):
        return self.n_stocks

    def __iter__(self):
        idxes = np.arange(len(self))
        np.random.shuffle(idxes)
        for idx in idxes:
            yield self.get_single_stock_instance(idx)


class MetaStockDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            meta_type: str ='train', 
            data_dir: Path | str ='', 
            dtype: str ='kdd17', 
            stock_universe: int =0, 
            n_sample: int =5,
            n_support: int =4, 
            n_query: int =1,
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
            n_sample (int, optional): Number of sample to train. Defaults to 5.
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
        self.window_sizes = window_sizes
        self.n_sample = n_sample
        assert n_support % 2 == 0, '`n_support must be a even number'
        self.n_support = n_support
        self.n_query = n_query
        self.n_lag = n_lag
        self.stock_universe = str(stock_universe)
        self.n_classes = n_classes

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
        all_tasks = defaultdict()
        for window_size in self.window_sizes:
            tasks = self.generate_tasks_per_window_size(window_size)
            all_tasks[window_size] = StockDataDict(tasks, window_size=window_size)
        return all_tasks

    def generate_tasks_per_window_size(self, window_size: int):
        """
        Args:
            window_size (int): window size

        Returns:
            tasks: Dict[str, List[np.ndarray]]
        """
        
        tasks = defaultdict(list)
        for symbol in self.symbols:
            data = self.generate_task_per_window_size_and_single_stock(symbol, window_size)
            for k, v in data.items():
                tasks[k].append(v)
        return tasks

    def check_condition(self, array):
        cond1 = array.sum() >= self.n_classes
        cond2 = np.isin(array, self.labels_dict['fall']).sum() >= self.n_support
        cond3 = np.isin(array, self.labels_dict['rise']).sum() >= self.n_support
        return cond1 & cond2 & cond3

    def generate_task_per_window_size_and_single_stock(self, symbol: str, window_size: int):
        """Generate tasks per single stock and window size
        
        For each single stock and single window size `T`, 
        first, choose target data in `t_start`:`t_end-1`
        
        Data: n_classes=2, n_query=1. It chooses the data by latest rise & fall 
            in `t_start`:`t_end-1`, to guess `t_end` step
        - Query: (n_sample, n_query*n_classes, T, I)
        - Support: (n_sample, n_support*n_classes, T, I)
        
        Labels: both with `t_e` data as label
        - Query Labels: (n_sample,)
        - Support Labels: (n_sample,) 

        
        (x) Masks: target mask for the input data of the classes
        - Query Masks: (n_sample, n_query*n_classes)
        - Support Masks: (n_sample, n_support*n_classes) 

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

        # index candidates for queries
        # satisfied condition label index | smallest support index | smallest query index
        candidates = labels_indices[(i+2):]  

        data = dict(
            query = [],
            query_labels = [],
            # query_masks = [],
            support = [],
            support_labels = [],
            # support_masks = []
        )

        y_q = np.random.choice(candidates, size=(self.n_sample,), replace=False)   # index in the dataframe
        for q_target in y_q:
            # Queries
            q_idx = np.arange(len(labels_indices))[labels_indices == q_target][0]  # get the index of label data
            q_fall, q_rise = self.get_rise_fall(df_stock, labels_indices, idx=q_idx, n_select=self.n_query)
            q_end = np.concatenate([q_fall, q_rise])
            q_start = q_end - window_size
            q_data, q_mask = self.generate_data(df_stock, y_start=q_start, y_end=q_end)

            data['query'].append(q_data)
            # data['query_masks'].append(q_mask)
            data['query_labels'].append(df_stock.loc[q_target, 'label'])

            # Supports
            s_idx = q_idx - self.n_lag
            s_target = labels_indices[s_idx]
            s_fall, s_rise = self.get_rise_fall(df_stock, labels_indices, idx=s_idx, n_select=self.n_support)
            s_end = np.concatenate([s_fall, s_rise])
            s_start = s_end - window_size
            s_data, s_mask = self.generate_data(df_stock, y_start=s_start, y_end=s_end)
            
            data['support'].append(s_data)
            # data['support_masks'].append(s_mask)
            data['support_labels'].append(df_stock.loc[s_target, 'label'])

        for k, v in data.items():
            data[k] = np.array(v)

        return data

    def generate_data(self, df: pd.DataFrame, y_start: np.ndarray, y_end: np.ndarray):
        """
        n_data = n_support*n_classes or n_query*n_classes
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