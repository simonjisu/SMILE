import sys
import argparse
from pathlib import Path

main_path = Path('..').resolve()
sys.path.append(str(main_path))

from src.dataset import MetaStockDataset
from src.utils import ARGProcessor
import numpy as np

def main(args):
    setting_file = args.exp
    if '.yml' not in args.exp:
        setting_file += '.yml'
    setting_file = Path('.') / setting_file
    
    meta_args = ARGProcessor(setting_file=setting_file)
    data_kwargs = meta_args.get_args(cls=MetaStockDataset)
    meta_train = MetaStockDataset(meta_type='train', **data_kwargs)
    meta_test1 = MetaStockDataset(meta_type='test1', **data_kwargs)
    meta_test2 = MetaStockDataset(meta_type='test2', **data_kwargs)
    meta_test3 = MetaStockDataset(meta_type='test3', **data_kwargs)

    for i, m_test in enumerate([meta_train, meta_test1, meta_test2, meta_test3]):
        for window_size in m_test.window_sizes:
            if i == 0:
                print(f'Meta Train Datset{i} - win_size={window_size}')
            else:
                print(f'Meta Test Datset{i} - win_size={window_size}')
            n_candidates = []

            for symbol in m_test.symbols:
                labels_indices = m_test.candidates[symbol]
                labels_candidates = labels_indices[labels_indices >= window_size]
                n_candidates.append(len(labels_candidates))
            print(f'Num Symbols: {len(n_candidates)}, Mean: {np.mean(n_candidates)}, Std: {np.std(n_candidates)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='', type=str)
    args = parser.parse_args()
    main(args)