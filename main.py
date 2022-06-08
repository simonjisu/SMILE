import argparse
from pathlib import Path

from src.dataset import MetaStockDataset
from src.utils import ARGProcessor
from src.model import MetaModel
from src.trainer import Trainer

def main(args):
    
    setting_file = args.exp
    if '.yml' not in args.exp:
        setting_file += '.yml'
    meta_args = ARGProcessor(setting_file=Path('./experiments') / setting_file)
    data_kwargs = meta_args.get_args(cls=MetaStockDataset)
    if not args.meta_test:
        meta_trainset = MetaStockDataset(meta_type='train', meta_train_stocks=None, **data_kwargs)
        model_kwargs = meta_args.get_args(cls=MetaModel)
        model = MetaModel(**model_kwargs)

        trainer_kwargs = meta_args.get_args(cls=Trainer)
        trainer = Trainer(**trainer_kwargs)
        trainer.meta_train(model, meta_trainset=meta_trainset)

    else:
        meta_trainset = MetaStockDataset(meta_type='train', meta_train_stocks=None, **data_kwargs)
        meta_test1 = MetaStockDataset(meta_type='test1', meta_train_stocks=meta_trainset.symbols, **data_kwargs)
        meta_test2 = MetaStockDataset(meta_type='test2', meta_train_stocks=meta_trainset.symbols, **data_kwargs)
        meta_test3 = MetaStockDataset(meta_type='test3', meta_train_stocks=meta_trainset.symbols, **data_kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='', type=str)
    parser.add_argument('--meta_test', action='store_true')