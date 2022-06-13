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
    setting_file = Path('./experiments') / setting_file
    meta_args = ARGProcessor(setting_file=setting_file)
    data_kwargs = meta_args.get_args(cls=MetaStockDataset)
    if not args.meta_test:
        assert args.exp_num == 0, 'Should not give `exp_num` argument when it is training'
        meta_trainset = MetaStockDataset(meta_type='train', **data_kwargs)
        model_kwargs = meta_args.get_args(cls=MetaModel)
        model = MetaModel(**model_kwargs)

        trainer_kwargs = meta_args.get_args(cls=Trainer)
        trainer = Trainer(**trainer_kwargs)

        # meta train
        trainer.meta_train(model, meta_trainset=meta_trainset)
        meta_args.save(trainer.exp_dir / 'settings.yml', meta_args.kwargs)

        # meta test 
        meta_test1 = MetaStockDataset(meta_type='test1', **data_kwargs)
        meta_test2 = MetaStockDataset(meta_type='test2', **data_kwargs)
        meta_test3 = MetaStockDataset(meta_type='test3', **data_kwargs)
        trainer.meta_test(model=model, meta_dataset=meta_test1, exp_num=trainer.exp_num)
        trainer.meta_test(model=model, meta_dataset=meta_test2, exp_num=trainer.exp_num)
        trainer.meta_test(model=model, meta_dataset=meta_test3, exp_num=trainer.exp_num)
        
    else:
        # only meta test for all/specific experiments
        meta_test1 = MetaStockDataset(meta_type='test1', **data_kwargs)
        meta_test2 = MetaStockDataset(meta_type='test2', **data_kwargs)
        meta_test3 = MetaStockDataset(meta_type='test3', **data_kwargs)
        model_kwargs = meta_args.get_args(cls=MetaModel)
        model = MetaModel(**model_kwargs)

        trainer_kwargs = meta_args.get_args(cls=Trainer)
        trainer = Trainer(**trainer_kwargs)
        # trainer will find the experiment results by the combination of 
        # `exp_name` in setting file and `exp_num` in argument
        # if args.exp_num is zero, means test for all and only record a file in `all_results.csv`
        if args.exp_num == 0:
            exp_nums = sorted(map(lambda x: int(x.name.split('_')[-1]), trainer.log_dir.glob(f'{trainer.exp_name}_*')))
            record_file = open('./all_results.md', 'w', encoding='utf-8')
            print('| Experiment | Test Type | Test Accuracy | Test Loss |', file=record_file)
            
        else:
            exp_nums = [args.exp_num]

        for exp_num in exp_nums:
            print(f'===== {trainer.exp_name}_{exp_num} =====')
            for meta_test in [meta_test1, meta_test2, meta_test3]:
                test_acc, test_loss = trainer.meta_test(
                    model=model, 
                    meta_dataset=meta_test, 
                    exp_num=exp_num,
                    n_test=args.n_test
                )
                if args.exp_num == 0:
                    print(f'| {trainer.exp_name}_{exp_num} | {meta_test.meta_type} | {test_acc:.4f} | {test_loss:.4f} |', file=record_file)
        
        if args.exp_num == 0:
            record_file.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='', type=str)
    parser.add_argument('--meta_test', action='store_true')
    parser.add_argument('--exp_num', default=0, type=int)
    parser.add_argument('--n_test', default=100, type=int)
    args = parser.parse_args()
    main(args)