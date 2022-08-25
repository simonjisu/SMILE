import argparse
from pathlib import Path

from src.dataset import MetaStockDataset
from src.utils import ARGProcessor
from src.model import MetaModel
from src.trainer import Trainer


def run_test(trainer, model):
    pass

def main(args):
    if not args.meta_test:
        setting_file = args.exp
        if '.yml' not in args.exp:
            setting_file += '.yml'
        setting_file = Path(args.exp_dir) / setting_file
        meta_args = ARGProcessor(setting_file=setting_file)
        data_kwargs = meta_args.get_args(cls=MetaStockDataset)

        meta_train = MetaStockDataset(meta_type='train', **data_kwargs)
        model_kwargs = meta_args.get_args(cls=MetaModel)
        model = MetaModel(**model_kwargs)

        trainer_kwargs = meta_args.get_args(cls=Trainer)
        trainer = Trainer(**trainer_kwargs)

        # meta train
        trainer.init_experiments(exp_num=None, record_tensorboard=True)
        trainer.meta_train(model, meta_dataset=meta_train)
        meta_args.save(trainer.exp_dir / 'settings.yml')

        # meta test 
        print('=='*10)
        best_step, train_acc, train_loss, state_dict = trainer.get_best_results(
            exp_num=trainer.exp_num, record_tensorboard=True
        )
        model = MetaModel(**model_kwargs)
        model.load_state_dict(state_dict=state_dict)

        print(f'[Meta Train Query Result] Best Step: {best_step} | Accuracy: {train_acc:.4f} | Loss: {train_loss:.4f}')
        meta_test1 = MetaStockDataset(meta_type='test1', **data_kwargs)
        meta_test2 = MetaStockDataset(meta_type='test2', **data_kwargs)
        meta_test3 = MetaStockDataset(meta_type='test3', **data_kwargs)
        for meta_test in [meta_test1, meta_test2, meta_test3]:
            test_acc_loss, _ = trainer.meta_test(
                model=model, 
                meta_dataset=meta_test, 
                n_test=args.n_test
            )
            prefix = meta_test.meta_type.capitalize()
            test_acc, test_acc_std = test_acc_loss[f'{prefix}-Query_Accuracy']
            test_loss, test_loss_std = test_acc_loss[f'{prefix}-Query_Loss']
            print(f'[Meta {prefix}] Loss: {test_loss:.4f} +/- {test_loss_std:.4f} | Accuracy: {test_acc:.4f} +/- {test_acc_std:.4f}')
            
    else:
        record_file = open(f'./all_results.csv', 'w', encoding='utf-8')
        record_file_win = open(f'./all_results_win.csv', 'w', encoding='utf-8')
        print('Experiment,TestType,TestLoss,TestLossStd,TestAccuracy,TestAccuracyStd,TrainAccuracy,TrainLoss', file=record_file) 
        print('Experiment,TestType,WindowSize,TestLoss,TestLossStd,TestAccuracy,TestAccuracyStd,TrainAccuracy,TrainLoss', file=record_file_win) 
        
        all_exps = [p for p in Path('./logging').glob('*') if p.is_dir()]
        for exp in all_exps:
            print(f'Processing: {exp.name}')
            setting_file = exp / 'settings.yml'
        
            meta_args = ARGProcessor(setting_file=setting_file)
            data_kwargs = meta_args.get_args(cls=MetaStockDataset)

            # only meta test for all/specific experiments
            meta_test1 = MetaStockDataset(meta_type='test1', **data_kwargs)
            meta_test2 = MetaStockDataset(meta_type='test2', **data_kwargs)
            meta_test3 = MetaStockDataset(meta_type='test3', **data_kwargs)

            model_kwargs = meta_args.get_args(cls=MetaModel)
            trainer_kwargs = meta_args.get_args(cls=Trainer)
            trainer = Trainer(**trainer_kwargs)
            
            exp_num = sorted(
                map(lambda x: int(x.name.split('_')[-1]), 
                trainer.log_dir.glob(f'{trainer.exp_name}_*'))
            )[-1]
            ename = f'{trainer.exp_name}_{exp_num}'
            print(f'===== {ename} =====')
            best_step, train_acc, train_loss, state_dict = trainer.get_best_results(
                exp_num=exp_num, record_tensorboard=False)  # get best results and state dict
            
            model = MetaModel(**model_kwargs)
            model.load_state_dict(state_dict=state_dict)

            print(f'[Meta Train Query Result] Best Step: {best_step} | Accuracy: {train_acc:.4f} | Loss: {train_loss:.4f}')

            for meta_test in [meta_test1, meta_test2, meta_test3]:
                test_acc_loss, test_win_acc_loss = trainer.meta_test(
                    model=model, 
                    meta_dataset=meta_test, 
                    n_test=args.n_test
                )
                prefix = meta_test.meta_type.capitalize()
                test_acc, test_acc_std = test_acc_loss[f'{prefix}-Query_Accuracy']
                test_loss, test_loss_std = test_acc_loss[f'{prefix}-Query_Loss']
                print(f'[Meta {prefix}] Loss: {test_loss:.4f} +/- {test_loss_std:.4f} | Accuracy: {test_acc:.4f} +/- {test_acc_std:.4f}')
                # 'Experiment,TestType,TestLoss,TestLossStd,TestAccuracy,TestAccuracyStd,TrainAccuracy,TrainLoss'
                print(
                    f'{ename},{prefix},{test_loss:.4f},{test_loss_std:.4f},{test_acc:.4f},{test_acc_std:.4f},{train_acc:.4f},{train_loss:.4f}', 
                    file=record_file
                )
                for win_size in meta_test.window_sizes:
                    # 'Experiment,TestType,WindowSize,TestLoss,TestLossStd,TestAccuracy,TestAccuracyStd,TrainAccuracy,TrainLoss'
                    test_acc, test_acc_std = test_win_acc_loss[f'{prefix}-WinSize={win_size}-Query_Accuracy']
                    test_loss, test_loss_std = test_win_acc_loss[f'{prefix}-WinSize={win_size}-Query_Loss']
                    print(
                        f'{ename},{prefix},{win_size},{test_loss:.4f},{test_loss_std:.4f},{test_acc:.4f},{test_acc_std:.4f},{train_acc:.4f},{train_loss:.4f}', 
                        file=record_file_win
                    )
        
        record_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='./experiments', type=str)
    parser.add_argument('--exp', default='', type=str)
    parser.add_argument('--meta_test', action='store_true')
    parser.add_argument('--n_test', default=1, type=int)
    args = parser.parse_args()
    main(args)