import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(
            self, 
            exp_name: str, 
            log_dir: str, 
            total_steps: int,
            n_inner_step: int, 
            n_finetuning_step: int, 
            n_valid_step: int,
            every_valid_step: int,
            beta: float,
            gamma: float,
            lambda1: float,
            lambda2: float,
            outer_lr: float,
            clip_value: float,
            device: str='cpu',
            print_step: int=5,
        ):
        self.device = device
        self.print_step = print_step
        self.total_steps = total_steps
        self.n_inner_step = n_inner_step
        self.n_finetuning_step = n_finetuning_step
        self.n_valid_step = n_valid_step
        self.every_valid_step = every_valid_step
        
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1  # penalty on model(encoder, mapping_net, decoder) parameters
        self.lambda2 = lambda2  # penalty on decoder
        self.outer_lr = outer_lr
        self.clip_value = clip_value
        
        self.exp_name = exp_name
        self.log_dir = Path(log_dir).resolve()

        # aggregate method by window sizes
        self.log_keys = {
            'Support Loss': np.sum, 
            'Support Accuracy': np.mean, 
            'Query Loss': np.sum, 
            'Query Accuracy': np.mean,
            'Finetune Loss': np.sum,
            'Finetune Accuracy': np.mean,
            'Total Loss': np.sum,
            'Inner LR': np.mean,  # average Learing Rate
            'Finetuning LR': np.mean, 
            'KLD Loss': np.sum, 
            'Z Penalty': np.sum, 
            'Orthogonality Penalty': np.sum
        }
        
    def init_experiments(self, exp_num=None, record_tensorboard: bool=True):
        # check if exp exists
        exp_dirs = sorted(list(self.log_dir.glob(f'{self.exp_name}_*')))
        if exp_num is None:
            exp_num = int(exp_dirs[-1].name[len(self.exp_name)+1:]) if exp_dirs else 0
            self.exp_num = exp_num + 1
        else:
            self.exp_num = exp_num
        self.exp_dir = self.log_dir / f'{self.exp_name}_{self.exp_num}'
        if record_tensorboard:
            self.writer = SummaryWriter(str(self.exp_dir))
        self.ckpt_path = self.exp_dir / 'checkpoints'
        self.ckpt_step_path = self.ckpt_path / 'step'
        if not self.ckpt_path.exists():
            self.ckpt_path.mkdir()
        if not self.ckpt_step_path.exists():
            self.ckpt_step_path.mkdir()

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

    def step_batch(self, model, batch_data):
        total_loss, records = model.meta_run(
            data=batch_data, 
            beta=self.beta, 
            gamma=self.gamma, 
            lambda2=self.lambda2, 
            n_inner_step=self.n_inner_step, 
            n_finetuning_step=self.n_finetuning_step, 
            rt_attn=False
        )
        return total_loss, records

    def _train(self, model, meta_dataset, optim, optim_lr, step):
        # Meta Train
        model.meta_train()
        optim.zero_grad()
        optim_lr.zero_grad()
        train_tasks = meta_dataset.generate_tasks()
        train_records = {k: [] for k in self.log_keys.keys()}
        
        # Outer Loop
        all_total_loss = 0.
        for window_size, tasks in train_tasks.items(): # window size x (n_sample * n_stock)
            batch_data = self.map_to_tensor(tasks, device=self.device)
            total_loss, records = self.step_batch(model, batch_data)
            # version - 2
            all_total_loss += total_loss
            for key, v in records.items():
                # if (key in ['Z', 'Z Prime']):
                #     self.writer.add_histogram(f'{key}-WinSize={window_size}', records[key], step)
                # else:
                train_records[key].append(v)
                self.writer.add_scalar(f'Train-WinSize={window_size}-{key}', v, step)

        all_total_loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
        nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        optim.step()
        optim_lr.step()

        return train_records

    def _valid(self, model, meta_dataset, n_valid: int):
        # turn-off dropout and sample by mean
        model.manual_model_eval()
        valid_records = {'Accuracy': [], 'Loss': []}  # n_valid x window_size 
        for val_step in range(n_valid):
            valid_step_loss = []
            valid_step_acc = []
            valid_tasks = meta_dataset.generate_tasks()

            for window_size, tasks in valid_tasks.items():
                batch_data = self.map_to_tensor(tasks, device=self.device)
                _, records = self.step_batch(model, batch_data)
                valid_step_loss.append(records['Query Loss'])
                valid_step_acc.append(records['Query Accuracy'])

            valid_records['Accuracy'].append(valid_step_acc)
            valid_records['Loss'].append(valid_step_loss)
        
        # aggregate window loss and accruacy: mean by n_valid
        valid_records['Accuracy'] = np.mean(valid_records['Accuracy'], axis=0)
        valid_records['Loss'] = np.mean(valid_records['Loss'], axis=0)

        return valid_records

    def meta_train(self, model, meta_dataset):
        model = model.to(self.device)
        lr_list = ['inner_lr', 'finetuning_lr']
        params = [x[1] for x in list(filter(lambda k: k[0] not in lr_list, model.named_parameters()))]
        lr_params = [x[1] for x in list(filter(lambda k: k[0] in lr_list, model.named_parameters()))]
        optim = torch.optim.Adam(params, lr=self.outer_lr, weight_decay=self.lambda1)
        optim_lr = torch.optim.Adam(lr_params, lr=self.outer_lr, weight_decay=self.lambda1)
        best_eval_acc = 0.0

        for step in range(self.total_steps):
            # Meta Train
            train_records = self._train(model, meta_dataset=meta_dataset, optim=optim, optim_lr=optim_lr, step=step)
            
            if (step % self.print_step == 0) or (step == self.total_steps-1):

                # logging summary(aggregate score for all window size tasks)
                for key, agg_func in self.log_keys.items():
                    self.writer.add_scalar(f'Train-{key}', agg_func(train_records[key]), step)

                print(f'[Meta Train]({step+1}/{self.total_steps})')
                for i, (key, agg_func) in enumerate(self.log_keys.items()):
                    s1 = '  ' if (i == 0) or (i == 6) else ''
                    s2 = '\n' if (i == 5) else " | "
                    print(f'{s1}{key}: {agg_func(train_records[key]):.4f}', end=s2)
                print()
                
            # Meta Valid
            if (step % self.every_valid_step == 0) or (step == self.total_steps-1):
                valid_records = self._valid(model=model, meta_dataset=meta_dataset, n_valid=self.n_valid_step)
                
                # record
                for key, agg_func in zip(['Accuracy', 'Loss'], [np.mean, np.sum]):
                    for i, window_size in enumerate(meta_dataset.window_sizes):
                        self.writer.add_scalar(f'Valid-WinSize={window_size}-{key}', valid_records[key][i], step)
                    self.writer.add_scalar(f'Valid-Task {key}', agg_func(valid_records[key]), step)

                print(f'[Meta Valid]({step+1}/{self.total_steps})')
                for i, (key, agg_func) in enumerate(zip(['Accuracy', 'Loss'], [np.mean, np.sum])):
                    s1 = '  ' if i == 0 else ''
                    s2 = ' | ' if i == 0 else '\n'
                    print(f'{s1}{key}: {agg_func(valid_records[key]):.4f}', end=s2)

                # model save best        
                cur_eval_loss = np.mean(valid_records['Loss'])
                cur_eval_acc = np.mean(valid_records['Accuracy'])
                # save by every step 
                torch.save(model.state_dict(), str(self.ckpt_step_path / f'{step}-{cur_eval_acc:.4f}-{cur_eval_loss:.4f}.ckpt'))
                # save best
                if cur_eval_acc > best_eval_acc:
                    best_eval_acc = cur_eval_acc 
                    torch.save(model.state_dict(), str(self.ckpt_path / f'best_model.ckpt'))

    def get_best_results(self, exp_num, record_tensorboard: bool=True):
        self.init_experiments(exp_num=exp_num, record_tensorboard=record_tensorboard)
        # best_ckpt = sorted((self.ckpt_path).glob('*.ckpt'), key=lambda x: x.name.split('-')[1], reverse=True)[0]
        best_ckpt = sorted((self.ckpt_step_path).glob('*.ckpt'), key=lambda x: x.name.split('-')[1], reverse=True)[0]
        best_step, train_acc, train_loss = best_ckpt.name.rstrip('.ckpt').split('-')
        state_dict = torch.load(best_ckpt)
        return int(best_step), float(train_acc), float(train_loss), state_dict

    def meta_test(self, model, meta_dataset, n_test: int=100, record_tensorboard: bool=False):
        # load model
        model = model.to(self.device)
        # test
        test_records = self._valid(model=model, meta_dataset=meta_dataset, n_valid=n_test)

        results = defaultdict(dict)
        results['Win']['Accuracy'] = {}
        results['Win']['Loss'] = {}

        for key, agg_func in zip(['Accuracy', 'Loss'], [np.mean, np.sum]):
            for i, window_size in enumerate(meta_dataset.window_sizes):
                k = f'{meta_dataset.meta_type.capitalize()}-WinSize={window_size}-{key}'
                if record_tensorboard:
                    self.writer.add_scalar(k, test_records[key][i], window_size)
                results['Win'][key][window_size] = test_records[key][i]
            k = f'{meta_dataset.meta_type.capitalize()}-Task-{key}'
            results[k] = agg_func(test_records[key])
            if record_tensorboard:
                self.writer.add_scalar(k, agg_func(test_records[key]), 0)
            results['Task'][key] = agg_func(test_records[key])
        return results
        