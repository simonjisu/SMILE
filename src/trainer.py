import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataset import StockDataDict
from typing import Dict

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
        else:
            self.writer = None
        self.ckpt_path = self.exp_dir / 'checkpoints'
        self.ckpt_step_train_path =  self.ckpt_path / 'step' / 'train'
        self.ckpt_step_valid_path =  self.ckpt_path / 'step' / 'valid'
        for p in [self.ckpt_path, self.ckpt_step_train_path, self.ckpt_step_valid_path]:
            if not p.exists():
                p.mkdir(parents=True)

    def get_best_results(self, exp_num, record_tensorboard: bool=True):
        self.init_experiments(exp_num=exp_num, record_tensorboard=record_tensorboard)
        best_ckpt = sorted(
            (self.ckpt_step_valid_path).glob('*.ckpt'),
            key=lambda x: x.name.split('-')[1], 
            reverse=True
        )[0]
        
        best_step, train_acc, train_loss = best_ckpt.name.rstrip('.ckpt').split('-')
        state_dict = torch.load(best_ckpt)
        return int(best_step), float(train_acc), float(train_loss), state_dict

    def outer_loop(
        self, 
        model, 
        meta_dataset: Dict[int, StockDataDict], 
        optim, 
        optim_lr
        ):
        # Meta Train
        model.meta_train()
        optim.zero_grad()
        optim_lr.zero_grad()
        train_tasks = meta_dataset.generate_tasks()  # StockDataDict
        train_tasks.to(self.device)
        # train_tasks: StockDataDict
        # - query: (B, 1, T, I)
        # - query_labels: (B)
        # - support: (B, N*K[n_support], T, I)
        # - support_labels: (B*N*K)
        
        #  model.recorder.reset_window_metrics()    
        # Reset record: only update for a single window size with `number of stocks`
        model.recorder.reset()  
        # Task specific Inner and Outer Loop
        total_loss, *_ = model(
            data=train_tasks, 
            beta=self.beta, 
            gamma=self.gamma, 
            lambda2=self.lambda2, 
            n_inner_step=self.n_inner_step, 
            n_finetuning_step=self.n_finetuning_step, 
            rt_attn=False
        )
        total_loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
        nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        optim.step()
        optim_lr.step()

        # model.recorder.update_window_metrics(train_tasks.window_size)

        return 

    def _valid(self, model, meta_dataset, n_valid: int, prefix: str):
        # turn-off dropout and sample by mean
        model.meta_eval()
        valid_logs = defaultdict(list)

        pregress = tqdm(range(n_valid), total=n_valid, desc=f'Running {prefix}')
        for val_idx in pregress:
            valid_tasks = meta_dataset.generate_tasks()
            valid_tasks.to(self.device)

            # Reset record: only update for a single window size with `number of stocks`
            model.recorder.reset()
            # Task specific Inner and Outer Loop
            model(
                data=valid_tasks, 
                beta=self.beta, 
                gamma=self.gamma, 
                lambda2=self.lambda2, 
                n_inner_step=self.n_inner_step, 
                n_finetuning_step=self.n_finetuning_step, 
                rt_attn=False
            )
            logs = model.recorder.compute(prefix)
            
            for log_string, value in logs.items():
                valid_logs[log_string].append(value)
        pregress.close()

        for k, v in valid_logs.items():
            valid_logs[k] = (np.mean(v), np.std(v))

        return valid_logs

    def run_train(
        self, 
        model, 
        meta_trainset,
        meta_validset_time,
        meta_validset_stock,
        meta_validset_mix, 
        print_log: bool=True
        ):
        
        model = model.to(self.device)
        lr_list = ['inner_lr', 'finetuning_lr']
        params = [x[1] for x in list(filter(lambda k: k[0] not in lr_list, model.named_parameters()))]
        lr_params = [x[1] for x in list(filter(lambda k: k[0] in lr_list, model.named_parameters()))]
        optim = torch.optim.Adam(params, lr=self.outer_lr, weight_decay=self.lambda1)
        optim_lr = torch.optim.Adam(lr_params, lr=self.outer_lr, weight_decay=self.lambda1)
        
        best_eval_f1 = 0.0

        for step in range(self.total_steps):
            # Meta Train
            self.outer_loop(model, meta_dataset=meta_trainset, optim=optim, optim_lr=optim_lr)

            if (step % self.print_step == 0) or (step == self.total_steps-1):
                prefix = 'Train'
                train_logs = model.recorder.compute(prefix)
                cur_eval_loss = train_logs[f'{prefix}-Query_Loss']
                cur_eval_acc = train_logs[f'{prefix}-Query_Accuracy']
                self.log_results(train_logs, prefix, step=step, total_steps=self.total_steps, print_log=print_log)
                torch.save(model.state_dict(), str(self.ckpt_step_train_path / f'{step}-{cur_eval_acc:.4f}-{cur_eval_loss:.4f}.ckpt'))
                
            # Meta Valid
            if (self.every_valid_step != 0):
                if (step % self.every_valid_step == 0) or (step == self.total_steps-1):
                    ref_step = step

                    prefix = 'Valid-Time'
                    valid_logs_time = self.run_valid(model, meta_validset_time, prefix)
                    self.log_results(valid_logs_time, prefix, step=ref_step, total_steps=self.total_steps, print_log=print_log)
                    cur_eval_acc_time = valid_logs_time[f'{prefix}-Query_Accuracy'][0]
                    cur_eval_loss_time = valid_logs_time[f'{prefix}-Query_Loss'][0]
                    
                    prefix = 'Valid-Stock'
                    valid_logs_stock = self.run_valid(model, meta_validset_stock, prefix)
                    self.log_results(valid_logs_stock, prefix, step=ref_step, total_steps=self.total_steps, print_log=print_log)
                    cur_eval_acc_stock = valid_logs_stock[f'{prefix}-Query_Accuracy'][0]
                    cur_eval_loss_stock = valid_logs_stock[f'{prefix}-Query_Loss'][0]
                    
                    prefix = 'Valid-Mix'
                    valid_logs_mix = self.run_valid(model, meta_validset_mix, prefix)
                    self.log_results(valid_logs_mix, prefix, step=ref_step, total_steps=self.total_steps, print_log=print_log)
                    cur_eval_acc_mix = valid_logs_mix[f'{prefix}-Query_Accuracy'][0]
                    cur_eval_loss_mix = valid_logs_mix[f'{prefix}-Query_Loss'][0]

                    prefix = 'Valid'
                    cur_eval_f1 = 3 / ((1/cur_eval_acc_time) + (1/cur_eval_acc_stock) + (1/cur_eval_acc_mix))
                    cur_eval_loss = (cur_eval_loss_time + cur_eval_loss_stock + cur_eval_loss_mix) / 3
                    valid_final_log = {f'{prefix}-F1': [cur_eval_f1], f'{prefix}-AvgLoss': [cur_eval_loss]}
                    self.log_results(valid_final_log, prefix, step=ref_step, total_steps=self.total_steps, print_log=print_log)

                    # save best
                    if (cur_eval_f1 > best_eval_f1):
                        best_eval_f1 = cur_eval_f1 
                        torch.save(model.state_dict(), str(self.ckpt_step_valid_path / f'{ref_step:06d}-{cur_eval_f1:.4f}-{cur_eval_loss:.4f}.ckpt'))


    def run_valid(self, model, meta_dataset, prefix):
        model = model.to(self.device)
        valid_logs = self._valid(
            model=model, meta_dataset=meta_dataset, n_valid=self.n_valid_step, prefix=prefix
        )
        return valid_logs

    def meta_test(self, model, meta_dataset, n_test: int=100, print_log: bool=True):
        # load model
        model = model.to(self.device)
        # test
        prefix = meta_dataset.meta_type.capitalize()
        test_logs = self._valid(
            model=model, meta_dataset=meta_dataset, n_valid=n_test, prefix=prefix
        )
        self.log_results(test_logs, prefix, step=0, total_steps=0, print_log=print_log)
        
        test_acc_loss = model.recorder.extract_query_loss_acc(test_logs)
        return test_acc_loss

    def log_results(self, logs, prefix, step, total_steps, print_log=False):
        for log_string, value in logs.items():
            if prefix != 'Train':
                # tuple for (mean, std) at Valid, Test mode
                value = value[0]
            if self.writer is not None:
                self.writer.add_scalar(log_string, value, step)

        if print_log:
            only_one_to_print = True if prefix == 'Valid' else False

            def extract(prefix, key, logs):
                if prefix == 'Train':
                    mean = logs[f'{prefix}-{key}']
                    std = None
                elif prefix == 'Valid':
                    # F1, Loss
                    mean = logs[f'{prefix}-{key}'][0]
                    std = None
                else:
                    # prefix-***
                    mean, std = logs[f'{prefix}-{key}']
                    
                s = f'{mean:.4f}'
                if std is not None:
                    s += f' +/- {std:.4f}'
                return s

            if only_one_to_print:
                f1 = extract(prefix, 'F1', logs)
                avgloss = extract(prefix, 'AvgLoss', logs)
                print(f'[Meta {prefix}] Result - F1: {f1}, AvgLoss: {avgloss}')
                print()
            else:
                s_acc = extract(prefix, 'Support_Accuracy', logs)
                s_loss = extract(prefix, 'Support_Loss', logs)
                q_acc = extract(prefix, 'Query_Accuracy', logs)
                q_loss = extract(prefix, 'Query_Loss', logs)
                f_acc = extract(prefix, 'Finetune_Accuracy', logs)
                f_loss = extract(prefix, 'Finetune_Loss', logs)
                kld_loss = extract(prefix, 'KLD_Loss', logs)
                oth_loss = extract(prefix, 'Orthogonality_Loss', logs)
                z_loss = extract(prefix, 'Z_Loss', logs)
                total_loss = extract(prefix, 'Total_Loss', logs)
            
                print(f'[Meta {prefix}]({step+1}/{total_steps})')
                print(f'  - [Support] Loss: {s_loss}, Accuracy: {s_acc}')
                print(f'  - [Query] Loss: {q_loss}, Accuracy: {q_acc}')
                print(f'  - [Finetune] Loss: {f_loss}, Accuracy: {f_acc}')
                print(f'  - [Loss] Z: {z_loss}, KLD: {kld_loss}, Orthogonality: {oth_loss}, Total: {total_loss}')
                print()