import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataset import StockDataDict
from typing import Dict
import matplotlib.pyplot as plt

class Trainer():
    def __init__(
            self, 
            exp_name: str, 
            log_dir: str, 
            total_steps: int,
            n_inner_step: int, 
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
        optimizer, 
        lr_optimizer
        ):
        # Meta Train
        model.meta_train()
        optimizer.zero_grad()
        lr_optimizer.zero_grad()
        train_tasks = meta_dataset.generate_tasks()  # StockDataDict
        train_tasks.to(self.device)
        # train_tasks: StockDataDict
        # - query: (B, 1, T, I)
        # - query_labels: (B)
        # - support: (B, N*K[n_support], T, I)
        # - support_labels: (B*N*K)
        
        # Reset record: only update for a single window size with `number of stocks`
        model.recorder.reset()
        # Task specific Inner and Outer Loop
        total_loss, *_ = model(
            data=train_tasks, 
            beta=self.beta, 
            gamma=self.gamma, 
            lambda2=self.lambda2, 
            n_inner_step=self.n_inner_step, 
            rt_attn=False
        )
        total_loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
        nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        optimizer.step()
        lr_optimizer.step()

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
                rt_attn=False
            )
            logs = model.recorder.compute(prefix)
            
            for log_string, value in logs.items():
                # Precision, Recall: (2)
                valid_logs[log_string].append(value)
        pregress.close()

        for k, v in valid_logs.items():
            if k.split('_')[-1] in ['Precision', 'Recall']:
                valid_logs[k] = (np.mean(v, axis=0), np.std(v, axis=0))
            else:
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
        lr_optimizer = torch.optim.AdamW([p for k, p in model.named_parameters() if k in ['inner_lr']], lr=self.outer_lr)
        optimizer = torch.optim.AdamW([p for k, p in model.named_parameters() if k not in ['inner_lr']], lr=self.outer_lr, weight_decay=self.lambda1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.001)
        best_eval_f1 = 0.0
        for step in range(self.total_steps):
            # Meta Train
            self.outer_loop(model, meta_dataset=meta_trainset, optimizer=optimizer, lr_optimizer=lr_optimizer)

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

        # log for query_distribution
        self.log_q_dist(meta_trainset, meta_validset_time, meta_validset_stock, meta_validset_mix)

    def run_valid(self, model, meta_dataset, prefix):
        model = model.to(self.device)
        valid_logs = self._valid(
            model=model, meta_dataset=meta_dataset, n_valid=self.n_valid_step, prefix=prefix
        )
        return valid_logs

    def log_results(self, logs, prefix, step, total_steps, print_log=False):
        for log_string, value in logs.items():
            if prefix != 'Train':
                # tuple for (mean, std) at Valid, Test mode
                value = value[0]
            if self.writer is not None:
                if log_string.split('_')[-1] in ['Precision', 'Recall']:
                    for i, v in enumerate(value):
                        self.writer.add_scalar(log_string+f': Class {i}', v, step)
                else:
                    self.writer.add_scalar(log_string, value, step)

        if print_log:
            only_one_to_print = True if prefix in ['Valid', 'Test'] else False

            def extract(prefix, key, logs):
                if prefix == 'Train':
                    mean = logs[f'{prefix}-{key}']
                    std = None
                elif prefix == 'Valid':
                    # F1, Loss
                    mean = logs[f'{prefix}-{key}'][0]
                    std = None
                else:
                    # Valid-***, Test-***
                    mean, std = logs[f'{prefix}-{key}']
                
                s = ''
                if key.split('_')[-1] in ['Precision', 'Recall']:
                    for i in range(len(mean)):
                        s += f' (Class {i}) {mean[i]:.4f}'
                        if std is not None:
                            s += f' +/- {std[i]:.4f}'
                else:
                    s += f'{mean:.4f}'
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
                s_precision = extract(prefix, 'Support_Precision', logs)
                s_recall = extract(prefix, 'Support_Recall', logs)
                s_loss = extract(prefix, 'Support_Loss', logs)
                s_param_l2_loss = extract(prefix, 'Support_ParamL2Loss', logs)
                s_pred_loss = extract(prefix, 'Support_PredLoss', logs)
                q_acc = extract(prefix, 'Query_Accuracy', logs)
                q_precision = extract(prefix, 'Query_Precision', logs)
                q_recall = extract(prefix, 'Query_Recall', logs)
                q_loss = extract(prefix, 'Query_Loss', logs)
                q_param_l2_loss = extract(prefix, 'Query_ParamL2Loss', logs)
                q_pred_loss = extract(prefix, 'Query_PredLoss', logs)
                kld_loss = extract(prefix, 'KLD_Loss', logs)
                oth_loss = extract(prefix, 'Orthogonality_Loss', logs)
                z_loss = extract(prefix, 'Z_Loss', logs)
                total_loss = extract(prefix, 'Total_Loss', logs)
            
                print(f'[Meta {prefix}]({step+1}/{total_steps})')
                print(f'  - [Support] Loss: {s_loss}, ParamL2Loss: {s_param_l2_loss}, PredLoss: {s_pred_loss}, Accuracy: {s_acc}')
                print(f'  - [Support] Precision:{s_precision}, Recall:{s_recall}')
                print(f'  - [Query] Loss: {q_loss}, ParamL2Loss: {q_param_l2_loss}, PredLoss: {q_pred_loss}, Accuracy: {q_acc}')
                print(f'  - [Query] Precision:{q_precision}, Recall:{q_recall}')
                print(f'  - [Loss] Z: {z_loss}, KLD: {kld_loss}, Orthogonality: {oth_loss}, Total: {total_loss}')
                print()

    def plot_q_dist(self, meta_dataset):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        idx = np.arange(max(meta_dataset.q_dist.keys()))
        values = [meta_dataset.q_dist[i] if meta_dataset.q_dist.get(i) else 0 for i in idx]

        ax.bar(idx, values)
        ax.set_xlabel('Query index in labels')
        ax.set_ylabel('Count')
        ax.set_title(f'Meta Type: {meta_dataset.meta_type}')
        plt.tight_layout()
        fig.savefig(self.ckpt_path / f'q_dist_{meta_dataset.meta_type}.png')
        return fig

    def log_q_dist(self, meta_trainset, meta_validset_time, meta_validset_stock, meta_validset_mix):
        for ds in [meta_trainset, meta_validset_time, meta_validset_stock, meta_validset_mix]:
            fig = self.plot_q_dist(meta_dataset=ds)
            self.writer.add_figure(f'Query Distribution: {ds.meta_type}', fig)

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
