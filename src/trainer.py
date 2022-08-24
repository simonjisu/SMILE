import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        self.ckpt_path = self.exp_dir / 'checkpoints'
        self.ckpt_step_path = self.ckpt_path / 'step'
        if not self.ckpt_path.exists():
            self.ckpt_path.mkdir()
        if not self.ckpt_step_path.exists():
            self.ckpt_step_path.mkdir()

    def get_best_results(self, exp_num, record_tensorboard: bool=True):
        self.init_experiments(exp_num=exp_num, record_tensorboard=record_tensorboard)
        best_ckpt = sorted(
            (self.ckpt_step_path).glob('*.ckpt'),
            key=lambda x: x.name.split('-')[1], 
            reverse=True
        )[0]
        best_step, train_acc, train_loss = best_ckpt.name.rstrip('.ckpt').split('-')
        state_dict = torch.load(best_ckpt)
        return int(best_step), float(train_acc), float(train_loss), state_dict

    def _train(self, model, meta_dataset, optim, optim_lr):
        # Meta Train
        model.meta_train()
        optim.zero_grad()
        optim_lr.zero_grad()
        train_tasks = meta_dataset.generate_tasks()

        average_total_loss = model.meta_run(
            tasks=train_tasks,
            beta=self.beta, 
            gamma=self.gamma, 
            lambda2=self.lambda2, 
            n_inner_step=self.n_inner_step, 
            n_finetuning_step=self.n_finetuning_step, 
            rt_attn=False,
            device=self.device
        )

        average_total_loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
        nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        optim.step()
        optim_lr.step()

        return 

    def _valid(self, model, meta_dataset, n_valid: int, prefix: str):
        # turn-off dropout and sample by mean
        model.meta_valid()
        valid_logs = defaultdict(list)
        valid_win_logs = defaultdict(list)

        pregress = tqdm(range(n_valid), total=n_valid, desc=f'Running {prefix}')
        for val_idx in pregress:
            valid_tasks = meta_dataset.generate_tasks()
            
            _ = model.meta_run(
                tasks=valid_tasks,
                beta=self.beta, 
                gamma=self.gamma, 
                lambda2=self.lambda2, 
                n_inner_step=self.n_inner_step, 
                n_finetuning_step=self.n_finetuning_step, 
                rt_attn=False,
                device=self.device
            )

            logs, window_logs = self.get_logs(meta_dataset, model, prefix)
            
            # log by window: List[Dict[str, float]]
            for log_string, value in window_logs.items():
                valid_win_logs[log_string].append(value)

            # log all windows: averaged by number of window size
            for log_string, value in logs.items():
                valid_logs[log_string].append(value)
        pregress.close()

        for k, v in valid_win_logs.items():
            valid_win_logs[k] = (np.mean(v), np.std(v))

        for k, v in valid_logs.items():
            valid_logs[k] = (np.mean(v), np.std(v))

        return valid_logs, valid_win_logs

    def meta_train(self, model, meta_dataset, print_log: bool=True):
        model = model.to(self.device)
        lr_list = ['inner_lr', 'finetuning_lr']
        params = [x[1] for x in list(filter(lambda k: k[0] not in lr_list, model.named_parameters()))]
        lr_params = [x[1] for x in list(filter(lambda k: k[0] in lr_list, model.named_parameters()))]
        optim = torch.optim.Adam(params, lr=self.outer_lr, weight_decay=self.lambda1)
        optim_lr = torch.optim.Adam(lr_params, lr=self.outer_lr, weight_decay=self.lambda1)
        
        best_eval_acc = 0.0

        for step in range(self.total_steps):
            # Meta Train
            self._train(model, meta_dataset=meta_dataset, optim=optim, optim_lr=optim_lr)

            if (step % self.print_step == 0) or (step == self.total_steps-1):
                prefix = 'Train'
                train_logs, train_win_logs = self.get_logs(meta_dataset, model, prefix)
                self.log_results(train_logs, train_win_logs, prefix, step=step, print_log=print_log)

            # Meta Valid
            if (step % self.every_valid_step == 0) or (step == self.total_steps-1):
                prefix = 'Valid'
                valid_logs, valid_win_logs = self._valid(
                    model=model, meta_dataset=meta_dataset, n_valid=self.n_valid_step, prefix=prefix
                )
                self.log_results(valid_logs, valid_win_logs, prefix, step=step, print_log=print_log)
                
                # model save best        
                cur_eval_loss = valid_logs[f'{prefix}-Query_Loss'][0]
                cur_eval_acc = valid_logs[f'{prefix}-Query_Accuracy'][0]
                
                # save by every step 
                torch.save(model.state_dict(), str(self.ckpt_step_path / f'{step}-{cur_eval_acc:.4f}-{cur_eval_loss:.4f}.ckpt'))
                # save best
                if cur_eval_acc > best_eval_acc:
                    best_eval_acc = cur_eval_acc 
                    torch.save(model.state_dict(), str(self.ckpt_path / f'best_model.ckpt'))

    def meta_test(self, model, meta_dataset, n_test: int=100, print_log: bool=True):
        # load model
        model = model.to(self.device)
        # test
        prefix = meta_dataset.meta_type.capitalize()
        test_logs, test_win_logs = self._valid(
            model=model, meta_dataset=meta_dataset, n_valid=n_test, prefix=prefix
        )
        self.log_results(test_logs, test_win_logs, prefix, step=0, print_log=print_log)
        
        test_acc_loss = model.recorder.extract_query_loss_acc(test_logs)
        test_win_acc_loss = model.recorder.extract_query_loss_acc(test_win_logs)
        return test_acc_loss, test_win_acc_loss

    def get_logs(self, meta_dataset, model, prefix: str):
        window_logs = {}
        for window_size in meta_dataset.window_sizes:
            log_data = model.recorder.get_log_data(prefix=prefix, window_size=window_size)
            window_logs.update(log_data)

        logs = model.recorder.get_log_data(prefix=prefix, window_size=None)
        return logs, window_logs

    def log_results(self, logs, window_logs, prefix, step, print_log=False):
        # log by window
        # if prefix == 'Train': window_logs = list
        # else: window_logs = dict
        for log_string, value in window_logs.items():
            if prefix != 'Train':
                # tuple for (mean, std) at Valid, Test mode
                value = value[0]
            self.writer.add_scalar(log_string, value, step)

        # log all windows: averaged by number of window size
        for log_string, value in logs.items():
            if prefix != 'Train':
                # tuple for (mean, std) at Valid, Test mode
                value = value[0]
            self.writer.add_scalar(log_string, value, step)

        if print_log:
            def extract(prefix, key, logs):
                if prefix == 'Train':
                    mean = logs[f'{prefix}-{key}']
                    std = None
                else:
                    mean, std = logs[f'{prefix}-{key}']

                s = f'{mean:.4f}'
                if std is not None:
                    s += f' +/- {std:.4f}'
                return s

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

            print(f'[Meta {prefix}]({step+1}/{self.total_steps})')
            print(f'  - [Support] Loss: {s_loss}, Accuracy: {s_acc}')
            print(f'  - [Query] Loss: {q_loss}, Accuracy: {q_acc}')
            print(f'  - [Finetune] Loss: {f_loss}, Accuracy: {f_acc}')
            print(f'  - [Loss] Z: {z_loss}, KLD: {kld_loss}, Orthogonality: {oth_loss}, Total: {total_loss}')
            print()