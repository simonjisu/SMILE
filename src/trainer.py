import torch
import torch.nn as nn
import numpy as np
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(
            self, 
            exp_name, 
            log_dir, 
            total_steps,
            n_inner_step, 
            n_finetuning_step, 
            n_valid_step,
            every_valid_step,
            beta,
            gamma,
            lambda1,
            lambda2,
            outer_lr,
            clip_value,
            device: str='cpu',
            print_step: int=5
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
        

        # check if exp exists
        self.exp_name = exp_name
        self.log_dir = Path(log_dir)
        exp_dirs = list(self.log_dir.glob(f'{self.exp_name}_*'))
        exp_num = int(exp_dirs[-1].name[len(self.exp_name)+1:]) if exp_dirs else 0
        self.exp_dir = self.log_dir / f'{self.exp_name}_{exp_num+1}'
        self.writer = SummaryWriter(str(self.exp_dir))
        self.ckpt_path = self.exp_dir / 'checkpoints'
        if not self.ckpt_path.exists():
            self.ckpt_path.mkdir()
        self.log_keys = [
            'Support Loss', 'Support Accuracy', 'Query Loss', 'Query Accuracy', 
            'Inner LR', 'Finetuning LR', 'KLD Loss', 'Z Penalty', 'Orthogonality Penalty'
        ]

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

    def manual_model_eval(self, model, mode=False):
        # [PyTorch Issue] RuntimeError: cudnn RNN backward can only be called in training mode
        # cannot use model.eval()
        # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        for module in model.children():
            # model.training = mode
            if isinstance(module, nn.Dropout) or isinstance(module, nn.LayerNorm):
                module.train(mode)
        return model

    def meta_train(self, model, meta_trainset):
        model = model.to(self.device)
        lr_list = ['inner_lr', 'finetuning_lr']
        params = [x[1] for x in list(filter(lambda k: k[0] not in lr_list, model.named_parameters()))]
        lr_params = [x[1] for x in list(filter(lambda k: k[0] in lr_list, model.named_parameters()))]
        optim = torch.optim.Adam(params, lr=self.outer_lr, weight_decay=self.lambda1)
        optim_lr = torch.optim.Adam(lr_params, lr=self.outer_lr, weight_decay=self.lambda1)
        best_eval_acc = 0.0

        for step in range(self.total_steps):
            # Meta Train
            model.train()
            optim.zero_grad()
            optim_lr.zero_grad()
            train_tasks = meta_trainset.generate_tasks()
            train_records = {k: [] for k in self.log_keys}
            
            all_total_loss = 0.
            for window_size, tasks in train_tasks.items(): # window size x (n_sample * n_stock)
                batch_data = self.map_to_tensor(tasks, device=self.device)
                total_loss, records = self.step_batch(model, batch_data)
                # version - 2
                all_total_loss += total_loss
                for key, v in records.items():
                    if (key in ['Latents']):
                        continue
                    else:
                        train_records[key].append(v)
            all_total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            optim.step()
            optim_lr.step()
                # version - 1
                # total_loss.backward()

                # nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
                # nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
                # optim.step()
                # optim_lr.step()
                # for key, v in records.items():
                #     if (key in ['Latents']):
                #         self.writer.add_histogram(f'Latents-WinSize={window_size}', records['Latents'], step)
                #     else:
                #         train_records[key].append(v)
                #         self.writer.add_scalar(f'Train-WinSize={window_size}-{key}', v, step)
                
            # logging summary(average score for all window size tasks)
            # for key in self.log_keys:
            #     self.writer.add_scalar(f'Train-{key}', np.mean(train_records[key]), step)
                
            if (step % self.print_step == 0) or (step == self.total_steps-1):
                print(f'[Meta Train]({step+1}/{self.total_steps})')
                for i, key in enumerate(self.log_keys):
                    s1 = '  ' if (i == 0) or (i == 4) else ''
                    s2 = '\n' if i == 3 else " | "
                    print(f'{s1}{key}: {np.mean(train_records[key]):.4f}', end=s2)
                print()
                
            # Meta Valid
            if (step % self.every_valid_step == 1) or (step == self.total_steps-1):
                # [PyTorch Issue] RuntimeError: cudnn RNN backward can only be called in training mode
                # cannot use model.eval()
                # model = self.manual_model_eval(model, mode=False)
                model.manual_model_eval(False)
                valid_records = {'Accuracy': [], 'Loss': []}
                # n_valid_step x window_size 
                for val_step in range(self.n_valid_step):
                    valid_step_loss = []
                    valid_step_acc = []
                    valid_tasks = meta_trainset.generate_tasks()

                    for window_size, tasks in valid_tasks.items():
                        batch_data = self.map_to_tensor(tasks, device=self.device)
                        _, records = self.step_batch(model, batch_data)
                        valid_step_loss.append(records['Query Loss'])
                        valid_step_acc.append(records['Query Accuracy'])

                    valid_records['Accuracy'].append(valid_step_acc)
                    valid_records['Loss'].append(valid_step_loss)
                # average window loss and accruacy: mean by n_valid_step
                valid_records['Accuracy'] = np.mean(valid_records['Accuracy'], axis=0)
                valid_records['Loss'] = np.mean(valid_records['Loss'], axis=0)

                # for key in ['Accuracy', 'Loss']:
                #     for i, window_size in enumerate(meta_trainset.window_sizes):
                #         self.writer.add_scalar(f'Valid-WinSize={window_size}-{key}', valid_records[key][i], step)
                #     self.writer.add_scalar(f'Valid-Task {key}', np.mean(valid_records[key]), step)

                print(f'[Meta Valid]({step+1}/{self.total_steps})')
                for key in ['Accuracy', 'Loss']:
                    print(f'  Task {key}: {np.mean(valid_records[key]):.4f}')

                cur_eval_loss = np.mean(valid_records['Loss'])
                cur_eval_acc = np.mean(valid_records['Accuracy'])
                if cur_eval_acc > best_eval_acc:
                    best_eval_acc = cur_eval_acc 
                    torch.save(model.state_dict(), str(self.ckpt_path / f'{step}-{cur_eval_acc:.4f}-{cur_eval_loss:.4f}.ckpt'))

    def meta_test(self, meta_test1, meta_test2, meta_test3):
        pass