from collections import defaultdict
import yaml
import inspect
import torch
import torch.nn as nn
import torchmetrics as tm
from typing import Dict, Tuple, List

class ARGProcessor():
    def __init__(self, setting_file):
        self.setting_file = setting_file
        self.load()

    def load(self):
        with open(self.setting_file) as file:
            self.kwargs = yaml.load(file, Loader=yaml.FullLoader)

    def save(self, path):
        with open(path, 'w') as file:
            yaml.dump(self.kwargs, file)
    
    def get_args(self, cls):
        cls_kwargs = {
            k: self.kwargs.get(k) for k in inspect.signature(cls.__init__).parameters.keys() 
            if self.kwargs.get(k) is not None
        }
        return cls_kwargs


class MetricRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        cs = tm.MetricCollection({
            'Accuracy': tm.Accuracy(), 
            'Loss': tm.SumMetric()
        })
        self.metrics = tm.MetricCollection([
            cs.clone('Support_'), cs.clone('Query_'), cs.clone('Finetune_'),
            tm.MetricCollection({
                'Inner': tm.MeanMetric(), 'Finetuning': tm.MeanMetric()
            }, postfix='_LR'),
            tm.MetricCollection({
                'Total': tm.SumMetric(), 
                'KLD': tm.SumMetric(), 
                'Z': tm.SumMetric(),
                'Orthogonality': tm.SumMetric()
            }, postfix='_Loss')
        ])

        # self.reset_window_metrics()

    @property
    def keys(self):
        return list(self.metrics.keys())

    # def reset_window_metrics(self):
    #     self.window_metrics = defaultdict(dict)

    def update(self, key, scores=None | torch.FloatTensor, targets=None | torch.LongTensor):
        if 'Accuracy' in key:
            if targets is None:
                raise KeyError('Must insert `targets` to calculate accuracy.')
            self.metrics[key].update(scores, targets)
        else:
            self.metrics[key].update(scores)

    def compute(self, prefix: str):
        results = {}
        for k in self.keys:
            m = self.metrics[k].compute()
            if isinstance(m, torch.Tensor):
                m = m.cpu().detach().numpy()
            results[f'{prefix}-{k}'] = m
        return results

    def reset(self):
        for k in self.keys:
            self.metrics[k].reset()

    # def update_window_metrics(self, window_size):
    #     results = self.compute()
    #     self.window_metrics[window_size] = results

    # def get_window_metrics(self, window_size):
    #     return self.window_metrics[window_size]

    # def compute_total_metrics(self):
    #     # averaged by number of window size
    #     windows, metrics = list(zip(*self.window_metrics.items()))
    #     results = {k: 0.0 for k in self.keys}
    #     for m in metrics:
    #         for k in self.keys:
    #             results[k] += m[k]
    #     for k in self.keys:
    #         results[k] /= len(windows)  # TODO: calculate average performance of 4 tasks?

    #     return results

    # def get_log_data(self, prefix: str, window_size: int | None=None):
    #     log_string = f'{prefix}'
    #     if window_size is not None:
    #         log_string += f'-WinSize={window_size}'
    #         metrics = self.get_window_metrics(window_size)
    #     else:
    #         metrics = self.compute_total_metrics()

    #     log_data = {}
    #     for key in self.keys:
    #         value = metrics[key]
    #         log_data[f'{log_string}-{key}'] = value

    #     return log_data

    def extract_query_loss_acc(self, logs: Dict[str, float] | List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        to_filter = ['Query_Accuracy', 'Query_Loss']
        check_func = lambda x: sum([1 if f in x[0] else 0 for f in to_filter if f in x[0]])
        if isinstance(logs, dict):
            # cumulated logs
            filtered = dict(filter(check_func, logs.items()))
        else:
            filtered = {}
            for l in logs:
                win_filtered = dict(filter(check_func, l.items()))
                filtered.update(win_filtered)
        return filtered