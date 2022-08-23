
from collections import defaultdict
import yaml
import inspect
import numpy as np

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

class MetricRecords():
    def __init__(self):
        self.metrics = {
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

        self.records = defaultdict(list)

    def update(self):
        pass