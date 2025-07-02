import bisect
from itertools import accumulate
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import json
from collections import defaultdict
import random
import os
from utils_ret import get_transform, read_frames_decord

class MultiTaskDataset(Dataset):
    """ Multi-task dataset for combining task-specific datasets with different data types. (e.g. Retrieval, Action Recognition)
    """
    def __init__(self, datasets_dict, balance_sample_num=True, batch_size=16, scale=2.0):
        # self.unified_dataset = list(datasets_dict.values())
        # self.unified_dataset_lengths = [len(dataset) for dataset in self.unified_dataset]
        # self.cumulative_lengths = [0] + list(accumulate(self.unified_dataset_lengths))
        # self.task_names = list(datasets_dict.keys())
        # Store direct references to the original datasets
        self.datasets = datasets_dict
        self.batch_size = batch_size
        if balance_sample_num:
            self._balance_sample_num(scale=scale)
        self.unified_dataset = tuple(datasets_dict.values())  # tuple is more memory efficient than list
        self.unified_dataset_lengths = tuple(len(dataset) for dataset in self.unified_dataset)
        self.cumulative_lengths = tuple([0] + list(accumulate(self.unified_dataset_lengths)))
        self.task_names = tuple(datasets_dict.keys())
        task_info = {task_name: len(dataset) for task_name, dataset in datasets_dict.items()}
        print(f"Multi-task dataset info: \n{task_info}")
    
    def _balance_sample_num(self, scale=2.0):
        """ Balance the sample number of each task by copying the smaller datasets
        For a smallest dataset, the largest dataset will not exceed the smallest dataset's sample number * scale.
        """
        lengths = [len(dataset) for dataset in self.datasets.values()]
        max_sample_num = max(lengths)
        for task_name, dataset in self.datasets.items():
            if task_name != 'THUMOS14' and len(dataset) < max_sample_num / scale:
                # copy the dataset several times to make it up to the max sample number
                num_copies = max_sample_num // len(dataset)
                dataset.copy_dataset(num_copies)
            if task_name == 'THUMOS14':
                # copy the dataset several times to make it up to the max sample number
                num_copies = max_sample_num // len(dataset) // self.batch_size
                dataset.copy_dataset(num_copies)

    def __len__(self):
        return self.cumulative_lengths[-1]
        
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        idx_offset = idx - self.cumulative_lengths[dataset_idx]
        if dataset_idx == -1:
            print(idx, self.cumulative_lengths) # fake batch dataset has to be appended at the end for now
            print(self.unified_dataset_lengths)
            print(bisect.bisect_right(self.cumulative_lengths, idx))
        return self.unified_dataset[dataset_idx][idx_offset]
