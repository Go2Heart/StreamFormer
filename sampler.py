import math
import random
from collections import defaultdict

import torch
from torch.utils.data import DistributedSampler, Sampler


class BatchTaskUniqueSampler(Sampler):
    def __init__(self, dataset, batch_size, indices=None, shuffle=True, drop_last=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.task_to_indices = defaultdict(list)
        offset = 0
        for length in dataset.unified_dataset_lengths:
            task_name = dataset[offset]["task_name"]
            self.task_to_indices[task_name] = list(range(offset, offset + length))
            offset += length

        self.task_names = list(self.task_to_indices.keys())
        if self.shuffle:
            # shuffle among task indices
            for task in self.task_to_indices:
                random.shuffle(self.task_to_indices[task])

    def __iter__(self):
        remaining_tasks = self.task_names.copy()
        while remaining_tasks:
            task_name = random.choice(remaining_tasks)
            if len(self.task_to_indices[task_name]) >= self.batch_size:
                batch = self.task_to_indices[task_name][: self.batch_size]
                self.task_to_indices[task_name] = self.task_to_indices[task_name][
                    self.batch_size :
                ]
                yield batch

            if len(self.task_to_indices[task_name]) < self.batch_size:
                remaining_tasks.remove(task_name)

            if self.shuffle:
                random.shuffle(remaining_tasks)

        # 处理剩余的样本（如果需要）
        if not self.drop_last:
            for task_name, indices in self.task_to_indices.items():
                if indices:
                    yield indices

    def __len__(self):
        # the number of batches
        total_samples = sum([len(indices) for indices in self.task_to_indices.values()])
        return total_samples // self.batch_size


class DistributedBatchTaskSequentialSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=True,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.batch_size = batch_size
        self.task_to_indices = defaultdict(list)

        # Allocate indices for each task
        offset = 0
        for length in dataset.unified_dataset_lengths:
            task_name = dataset[offset]["task_name"]  # Access task_name from dataset
            self.task_to_indices[task_name].extend(range(offset, offset + length))
            offset += length
        self.task_names = list(self.task_to_indices.keys())

    def __iter__(self):
        # Set random seed to ensure that all processes generate the same random sequence
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            for task in self.task_names:
                indices = self.task_to_indices[task]
                rand_indices = torch.randperm(len(indices), generator=g).tolist()
                self.task_to_indices[task] = [indices[i] for i in rand_indices]

        indices = []
        task_current_indices = {task: 0 for task in self.task_names}
        available_tasks = self.task_names.copy()
        while available_tasks:
            # Randomly choose a task, with the probability proportional to the number of remaining samples in the task
            task = available_tasks[0]
            task_indices = self.task_to_indices[task]
            start = task_current_indices[task] + self.rank
            if task in ["THUMOS14", "ActivityNet", "FineAction", "HACS"]:
                # using fake batch sampling to balance the batch size with other tasks
                end = start + 1 * self.num_replicas - self.rank
            else:
                end = start + self.batch_size * self.num_replicas - self.rank
            batch_indices = task_indices[start : end : self.num_replicas]
            task_indices_offset = self.task_to_indices[task][0]

            if (
                len(batch_indices) == 0
                or max(batch_indices) >= len(task_indices) + task_indices_offset
            ):
                available_tasks.remove(
                    task
                )  # drop insufficient samples for training # TODO: consider padding when validation
                continue

            if task in ["THUMOS14", "ActivityNet", "FineAction", "HACS"]:
                # batch_indices = batch_indices * self.batch_size
                # indices.extend(batch_indices)
                idx = batch_indices[0]
                indices.extend([idx] + [-1] * (self.batch_size - 1))
                # indices.extend([idx] * (self.batch_size ))
                # Update the current index for this task
                task_current_indices[task] += 1 * self.num_replicas

                # Remove the task if all its indices have been used
                if task_current_indices[task] + 1 * self.num_replicas > len(
                    task_indices
                ):  # judge whether the task is exhausted on all replicas
                    available_tasks.remove(task)
            else:
                indices.extend(batch_indices)

                # Update the current index for this task
                task_current_indices[task] += self.batch_size * self.num_replicas

                # Remove the task if all its indices have been used
                if task_current_indices[
                    task
                ] + self.batch_size * self.num_replicas > len(
                    task_indices
                ):  # judge whether the task is exhausted on all replicas
                    available_tasks.remove(task)

        # Ensure each process has the same number of samples
        if not self.drop_last:
            raise NotImplementedError(
                "Please use drop_last=True for distributed training"
            )
            self.num_samples = len(indices)
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            self.num_samples = len(indices) // self.batch_size * self.batch_size
            indices = indices[: self.num_samples]
        # print(f'Rank: {self.rank}, {len(indices)}, After:', indices)
        # Generate batches
        for i in range(0, len(indices), self.batch_size):
            yield indices[i : i + self.batch_size]

    def __len__(self):
        # Return the number of batches for this process
        return (
            self.num_samples + self.batch_size - 1
        ) // self.batch_size  # Avoids truncation errors

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch  # Store epoch for deterministic shuffling


class DistributedBatchTaskBalancedSampler(DistributedSampler):
    """
    This sampler is used to balance the batch size of each task, used for gradient accumulation
    e.g. If the batch size is 16, and task_samples = {task1: 1000, task2: 100}, then the indices will be sampled as:
    [task1_idx1, task1_idx2, ..., task1_idx160, task2_idx1, task2_idx2, ..., task2_idx16], with a scale factor of 1000/100 = 10
    and the actual batch size for this scenario is 16 * (10 + 1) = 176, and the accumulation step is 176 / 16 = 11
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=True,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.batch_size = batch_size
        self.task_to_indices = defaultdict(list)

        # Allocate indices for each task
        offset = 0
        for length in dataset.unified_dataset_lengths:
            task_name = dataset[offset]["task_name"]  # Access task_name from dataset
            self.task_to_indices[task_name].extend(range(offset, offset + length))
            offset += length
        self.task_names = list(self.task_to_indices.keys())
        # calculate the number of samples for each task
        # self.task_samples = {task: len(self.task_to_indices[task]) for task in self.task_names}
        # # calculate the scale factor for each task
        # self.task_scale_factors = {task: self.task_samples[task] // min(self.task_samples.values()) for task in self.task_names}

    def __iter__(self):
        # Set random seed to ensure that all processes generate the same random sequence
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            for task in self.task_names:
                indices = self.task_to_indices[task]
                rand_indices = torch.randperm(len(indices), generator=g).tolist()
                self.task_to_indices[task] = [indices[i] for i in rand_indices]

        indices = []
        task_current_indices = {task: 0 for task in self.task_names}
        available_tasks = self.task_names.copy()
        # use a pointer to indicate the current index in scale factor
        task_begin_index = {task: 0 for task in self.task_names}
        task_end_index = {task: 0 for task in self.task_names}
        while available_tasks:
            task = available_tasks[0]
            task_indices = self.task_to_indices[task]
            start = task_current_indices[task] + self.rank
            if task in ["THUMOS14", "ActivityNet", "FineAction", "HACS"]:
                # using fake batch sampling to balance the batch size with other tasks
                end = start + 1 * self.num_replicas - self.rank
            else:
                end = start + self.batch_size * self.num_replicas - self.rank
            batch_indices = task_indices[start : end : self.num_replicas]
            task_indices_offset = self.task_to_indices[task][0]

            if (
                len(batch_indices) == 0
                or max(batch_indices) >= len(task_indices) + task_indices_offset
            ):
                available_tasks.remove(
                    task
                )  # drop insufficient samples for training # TODO: consider padding when validation
                continue

            if task in ["THUMOS14", "ActivityNet", "FineAction", "HACS"]:
                # batch_indices = batch_indices * self.batch_size
                # indices.extend(batch_indices)
                idx = batch_indices[0]
                indices.extend([idx] + [-1] * (self.batch_size - 1))
                # indices.extend([idx] * (self.batch_size ))
                # Update the current index for this task
                task_current_indices[task] += 1 * self.num_replicas

                # Remove the task if all its indices have been used
                if task_current_indices[task] + 1 * self.num_replicas > len(
                    task_indices
                ):  # judge whether the task is exhausted on all replicas
                    task_end_index[task] = len(indices)
                    available_tasks.remove(task)

            else:
                indices.extend(batch_indices)

                # Update the current index for this task
                task_current_indices[task] += self.batch_size * self.num_replicas

                # Remove the task if all its indices have been used
                if task_current_indices[
                    task
                ] + self.batch_size * self.num_replicas > len(
                    task_indices
                ):  # judge whether the task is exhausted on all replicas
                    task_end_index[task] = len(indices)
                    available_tasks.remove(task)

        # Ensure each process has the same number of samples
        if not self.drop_last:
            raise NotImplementedError(
                "Please use drop_last=True for distributed training"
            )
            self.num_samples = len(indices)
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            self.num_samples = len(indices) // self.batch_size * self.batch_size
            indices = indices[: self.num_samples]
        # rearange the indices according to the task_scale_factors
        # print(f'Rank: {self.rank}, {len(indices)}, After:', indices)
        # Generate batches

        last_index = 0
        task_samples = {}
        for task in self.task_names:
            task_samples[task] = task_end_index[task] - last_index
            last_index = task_end_index[task]
        task_scale_factors = {
            task: task_samples[task] // min(task_samples.values())
            for task in self.task_names
        }
        # rearange the indices according to the task_scale_factors
        indices_rearranged = []
        # calculate the begin index for each task
        for it, task in enumerate(self.task_names):
            if it == 0:
                task_begin_index[task] = 0
            else:
                task_begin_index[task] = task_end_index[self.task_names[it - 1]]
        while len(indices_rearranged) < len(indices):
            for task in self.task_names:
                if task_end_index[task] <= task_begin_index[task]:
                    continue
                for i in range(task_scale_factors[task]):
                    for j in range(self.batch_size):
                        indices_rearranged.extend(
                            [indices[task_begin_index[task] + i * self.batch_size + j]]
                        )
                task_begin_index[task] += task_scale_factors[task] * self.batch_size
        # with open(f'indices_rearranged{self.rank}.log', 'w') as f:
        #     f.write(str(len(indices_rearranged)))
        #     f.write(str(indices_rearranged))
        for i in range(0, len(indices_rearranged), self.batch_size):
            yield indices_rearranged[i : i + self.batch_size]

    def __len__(self):
        # Return the number of batches for this process
        return (
            self.num_samples + self.batch_size - 1
        ) // self.batch_size  # Avoids truncation errors

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch  # Store epoch for deterministic shuffling


class DistributedBatchTaskUniqueSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=True,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.batch_size = batch_size
        self.task_to_indices = defaultdict(list)

        # Allocate indices for each task
        offset = 0
        for length in dataset.unified_dataset_lengths:
            task_name = dataset[offset]["task_name"]  # Access task_name from dataset
            self.task_to_indices[task_name].extend(range(offset, offset + length))
            offset += length
        self.task_names = list(self.task_to_indices.keys())

    def __iter__(self):
        # Set random seed to ensure that all processes generate the same random sequence
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            for task in self.task_names:
                indices = self.task_to_indices[task]
                rand_indices = torch.randperm(len(indices), generator=g).tolist()
                self.task_to_indices[task] = [indices[i] for i in rand_indices]

        indices = []
        task_current_indices = {task: 0 for task in self.task_names}
        available_tasks = self.task_names.copy()
        random.seed(self.epoch)
        # info = ""
        weights_factor = {task: 1 for task in self.task_names}
        weights_factor["THUMOS14"] = self.batch_size
        weights_factor["ActivityNet"] = self.batch_size
        weights_factor["FineAction"] = self.batch_size
        weights_factor["HACS"] = self.batch_size
        while available_tasks:
            # Randomly choose a task, with the probability proportional to the number of remaining samples in the task

            task = random.choices(
                available_tasks,
                weights=[
                    weights_factor[task]
                    * (len(self.task_to_indices[task]) - task_current_indices[task])
                    for task in available_tasks
                ],
            )[0]
            task_indices = self.task_to_indices[task]
            start = task_current_indices[task] + self.rank
            if task in ["THUMOS14", "ActivityNet", "FineAction", "HACS"]:
                # using fake batch sampling to balance the batch size with other tasks
                end = start + 1 * self.num_replicas - self.rank
            else:
                end = start + self.batch_size * self.num_replicas - self.rank
            batch_indices = task_indices[start : end : self.num_replicas]
            task_indices_offset = self.task_to_indices[task][0]

            # with open(f'sampler{self.rank}.log', 'a') as f:
            # f.write(info)
            if (
                len(batch_indices) == 0
                or max(batch_indices) >= len(task_indices) + task_indices_offset
            ):
                available_tasks.remove(
                    task
                )  # drop insufficient samples for training # TODO: consider padding when validation
                continue

            if task in ["THUMOS14", "ActivityNet", "FineAction", "HACS"]:
                # batch_indices = batch_indices * self.batch_size
                # indices.extend(batch_indices)
                idx = batch_indices[0]
                indices.extend([idx] + [-1] * (self.batch_size - 1))
                # indices.extend([idx] * (self.batch_size ))
                # Update the current index for this task
                task_current_indices[task] += 1 * self.num_replicas

                # Remove the task if all its indices have been used
                if task_current_indices[task] + 1 * self.num_replicas > len(
                    task_indices
                ):  # judge whether the task is exhausted on all replicas
                    available_tasks.remove(task)
            else:
                indices.extend(batch_indices)

                # Update the current index for this task
                task_current_indices[task] += self.batch_size * self.num_replicas

                # Remove the task if all its indices have been used
                if task_current_indices[
                    task
                ] + self.batch_size * self.num_replicas > len(
                    task_indices
                ):  # judge whether the task is exhausted on all replicas
                    available_tasks.remove(task)

        # Ensure each process has the same number of samples
        if not self.drop_last:
            raise NotImplementedError(
                "Please use drop_last=True for distributed training"
            )
            self.num_samples = len(indices)
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            self.num_samples = len(indices) // self.batch_size * self.batch_size
            indices = indices[: self.num_samples]
        # print(f'Rank: {self.rank}, {len(indices)}, After:', indices)
        # Generate batches
        for i in range(0, len(indices), self.batch_size):
            yield indices[i : i + self.batch_size]

    def __len__(self):
        # Return the number of batches for this process
        return (
            self.num_samples + self.batch_size - 1
        ) // self.batch_size  # Avoids truncation errors

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch  # Store epoch for deterministic shuffling
