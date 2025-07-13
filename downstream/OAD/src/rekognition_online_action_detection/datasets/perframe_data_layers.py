# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import time
import os.path as osp
from bisect import bisect_right

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd

from .datasets import DATA_LAYERS as registry


@registry.register('LSTRTHUMOS')
@registry.register('LSTRTVSeries')
class LSTRDataLayer(data.Dataset):

    def __init__(self, cfg, phase='train'):
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        # Anticipation choice
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        self.training = phase == 'train'

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            seed = np.random.randint(self.work_memory_length) if self.training else 0
            for work_start, work_end in zip(
                range(seed, target.shape[0], self.work_memory_length + self.anticipation_length),
                range(seed + self.work_memory_length, target.shape[0] - self.anticipation_length, self.work_memory_length + self.anticipation_length)):
                self.inputs.append([
                    session, work_start, work_end, target,
                ])

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        total_target = copy.deepcopy(target)
        # target = target[work_start: work_end][::self.work_memory_sample_rate]

        # Get work memory
        if self.future_num_samples > 0:
            target = target[work_start: work_end + self.anticipation_length]
            target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                     target[self.work_memory_length::self.anticipation_sample_rate]),
                                    axis=0)
            work_indices = np.arange(work_start, work_end).clip(0)
            work_indices = work_indices[::self.work_memory_sample_rate]
            work_visual_inputs = visual_inputs[work_indices]
            work_motion_inputs = motion_inputs[work_indices]
        else:
            target = target[work_start: work_end][::self.work_memory_sample_rate]
            work_indices = np.arange(work_start, work_end).clip(0)
            work_indices = work_indices[::self.work_memory_sample_rate]
            work_visual_inputs = visual_inputs[work_indices]
            work_motion_inputs = motion_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
                # memory_key_padding_mask[:last_zero] = -1e5
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_end + 1, total_target.shape[0] - 1), min(work_end + self.future_num_samples, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
        else:
            future_target = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            if future_target is not None:
                future_target = torch.as_tensor(future_target.astype(np.float32))
                return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, (target, future_target)
            return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target
        else:
            if future_target is not None:
                future_target = torch.as_tensor(future_target.astype(np.float32))
                return fusion_visual_inputs, fusion_motion_inputs, (target, future_target)
            return fusion_visual_inputs, fusion_motion_inputs, target

    def __len__(self):
        return len(self.inputs)

@registry.register('LSTREK100')
class LSTRDataLayer_(data.Dataset):

    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        # Anticipation choice
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        self.training = phase == 'train'
        self.clip_mixup_rate = 0.5
        self.clip_mixup_sample = 'uniform'
        self.plus_ratio = 0.25
        self.rgb = dict()
        self.flow = dict()
        self.noun = dict()
        self.target = dict()
        self.verb = dict()

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        path_to_data = 'external/rulstm/RULSTM/data/ek100/'
        segment_list = pd.read_csv(osp.join(path_to_data,
                                            'training.csv' if self.training else 'validation.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        segment_list['len_f'] = segment_list['end_f'] - segment_list['start_f']

        self.segment_by_cls = {cls: segment_list[segment_list['action'] == cls]
                               for cls in range(0, self.cfg.DATA.NUM_CLASSES - 1)}

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            verb_target = np.load(osp.join(self.data_root, self.target_perframe.replace('target', 'verb'), session + '.npy'))
            noun_target = np.load(osp.join(self.data_root, self.target_perframe.replace('target', 'noun'), session + '.npy'))
            self.rgb[session] = np.load(osp.join(self.data_root, self.visual_feature, session + '.npy'))
            self.flow[session] = np.load(osp.join(self.data_root, self.motion_feature, session + '.npy'))
            self.target[session] = target
            self.noun[session] = noun_target
            self.verb[session] = verb_target

            segments_per_session = segment_list[segment_list['video'] == session]
            for segment in segments_per_session.iterrows():
                start_tick = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                end_tick = int(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS)
                start_tick += np.random.randint(self.anticipation_length) if self.training else 0
                start_tick = min(start_tick, end_tick)
                work_end = start_tick - self.anticipation_length
                work_start = work_end - self.work_memory_length
                segments_before_current = segments_per_session[segments_per_session['end_f'] < segment[1]['start_f']]
                segments_before_end = segments_per_session[segments_per_session['end_f'] < segment[1]['end_f']]

                if work_start < 0:
                    continue
                self.inputs.append([
                    session, work_start, work_end,
                    target,
                    verb_target,
                    noun_target,
                    segments_before_end,
                    segments_before_current
                ])

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        # session, work_start, work_end, target = self.inputs[index]
        session, work_start, work_end, target, verb_target, noun_target, segments_work, segments_before = self.inputs[index]

        visual_inputs = self.rgb[session]
        motion_inputs = self.flow[session]

        # Get target
        total_target = copy.deepcopy(target)
        total_noun_target = copy.deepcopy(noun_target)
        total_verb_target = copy.deepcopy(verb_target)
        # target = target[work_start: work_end][::self.work_memory_sample_rate]

        # Get work memory
        if self.future_num_samples > 0:
            target = target[work_start: work_end + self.anticipation_length]
            noun_target = noun_target[work_start: work_end + self.anticipation_length]
            verb_target = verb_target[work_start: work_end + self.anticipation_length]
            target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                     target[self.work_memory_length::self.anticipation_sample_rate]),
                                    axis=0)
            noun_target = np.concatenate((noun_target[:self.work_memory_length:self.work_memory_sample_rate],
                                          noun_target[self.work_memory_length::self.anticipation_sample_rate]),
                                         axis=0)
            verb_target = np.concatenate((verb_target[:self.work_memory_length:self.work_memory_sample_rate],
                                          verb_target[self.work_memory_length::self.anticipation_sample_rate]),
                                         axis=0)
            work_indices = np.arange(work_start, work_end).clip(0)
            work_indices = work_indices[::self.work_memory_sample_rate]
            work_visual_inputs = visual_inputs[work_indices]
            work_motion_inputs = motion_inputs[work_indices]
        else:
            target = target[work_start: work_end][::self.work_memory_sample_rate]
            noun_target = noun_target[work_start: work_end][::self.work_memory_sample_rate]
            verb_target = verb_target[work_start: work_end][::self.work_memory_sample_rate]
            work_indices = np.arange(work_start, work_end).clip(0)
            work_indices = work_indices[::self.work_memory_sample_rate]
            work_visual_inputs = visual_inputs[work_indices]
            work_motion_inputs = motion_inputs[work_indices]

        # Mixclip++
        if self.training and self.clip_mixup_rate > 0:
            assert segments_work is not None
            segments_work = segments_work[segments_work['end_f'] / 30 * self.cfg.DATA.FPS > work_start]
            prob = None
            num_clip = int(len(segments_work) * 0.5)
            segments_to_mix = segments_work.sample(num_clip, replace=False,
                                                   weights=prob)
            for segment in segments_to_mix.iterrows():
                start, end = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS), int(np.ceil(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                random_cls = np.random.randint(0, self.cfg.DATA.NUM_CLASSES - 1)
                vid = segment[1]['video']
                # random_cls = segment[1]['action']
                assert vid == session
                segment_random_cls = self.segment_by_cls[random_cls]
                segment_random_cls = segment_random_cls[segment_random_cls['video'] != vid]
                valid_segments = segment_random_cls[segment_random_cls['len_f'] > segment[1]['len_f']]
                if len(valid_segments) == 0:
                    continue
                else:
                    sample_segment = valid_segments.sample(1)
                    new_vid = sample_segment['video'].values[0]
                    new_start_tick = int(sample_segment['start_f'] / 30 * self.cfg.DATA.FPS)
                    new_end_tick = int(sample_segment['end_f'] / 30 * self.cfg.DATA.FPS)

                    new_visual_inputs = self.rgb[new_vid]
                    new_motion_inputs = self.flow[new_vid]
                    new_target = self.target[new_vid]
                    new_noun = self.noun[new_vid]
                    new_verb = self.verb[new_vid]

                    sel_indices = np.where((work_indices >= start) & (work_indices <= end))
                    shift = np.random.randint(new_end_tick - new_start_tick - len(sel_indices) + 1)

                    work_visual_inputs[sel_indices] = self.plus_ratio * new_visual_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * \
                                                      work_visual_inputs[sel_indices]
                    work_motion_inputs[sel_indices] = self.plus_ratio * new_motion_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * \
                                                      work_motion_inputs[sel_indices]
                    target[sel_indices] = self.plus_ratio * new_target[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * target[sel_indices]
                    verb_target[sel_indices] = self.plus_ratio * new_verb[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * verb_target[sel_indices]
                    noun_target[sel_indices] = self.plus_ratio * new_noun[new_start_tick + shift:new_start_tick + shift + len(sel_indices)] + (1 - self.plus_ratio) * noun_target[sel_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
                # memory_key_padding_mask[:last_zero] = -1e5

            # Mixclip++
            if self.training and self.clip_mixup_rate > 0:
                assert segments_before is not None
                segments_before = segments_before[segments_before['end_f'] / 30 * self.cfg.DATA.FPS > long_start]
                if self.clip_mixup_sample == 'uniform':
                    prob = None
                elif self.clip_mixup_sample == 'by_length':
                    prob = (segments_before['len_f']).tolist()
                    prob = prob / np.sum(prob)
                else:
                    raise ValueError
                num_clip = int(len(segments_before) * self.clip_mixup_rate)
                segments_to_mixup = segments_before.sample(num_clip, replace=False,
                                                           weights=prob)
                for old_segment in segments_to_mixup.iterrows():
                    old_start_tick = int(old_segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                    old_end_tick = int(np.ceil(old_segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                    old_action = old_segment[1]['action']
                    # old_action = np.random.randint(0, self.cfg.DATA.NUM_CLASSES - 1)
                    old_vid = old_segment[1]['video']
                    assert old_vid == session
                    segment_same_class = self.segment_by_cls[old_action]
                    segment_same_class = segment_same_class[segment_same_class['video'] != old_vid]
                    valid_segments = segment_same_class[segment_same_class['len_f'] > old_segment[1]['len_f']]
                    if len(valid_segments) == 0:
                        continue
                    else:
                        sample_segment = valid_segments.sample(1)
                        new_vid = sample_segment['video'].values[0]
                        new_start_tick = int(sample_segment['start_f'] / 30 * self.cfg.DATA.FPS)
                        new_end_tick = int(sample_segment['end_f'] / 30 * self.cfg.DATA.FPS)

                        new_visual_inputs = self.rgb[new_vid]
                        new_motion_inputs = self.flow[new_vid]

                        sel_indices = np.where((long_indices >= old_start_tick) & (long_indices <= old_end_tick))

                        shift = np.random.randint(max(new_end_tick - new_start_tick - len(sel_indices) + 1, 1))
                        long_visual_inputs[sel_indices] = new_visual_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)]
                        long_motion_inputs[sel_indices] = new_motion_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)]

        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_end + 1, total_target.shape[0] - 1), min(work_end + self.future_num_samples, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
            future_noun_target = total_noun_target[future_indices]
            future_verb_target = total_verb_target[future_indices]
        else:
            future_target = None
            future_noun_target = None
            future_verb_target = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        # print(target.shape, noun_target.shape, verb_target.shape, future_target.shape, future_noun_target.shape, future_verb_target.shape)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            if future_target is not None:
                future_target = torch.as_tensor(future_target.astype(np.float32))
                verb_target = torch.as_tensor(verb_target.astype(np.float32))
                noun_target = torch.as_tensor(noun_target.astype(np.float32))
                future_noun_target = torch.as_tensor(future_noun_target.astype(np.float32))
                future_verb_target = torch.as_tensor(future_verb_target.astype(np.float32))
                return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, (target, noun_target, verb_target, future_target, future_noun_target, future_verb_target)
            return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, (target, noun_target, verb_target)
        else:
            if future_target is not None:
                future_target = torch.as_tensor(future_target.astype(np.float32))
                verb_target = torch.as_tensor(verb_target.astype(np.float32))
                noun_target = torch.as_tensor(noun_target.astype(np.float32))
                future_noun_target = torch.as_tensor(future_noun_target.astype(np.float32))
                future_verb_target = torch.as_tensor(future_verb_target.astype(np.float32))
                return fusion_visual_inputs, fusion_motion_inputs, (target, noun_target, verb_target, future_target, future_noun_target, future_verb_target)
            return fusion_visual_inputs, fusion_motion_inputs, (target, noun_target, verb_target)

    def __len__(self):
        return len(self.inputs)


@registry.register('LSTRBatchInferenceTHUMOS')
@registry.register('LSTRBatchInferenceTVSeries')
@registry.register('LSTRBatchInferenceEK100')
class LSTRBatchInferenceDataLayer(data.Dataset):

    def __init__(self, cfg, phase='test'):
        self.cfg = cfg
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES

        assert phase == 'test', 'phase must be `test` for batch inference, got {}'

        self.inputs = []
        if self.cfg.DATA.DATA_NAME != 'EK100':
            for session in self.sessions:
                target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
                for work_start, work_end in zip(
                    range(0, target.shape[0] + 1),
                    range(self.work_memory_length, target.shape[0] + 1)):
                    self.inputs.append([
                        session, work_start, work_end, target, target.shape[0]
                    ])
        else:
            for session in self.sessions:
                target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
                verb_target = np.load(osp.join(self.data_root, self.target_perframe.replace('target', 'verb'), session + '.npy'))
                noun_target = np.load(osp.join(self.data_root, self.target_perframe.replace('target', 'noun'), session + '.npy'))
                for work_start, work_end in zip(
                    range(0, target.shape[0] + 1),
                    range(self.work_memory_length, target.shape[0] + 1)):
                    self.inputs.append([
                        session, work_start, work_end, target, noun_target, verb_target, target.shape[0]
                    ])

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        if self.cfg.DATA.DATA_NAME != 'EK100':
            session, work_start, work_end, target, num_frames = self.inputs[index]
        else:
            session, work_start, work_end, target, noun_target, verb_target, num_frames = self.inputs[index]

        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get target
        total_target = copy.deepcopy(target)
        target = target[work_start: work_end][::self.work_memory_sample_rate]
        if self.cfg.DATA.DATA_NAME == 'EK100':
            noun_target = noun_target[work_start: work_end][::self.work_memory_sample_rate]
            verb_target = verb_target[work_start: work_end][::self.work_memory_sample_rate]
            noun_target = torch.as_tensor(noun_target.astype(np.float32))
            verb_target = torch.as_tensor(verb_target.astype(np.float32))

        # Get work memory
        # target = target[work_start: work_end][::self.work_memory_sample_rate]
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
                # memory_key_padding_mask[:last_zero] = -1e5
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_end + 1, total_target.shape[0] - 1), min(work_end + self.future_num_samples, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
        else:
            future_target = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        if self.cfg.DATA.DATA_NAME == 'EK100':
            target = (target, noun_target, verb_target)

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            if future_target is not None:
                future_target = torch.as_tensor(future_target.astype(np.float32))
                return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, (target,
                    future_target), session, work_indices, num_frames)
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target,
                    session, work_indices, num_frames)
        else:
            if future_target is not None:
                future_target = torch.as_tensor(future_target.astype(np.float32))
                return (fusion_visual_inputs, fusion_motion_inputs, (target, future_target),
                    session, work_indices, num_frames)
            return (fusion_visual_inputs, fusion_motion_inputs, target,
                    session, work_indices, num_frames)

    def __len__(self):
        return len(self.inputs)
