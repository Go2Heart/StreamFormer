import os
import time
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
import torch.nn.functional as F
import time
try:
    import deepspeed
    deepspeed_is_valid = True
except:
    print('Install deepspeed (optionally)')
    deepspeed_is_valid = False
    pass


def mil_nce(outputs, target):

    pos_logits = outputs.masked_fill(~(target.bool()), -6e4)
    pos_term = pos_logits.logsumexp(-1)  # (bsz, num_frames)
    neg_term = outputs.logsumexp(-1)  # (bsz, num_frames)
    loss = (-pos_term + neg_term).mean()  # (bsz,)

    return loss

def siglip(outputs, target):

    labels = target.masked_fill(target == 0, -1)
    loss = -F.logsigmoid(labels * outputs).sum() / outputs.shape[0]

    return loss

def train_multi_task_batch(model, samples, multi_task_input, output_hidden_states=False):
    loss, logits = model(pixel_values=samples, multi_task_input=multi_task_input, output_hidden_states=output_hidden_states)
    if isinstance(multi_task_input['task_name'], list):
        pass
    else:
        loss = loss[multi_task_input['task_name']]
        logits = logits[multi_task_input['task_name']]
    return loss, logits

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    if hasattr(optimizer, "loss_scale"):
        return optimizer.loss_scale
    elif hasattr(optimizer, "cur_scale"):
        return optimizer.cur_scale
    else:
        ### TODO: For BF16, perhaps we do not need loss_scaling, return a default num
        return 1.0

@torch.no_grad()
def validation_one_epoch_retrieval(data_loader, model, device, ds=False, bf16=False, log_time=False):
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()
    if log_time: time3 = time.time()

    caption_features = []
    video_features = []

    for multi_task_input in metric_logger.log_every(data_loader, 10, header):
        
        if log_time: time0 = time.time()
        if log_time: print(f"Time to get batch: {time0 - time3}")
        captions = multi_task_input['task_input']['caption']
        videos = multi_task_input['task_input']['video']
        videos = videos.to(device, non_blocking=True)
        if log_time: time1 = time.time()
        if log_time: print(f"Time to move data to GPU: {time1 - time0}")
        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            batch_video_features, batch_text_features = model(pixel_values=videos, captions=captions)
        else:
            with torch.cuda.amp.autocast():
                batch_video_features, batch_text_features = model(pixel_values=videos, captions=captions, inference=True)
        
        caption_features.append(batch_text_features)
        video_features.append(batch_video_features)

    caption_features = torch.cat(caption_features, dim=0)
    video_features = torch.cat(video_features, dim=0)
    
    caption_features = F.normalize(caption_features, dim=-1)
    video_features = F.normalize(video_features, dim=-1)

    sims = caption_features @ video_features.T 
    gt = torch.arange(len(sims)).cuda()

    r1_indices = torch.topk(sims, k=1, dim=-1).indices.squeeze()
    r1 = torch.sum(r1_indices == gt) / len(sims) * 100 
    print('text2video Recall@1: ', r1)
@torch.no_grad()
def validation_one_epoch(data_loader, model, device, ds=False, bf16=False, log_time=False):
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()
    if log_time: time3 = time.time()
    for batch in metric_logger.log_every(data_loader, 10, header):
        if log_time: time0 = time.time()
        if log_time: print(f"Time to get batch: {time0 - time3}")
        videos = batch[0]
        target = batch[1]
        # target_label = batch[1]
        # target = torch.tensor([model.config.label2id[label] for label in target_label], dtype=torch.long)
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if log_time: time1 = time.time()
        if log_time: print(f"Time to move data to GPU: {time1 - time0}")
        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos).logits
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos).logits
                loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if log_time: time2 = time.time()
        if log_time: print(f"Time to compute output: {time2 - time1}")
        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if log_time: time3 = time.time()
        if log_time: print(f"Time to update meters: {time3 - time2}")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def multi_segment_proposal(prob, timestamps, factor=0.5, at_least_one=True):
    # if it's over 0.5, then it's a positive sample
    # if it's under 0.5, then it's a negative sample
    # return the timestamps list
    timestamps_list = []
    vis_idx = {}
    for idx in range(prob.shape[0]):
        if prob[idx] > factor and idx not in vis_idx:
            # find the start and end of the segment
            vis_idx[idx] = 1

            start = idx
            while start > 0 and prob[start - 1] > factor and (start - 1 not in vis_idx):
                start -= 1
                vis_idx[start] = 1

            end = idx
            while end < prob.shape[-1] - 1 and prob[end + 1] > factor and (end + 1 not in vis_idx):
                end += 1
                vis_idx[end] = 1
            timestamps_list.append([timestamps[start], timestamps[end], 1]) # add fake score '1' for now
    # if empty, then fall into max_prob_proposal
    if len(timestamps_list) == 0 and at_least_one:
        max_idx = torch.argmax(prob).item()
        max_value = prob[max_idx].item()
        threshold = factor * max_value
        start = max_idx
        while start > 0 and prob[start] > threshold:
            start -= 1
        end = max_idx
        while end < prob.shape[-1] - 1 and prob[end] > threshold:
            end += 1
        timestamps_list.append([timestamps[start], timestamps[end], prob[idx].item()])
        return timestamps_list
    elif len(timestamps_list) == 0 and not at_least_one:
        return None
    else:
        return timestamps_list

def threshold_prob_proposal(prob, timestamps, factor=0.7, output_score=False):

    max_idx = torch.argmax(prob).item()
    max_value = prob[max_idx].item()
    threshold = factor * max_value

    start = max_idx
    while start > 0 and prob[start] > threshold:
        start -= 1
    end = max_idx
    while end < prob.shape[-1] - 1 and prob[end] > threshold:
        end += 1
    # print(timestamps)
    proposal = [timestamps[start], timestamps[end]]
    if output_score:
        return [[timestamps[start], timestamps[end], prob[max_idx].item()]]
    return proposal


def iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    return max(min1 - max0, 0) / (max1 - min0)



@torch.no_grad()
def validation_one_epoch_grounding(data_loader, model, device, ds=False, bf16=False, log_time=False):
    
    criterion = siglip
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()
    if log_time: time3 = time.time()
    for batch_id, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if log_time: time0 = time.time()
        if log_time: print(f"Time to get batch: {time0 - time3}")
        videos = batch[0]
        target = batch[1] # (B, T)
        texts = batch[2]
        durations = batch[3]
        start = batch[4]
        end = batch[5]
        timestamps = batch[6]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if log_time: time1 = time.time()
        if log_time: print(f"Time to move data to GPU: {time1 - time0}")
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos, texts).logits
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos, texts).logits
                loss = criterion(output, target)

        if log_time: time2 = time.time()
        if log_time: print(f"Time to compute output: {time2 - time1}")
        batch_size = videos.shape[0]
        
        metric_logger.update(loss=loss.item())

        prob_factors = [0.5, 0.6, 0.7]
        results = {thresh: {0.3: 0, 0.5: 0, 0.7: 0} for thresh in prob_factors}
        total_counts = videos.shape[0]

        for factor in prob_factors:
            for idx in range(videos.shape[0]):

                prob = output[idx].sigmoid()
                proposal = threshold_prob_proposal(prob, timestamps[idx], factor=factor)
                gt_segment = [start[idx], end[idx]]
                current_iou = iou(gt_segment, proposal)
                metric_logger.meters[f'mIoU_{factor}'].update(current_iou, n=1)

                for c_iou in [0.3, 0.5, 0.7]:
                    if(current_iou >= c_iou):
                        results[factor][c_iou] += 1

                if batch_id % 50 == 0 and idx == 0:
                    print(prob, target[idx])
                    print(round(durations[idx].item(), 1), [round(proposal[0].item(), 1), round(proposal[1].item(), 1)], [gt_segment[0].item(), gt_segment[1].item()], current_iou.item())
                    print(results[factor], total_counts)

            metric_logger.meters[f'R1_3_{factor}'].update(results[factor][0.3] / total_counts, n=total_counts)
            metric_logger.meters[f'R1_5_{factor}'].update(results[factor][0.5] / total_counts, n=total_counts)
            metric_logger.meters[f'R1_7_{factor}'].update(results[factor][0.7] / total_counts, n=total_counts)


        if log_time: time3 = time.time()
        if log_time: print(f"Time to update meters: {time3 - time2}")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for factor in prob_factors:
        print(f"* Threshold {factor} : \n R@1_0.3: {metric_logger.meters[f'R1_3_{factor}'].global_avg:.3f}, "
            f"R@1_0.5: {metric_logger.meters[f'R1_5_{factor}'].global_avg:.3f}, "
            f"R@1_0.7: {metric_logger.meters[f'R1_7_{factor}'].global_avg:.3f}, "
            f"mIoU: {metric_logger.meters[f'mIoU_{factor}'].global_avg:.3f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
	
def train_one_epoch_multi_task(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
        start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
        num_training_steps_per_epoch=None, update_freq=None,
        bf16=False,
        multi_task_config=None,
    ):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    
    multi_task_list = []
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    for data_iter_step, multi_task_input in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]


        assert len(set(multi_task_input['task_name'])) == 1
        task_name = multi_task_input['task_name'][0]
        multi_task_input['task_name'] = task_name
        print("Current task name: ", task_name)
        if task_name not in multi_task_list:
            multi_task_list.append(task_name)
        task_input = multi_task_input['task_input']
        if task_name == 'SSV2' or task_name == 'Kinetics':
            # classification task
            samples = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            multi_task_input['task_input']['label'] = targets
            if loss_scaler is None:
                samples = samples.bfloat16() if bf16 else samples.half()
                loss, output = train_multi_task_batch(model, samples, multi_task_input)
            else:
                with torch.cuda.amp.autocast():
                    loss, output = train_multi_task_batch(model, samples, multi_task_input)
        elif task_name in ['MSRVTT', 'WebVid', 'TaskRetrieval']:
            # retrieval task
            samples = task_input['video'].to(device, non_blocking=True)
            if loss_scaler is None:
                samples = samples.bfloat16() if bf16 else samples.half()
                loss, output = train_multi_task_batch(
                    model, samples, multi_task_input)
            else:
                with torch.cuda.amp.autocast():
                    loss, output = train_multi_task_batch(
                        model, samples, multi_task_input)
        elif task_name == 'CharadesSTA':
            # grounding task
            samples = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            texts = task_input['caption']
            # durations = task_input['duration']
            # timestamps = task_input['timestamps']
            # s = task_input['start_time']
            # e = task_input['end_time']
            multi_task_input['task_input']['training'] = True
            if loss_scaler is None:
                samples = samples.bfloat16() if bf16 else samples.half()
                loss, output = train_multi_task_batch(
                    model, samples, multi_task_input)
            else:
                with torch.cuda.amp.autocast():
                    loss, output = train_multi_task_batch(
                        model, samples, multi_task_input)
        elif task_name in ['QVHighlights', 'TaCoS', 'ActivityNetCaptions', 'DiDeMo', 'QuerYD', 'TaskGrounding']:
            samples = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            texts = task_input['caption']
            # relevant_windows = task_input['relevant_windows']
            # gt_segments = task_input['gt_segments']
            # sample_ratio = task_input['sample_ratio']
            multi_task_input['task_input']['training'] = True
            # fps = task_input['fps']
            if loss_scaler is None:
                samples = samples.bfloat16() if bf16 else samples.half()
                loss, output = train_multi_task_batch(
                    model, samples, multi_task_input)
            else:
                with torch.cuda.amp.autocast():
                    loss, output = train_multi_task_batch(
                        model, samples, multi_task_input)
        elif task_name in ['THUMOS14', 'ActivityNet', 'FineAction', 'HACS', 'ActivityNetGrounding', 'FineActionGrounding', 'HACSGrounding','TaskLocalization']:
            # DEBUG: Hard code the batch size to 1 for THUMOS14, ignoring all other batch size settings
            samples = task_input['video'].to(device, non_blocking=True)
            if loss_scaler is None:
                samples = samples.bfloat16() if bf16 else samples.half()
                loss, output = train_multi_task_batch(model, samples, multi_task_input)
            else:
                with torch.cuda.amp.autocast():
                    loss, output = train_multi_task_batch(
                        model, samples, multi_task_input)
        elif task_name in ["YoutubeVIS", "LVVIS", "COCOPseudoVIS", "TaskVIS", "MEVIS", "ReferYoutubeVOS", "RefCOCOPseudo", "TaskReferVOS"]:
            samples = task_input['video'].to(device, non_blocking=True)
            if loss_scaler is None:
                samples = samples.bfloat16() if bf16 else samples.half()
                loss, output = train_multi_task_batch(model, samples, multi_task_input)
            else:
                with torch.cuda.amp.autocast():
                    loss, output = train_multi_task_batch(model, samples, multi_task_input)
        else:
            raise NotImplementedError(f"{task_name} is not supported in multi-task training")


        
        if isinstance(loss, dict):
            loss_value = {}
            for k,v in loss.items():
                loss_value[k] = v.item()
                if not math.isfinite(loss_value[k]):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)
            loss = sum(loss.values())
        else:
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        if task_name == 'SSV2' or task_name == 'Kinetics':
            class_acc = (output.max(-1)[-1] == targets).float().mean()
            metric_logger.update(class_acc=class_acc)
        elif task_name in ['MSRVTT', 'WebVid']:
            ground_truth = torch.arange(len(output), dtype=torch.long, device=output.device)
            ret_acc = (output.max(-1)[-1] == ground_truth).float().mean()
            metric_logger.update(ret_acc=ret_acc)
        elif task_name in ['CharadesSTA','QVHighlights','TaCoS', 'ActivityNetCaptions', 'DiDeMo', 'QuerYD', 'TaskGrounding']:
            pass        
        else:
            pass
        
        metric_logger.update(**{f'{task_name}_loss': loss_value})  # task loss
        metric_logger.update(loss=loss_value) # total loss
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(**{f'{task_name}_loss': loss_value}, head="loss")
                
            log_writer.update(**{f'{task_name}_loss_scale': loss_scale_value}, head="loss")
            if task_name == 'SSV2' or task_name == 'Kinetics':
                log_writer.update(class_acc=class_acc, head="loss")
            elif task_name in ['MSRVTT', 'WebVid']:
                log_writer.update(ret_acc=ret_acc, head="loss")

            
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch_multi_task(data_loader, model, device, multi_task_config, bf16=False, log_time=False, loc_eval=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()
    
    if log_time: time3 = time.time()
    caption_features = []
    video_features = []
    jsonl_data = []
    avg_back_percent = 0
    back_count = 0
    for batch_id, multi_task_input in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if log_time: time0 = time.time()
        if log_time: print(f"Time to get batch: {time0 - time3}")
        
        task_name = multi_task_input['task_name'][0]
        task_input = multi_task_input['task_input']
        multi_task_input['task_name'] = task_name
        
        
        if task_name == 'SSV2' or task_name == 'Kinetics':
            videos = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            multi_task_input['task_input']['label'] = targets
        elif task_name in ['MSRVTT', 'WebVid']:
            videos = task_input['video'].to(device, non_blocking=True)
            captions = task_input['caption']
        elif task_name == 'CharadesSTA':
            videos = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            texts = task_input['caption']
            durations = task_input['duration']
            timestamps = task_input['timestamps']
            start_time = task_input['start_time']
            end_time = task_input['end_time']
            gt_segments = task_input['gt_segments']
            sample_ratio = task_input['sample_ratio']
            fps = task_input['fps']
        elif task_name == 'QVHighlights':
            videos = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            texts = task_input['caption']
            # relevant_windows = task_input['relevant_windows']
            # gt_segments = task_input['gt_segments']
            # sample_ratio = task_input['sample_ratio']
            # fps = task_input['fps']
            # timestamps = task_input['timestamps']
        elif task_name == 'TaCoS':
            videos = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            texts = task_input['caption']
        elif task_name == 'ActivityNetCaptions':
            videos = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            texts = task_input['caption']
        elif task_name == 'DiDeMo':
            videos = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            texts = task_input['caption']
        elif task_name == 'QuerYD':
            videos = task_input['video'].to(device, non_blocking=True)
            targets = task_input['label'].to(device, non_blocking=True)
            texts = task_input['caption']
        elif task_name == 'THUMOS14':
            videos = task_input['video'].to(device, non_blocking=True)
        elif task_name == 'YoutubeVIS':
            videos = task_input['video'].to(device, non_blocking=True)
        if log_time: time1 = time.time()
        if log_time: print(f"Time to move data to GPU: {time1 - time0}")
        
        # Compute output
        
        with torch.cuda.amp.autocast(enabled=bf16):
            if task_name == 'SSV2' or task_name == 'Kinetics':
                loss, logits = model(videos, multi_task_input=multi_task_input)[task_name]
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                
                batch_size = videos.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                
            elif task_name in ['MSRVTT', 'WebVid']:
                batch_video_features, batch_text_features = model(videos, multi_task_input=multi_task_input)[task_name]
                
                video_features.append(batch_video_features)
                caption_features.append(batch_text_features)
            elif task_name == 'CharadesSTA':
                multi_task_input['task_input']['training'] = False
                loss, logits = model(videos, multi_task_input=multi_task_input)[task_name]
                # loss = output.loss
                if loss is not None:
                    metric_logger.update(loss=loss.item())
                
                prob_factors = [0.5, 0.6, 0.7]
                results = {thresh: {0.3: 0, 0.5: 0, 0.7: 0} for thresh in prob_factors}
                total_counts = videos.shape[0]
                
                for factor in prob_factors:
                    for idx in range(videos.shape[0]):
                        prob = logits[idx].sigmoid()
                        proposal = threshold_prob_proposal(prob, timestamps[idx], factor=factor)
                        gt_segment = [start_time[idx], end_time[idx]]
                        current_iou = iou(gt_segment, proposal)

                        metric_logger.meters[f'mIoU_{factor}'].update(current_iou, n=1)
                        
                        for c_iou in [0.3, 0.5, 0.7]:
                            if(current_iou >= c_iou):
                                results[factor][c_iou] += 1
                        
                        if batch_id % 50 == 0 and idx == 0:
                            print(prob, targets[idx])
                            print(round(durations[idx].item(), 1), [round(proposal[0].item(), 1), round(proposal[1].item(), 1)], [gt_segment[0].item(), gt_segment[1].item()], current_iou.item())
                            print(results[factor], total_counts)

                    metric_logger.meters[f'R1_3_{factor}'].update(results[factor][0.3] / total_counts, n=total_counts)
                    metric_logger.meters[f'R1_5_{factor}'].update(results[factor][0.5] / total_counts, n=total_counts)
                    metric_logger.meters[f'R1_7_{factor}'].update(results[factor][0.7] / total_counts, n=total_counts)


                # results = {0.3: 0, 0.5: 0, 0.7: 0}
                # total_counts = videos.shape[0]

                # assert videos.shape[0] == len(logits)
                # for idx in range(videos.shape[0]):
                #     proposal = logits[idx]['segments'][0] / (sample_ratio[idx] * fps[idx])
                #     gt = gt_segments[idx][0] / (sample_ratio[idx] * fps[idx])
                #     current_iou = iou(proposal, gt)
                #     metric_logger.meters[f'mIoU'].update(current_iou, n=1)

                #     for c_iou in [0.3, 0.5, 0.7]:
                #         if current_iou >= c_iou:
                #             results[c_iou] += 1

                #     if batch_id % 10 == 0 and idx == 0:
                #         print(f"{durations[idx].item():.1f} [{proposal[0].item():.1f}, {proposal[1].item():.1f}] [{gt[0].item():.1f}, {gt[1].item():.1f}] {current_iou.item():.1f}")

                # metric_logger.meters[f'R1_3'].update(results[0.3] / total_counts, n=total_counts)
                # metric_logger.meters[f'R1_5'].update(results[0.5] / total_counts, n=total_counts)
                # metric_logger.meters[f'R1_7'].update(results[0.7] / total_counts, n=total_counts)
            elif task_name == 'QVHighlights':
                # propose the segments to files, leave evaluation to official evaluation script for now
                multi_task_input['task_input']['training'] = False
                loss, logits = model(videos, multi_task_input=multi_task_input)[task_name]
                
                
                for idx in range(videos.shape[0]):
                    probs = logits[idx].sigmoid() # if it's over 0.5, then it's a positive sample
                    proposed_segments = multi_segment_proposal(probs, timestamps[idx], factor=0.5)
                    jsonl_data.append({
                        'qid': int(multi_task_input['task_input']['qid'][idx]),
                        'query': multi_task_input['task_input']['caption'][idx],
                        'vid': multi_task_input['task_input']['vid'][idx],
                        'pred_relevant_windows': [
                            [float(start), float(end), float(score)] for start, end, score in proposed_segments    
                        ],
                        # 'pred_saliency_scores': [1.0] * len(proposed_segments),
                    })
            elif task_name == 'THUMOS14':
                # ================ActionFormer================ #
                output = model(videos, multi_task_input=multi_task_input)[task_name]
                num_vids = len(output)
                for vid_idx in range(num_vids):
                    if output[vid_idx]['segments'].shape[0] > 0:
                        results['video-id'].extend(
                            [output[vid_idx]['video_id']] *
                            output[vid_idx]['segments'].shape[0]
                        )
                        results['t-start'].append(output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(output[vid_idx]['segments'][:, 1])
                        results['label'].append(output[vid_idx]['labels'])
                        results['score'].append(output[vid_idx]['scores'])
            elif task_name == "YoutubeVIS":
                batch_size = len(multi_task_input['task_input']['mask_target'])
                for mask in multi_task_input['task_input']['mask_target']:
                    t, h, w = mask.shape
                    num_zeros = (mask == 0).sum().item()
                    total_elements = mask.numel()  # 8 * 720 * 1280

                    back_percent = num_zeros / total_elements
                    avg_back_percent += back_percent
                    back_count += 1
                    # print(f"Number of zeros: {num_zeros:,}")
                    # print(f"Total elements: {total_elements:,}")
                    # print(f"Percentage of zeros: {num_zeros/total_elements:.2%}")
    
                logits = model(videos, multi_task_input=multi_task_input)[task_name]
                numpy_logits = logits.permute(0,3,1,2).cpu().numpy()
                video_ids = multi_task_input['task_input']['video_id']
                for batch_num, vid in enumerate(video_ids):
                    np.save(f"vis_outputs/{vid}", numpy_logits[batch_num])
                    
                
        if log_time: time2 = time.time()
        if log_time: print(f"Time to compute output: {time2 - time1}")
        
        if log_time: time3 = time.time()
        if log_time: print(f"Time to update meters: {time3 - time2}")
    
    
    if 'Kinetics' in multi_task_config.keys():
        print('Kinetics: * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    # if 'SSV2' in multi_task_config.keys():
    #     print('SSV2: * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    
    if 'MSRVTT' in multi_task_config.keys():
        caption_features = torch.cat(caption_features, dim=0)
        video_features = torch.cat(video_features, dim=0)
        
        caption_features = F.normalize(caption_features, dim=-1)
        video_features = F.normalize(video_features, dim=-1)

        sims = caption_features @ video_features.T 
        gt = torch.arange(len(sims)).cuda()

        r1_indices = torch.topk(sims, k=1, dim=-1).indices.squeeze()
        retrieval_top1 = torch.sum(r1_indices == gt) / len(sims) * 100 
        metric_logger.meters['retrieval_top1'].update(retrieval_top1.item(), n=videos.shape[0])
        print('text2video Recall@1: ', retrieval_top1)
        print('MSRVTT: * Retrieval_top1 {ret_top1.global_avg:.3f}'
            .format(ret_top1=metric_logger.retrieval_top1))
    
    if 'WebVid' in multi_task_config.keys():
        caption_features = torch.cat(caption_features, dim=0)
        video_features = torch.cat(video_features, dim=0)
        
        caption_features = F.normalize(caption_features, dim=-1)
        video_features = F.normalize(video_features, dim=-1)

        sims = caption_features @ video_features.T 
        gt = torch.arange(len(sims)).cuda()

        r1_indices = torch.topk(sims, k=1, dim=-1).indices.squeeze()
        retrieval_top1 = torch.sum(r1_indices == gt) / len(sims) * 100 
        metric_logger.meters['retrieval_top1'].update(retrieval_top1.item(), n=videos.shape[0])
        print('text2video Recall@1: ', retrieval_top1)
        print('WebVid: * Retrieval_top1 {ret_top1.global_avg:.3f}'
            .format(ret_top1=metric_logger.retrieval_top1))

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
