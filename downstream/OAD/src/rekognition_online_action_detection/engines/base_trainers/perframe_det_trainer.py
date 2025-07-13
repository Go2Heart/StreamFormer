# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from rekognition_online_action_detection.evaluation import compute_result

import sys
sys.path.append('')

try:
    from external.rulstm.RULSTM.utils import (get_marginal_indexes, marginalize, softmax,
                                                        topk_accuracy_multiple_timesteps,
                                                        topk_recall_multiple_timesteps,
                                                        tta)
except:
    raise ModuleNotFoundError

def do_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          ema,
                          device,
                          checkpointer,
                          logger):
    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    best_mAP = 0.0
    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        fut_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            if not training:
                ema.apply_shadow()

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    if cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES <= 0:
                        det_target = data[-1].to(device)

                        det_score = model(*[x.to(device) for x in data[:-1]])
                        det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_loss = criterion['MCE'](det_score, det_target)
                        det_losses[phase] += det_loss.item() * batch_size
                    else:
                        det_target, fut_target = data[-1][0].to(device), data[-1][1].to(device)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        fut_target = fut_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_scores, fut_scores = model(*[x.to(device) for x in data[:-1]])
                        for i, det_score in enumerate(det_scores):
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                det_loss = 0.2 * criterion['MCE'](det_score, det_target)
                            else:
                                det_loss += (0.8*i - 0.6) * criterion['MCE'](det_score, det_target)
                        det_losses[phase] += det_loss.item() * batch_size
                        for i, fut_score in enumerate(fut_scores):
                            fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                fut_loss = 0.1 * criterion['MCE'](fut_score, fut_target)
                            else:
                                fut_loss += 0.1 * criterion['MCE'](fut_score, fut_target)
                        fut_losses[phase] += fut_loss.item() * batch_size
                        det_loss += fut_loss
                            

                    # Output log for current batch
                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })

                    if training:
                        optimizer.zero_grad()
                        det_loss.backward()
                        optimizer.step()
                        ema.update()
                        scheduler.step()
                    else:
                        # Prepare for evaluation
                        det_score = det_score.softmax(dim=1).cpu().tolist()
                        det_target = det_target.cpu().tolist()
                        det_pred_scores.extend(det_score)
                        det_gt_targets.extend(det_target)
        end = time.time()

        # Output log for current epoch
        log = []
        log.append('Epoch {:2}'.format(epoch))
        log.append('train det_loss: {:.5f}'.format(
            det_losses['train'] / len(data_loaders['train'].dataset),
        ))
        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            det_result = compute_result['perframe'](
                cfg,
                det_gt_targets,
                det_pred_scores,
            )
            log.append('test det_loss: {:.5f}, det_mAP: {:.5f}'.format(
                det_losses['test'] / len(data_loaders['test'].dataset),
                det_result['mean_AP'],
            ))
        log.append('running time: {:.2f} sec'.format(
            end - start,
        ))
        logger.info(' | '.join(log))
        
        is_best = False
        if det_result['mean_AP'] > best_mAP:
            best_mAP = det_result['mean_AP']
            is_best = True
        
        checkpointer.save(epoch, model, optimizer, is_best=is_best, save_best_only=True)
        #checkpointer.save(epoch, model, optimizer)

        if not training:
            ema.restore()

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()

    logger.info('Best det_mAP: {:.5f}'.format(best_mAP))

def do_ek100_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          ema,
                          device,
                          checkpointer,
                          logger):
    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        fut_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            if not training:
                ema.apply_shadow()

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    if cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES <= 0:
                        det_target = data[-1].to(device)
                        det_score = model(*[x.to(device) for x in data[:-1]])
                        det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        det_loss = criterion['MCE'](det_score, det_target)
                        det_losses[phase] += det_loss.item() * batch_size
                    else:
                        det_target, noun_target, verb_target, \
                        fut_target, fut_noun_target, fut_verb_target = data[-1][0].to(device), data[-1][1].to(device), data[-1][2].to(device), \
                                                                                data[-1][3].to(device), data[-1][4].to(device), data[-1][5].to(device)
                        det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        noun_target = noun_target.reshape(-1, noun_target.shape[-1])
                        verb_target = verb_target.reshape(-1, verb_target.shape[-1])
                        fut_target = fut_target.reshape(-1, cfg.DATA.NUM_CLASSES)
                        fut_noun_target = fut_noun_target.reshape(-1, fut_noun_target.shape[-1])
                        fut_verb_target = fut_verb_target.reshape(-1, fut_verb_target.shape[-1])

                        det_scores, fut_scores, noun_score, fut_noun_score, verb_score, fut_verb_score = model(*[x.to(device) for x in data[:-1]])
                        for i, det_score in enumerate(det_scores):
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                det_loss = 0.2 * criterion['MCE_EQL'](det_score, det_target)
                            else:
                                det_loss += (0.8 * i - 0.6) * criterion['MCE_EQL'](det_score, det_target)
                        verb_score, noun_score = verb_score.reshape(-1, verb_score.shape[-1]), noun_score.reshape(-1, noun_score.shape[-1])
                        verb_loss = criterion['MCE'](verb_score, verb_target)
                        noun_loss = criterion['MCE'](noun_score, noun_target)
                        det_loss += (verb_loss + noun_loss)
                        det_losses[phase] += det_loss.item() * batch_size
                        for i, fut_score in enumerate(fut_scores):
                            fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            if i == 0:
                                fut_loss = 0.1 * criterion['MCE_EQL'](fut_score, fut_target)
                            else:
                                fut_loss += 0.1 * criterion['MCE_EQL'](fut_score, fut_target)
                        fut_verb_score, fut_noun_score = fut_verb_score.reshape(-1, fut_verb_score.shape[-1]), fut_noun_score.reshape(-1, fut_noun_score.shape[-1])
                        fut_verb_loss = criterion['MCE'](fut_verb_score, fut_verb_target)
                        fut_noun_loss = criterion['MCE'](fut_noun_score, fut_noun_target)
                        fut_loss += 0.1 * (fut_verb_loss + fut_noun_loss)
                        fut_losses[phase] += fut_loss.item() * batch_size

                        det_loss += fut_loss

                    # Output log for current batch
                    pbar.set_postfix({
                        'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                        'det_loss': '{:.5f}'.format(det_loss.item()),
                    })

                    if training:
                        optimizer.zero_grad()
                        det_loss.backward()
                        optimizer.step()
                        ema.update()
                        scheduler.step()
                    else:
                        # Prepare for evaluation
                        det_score = det_score.softmax(dim=1).cpu().tolist()
                        det_target = det_target.cpu().tolist()
                        det_pred_scores.extend(det_score)
                        det_gt_targets.extend(det_target)
        end = time.time()

        # Output log for current epoch
        log = []
        log.append('Epoch {:2}'.format(epoch))
        log.append('train det_loss: {:.5f}'.format(
            det_losses['train'] / len(data_loaders['train'].dataset),
        ))
        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            gt, pred = np.array(det_gt_targets), np.array(det_pred_scores)[:, 1:]
            action_labels = np.argmax(gt, axis=-1)
            action_pred = pred.reshape(-1, 1, pred.shape[-1])
            valid_index = list(np.where(action_labels != 0))[0]
            det_result = topk_recall_multiple_timesteps(action_pred[valid_index, ...], action_labels[valid_index], k=5)[0]
            log.append('test det_loss: {:.5f}, det_Recall: {:.5f}'.format(
                det_losses['test'] / len(data_loaders['test'].dataset),
                float(det_result) * 100,
            ))
        log.append('running time: {:.2f} sec'.format(
            end - start,
        ))
        logger.info(' | '.join(log))

        # Save checkpoint for model and optimizer
        checkpointer.save(epoch, model, optimizer)
        if not training:
            ema.restore()

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()

