# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..base_trainers.perframe_det_trainer import do_perframe_det_train, do_ek100_perframe_det_train


from ..engines import TRAINERS as registry
@registry.register('LSTR')
def do_lstr_train(cfg,
                  data_loaders,
                  model,
                  criterion,
                  optimizer,
                  scheduler,
                  ema,
                  device,
                  checkpointer,
                  logger):
    if cfg.DATA.DATA_NAME != 'EK100':
        do_perframe_det_train(cfg,
                              data_loaders,
                              model,
                              criterion,
                              optimizer,
                              scheduler,
                              ema,
                              device,
                              checkpointer,
                              logger)
    else:
        do_ek100_perframe_det_train(cfg,
                              data_loaders,
                              model,
                              criterion,
                              optimizer,
                              scheduler,
                              ema,
                              device,
                              checkpointer,
                              logger)
