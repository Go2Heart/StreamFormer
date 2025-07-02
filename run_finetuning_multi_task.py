import argparse
import datetime
import json
import os
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

import utils
from datasets import build_multi_task_dataset
from models import TimesformerForMultiTaskingSigLIP
from optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from sampler import (
    DistributedBatchTaskBalancedSampler,
    DistributedBatchTaskSequentialSampler,
    DistributedBatchTaskUniqueSampler,
)
from tools.finetune_tools import (
    train_one_epoch_multi_task,
    validation_one_epoch_multi_task,
)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import (
    construct_task_input_template,
    multiple_samples_collate,
    multiple_tasks_samples_collate,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Multi-tasking
    parser.add_argument("--multi_tasks_metadata", type=str, default=None)

    # Dataset
    parser.add_argument(
        "--data_set",
        default="Kinetics",
        choices=[
            "Kinetics",
            "Kinetics_sparse",
            "SSV2",
            "UCF101",
            "HMDB51",
            "image_folder",
            "mitv1_sparse",
            "ANet",
            "HACS",
            "ANet_interval",
            "HACS_interval",
            "THUMOS14",
        ],
        type=str,
        help="dataset",
    )
    parser.add_argument("--nb_classes", type=int, default=400)
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to the dataset"
    )
    parser.add_argument(
        "--prefix", type=str, default=None, help="Prefix for the dataset"
    )
    parser.add_argument("--split", type=str, default=",")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--sampling_rate", type=int, default=32)
    parser.add_argument("--test_num_segment", type=int, default=4)
    parser.add_argument("--test_num_crop", type=int, default=3)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--short_side_size", type=int, default=224)
    parser.add_argument("--label2id_path", type=str, default=None)
    parser.add_argument("--use_decord", action="store_true")
    parser.add_argument("--window_size", type=int, default=768)

    # Dataloader
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    # Model
    parser.add_argument(
        "--init_vit",
        type=str,
        choices=["clip", "siglip", "sam", "siglip_sam"],
        default="clip",
    )
    parser.add_argument(
        "--enable_causal_temporal", action="store_true"
    )  # TODO modify to the model, currently stored in the model's config
    parser.add_argument("--pretrained_model", type=str, default="timesformer-clip-base")
    # Distributed training
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--enable_deepspeed", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--zero_stage", default=0, type=int, help="ZeRO optimizer stage (default: 0)"
    )

    # Finetuning
    parser.add_argument(
        "--sampler_type", type=str, default="unique", choices=["balanced", "unique"]
    )
    parser.add_argument("--balance_sample_num_scale", type=float, default=2.0)
    parser.add_argument("--reprob", type=float, default=0)
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument(
        "--aa", type=str, default="rand-m7-n4-mstd0.5-inc1", help="Augmentation type"
    )
    parser.add_argument("--train_interpolation", type=str, default="bicubic")
    parser.add_argument("--update_freq", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_lr", type=float, default=0)
    parser.add_argument("--steps_per_print", default=100, type=int)
    parser.add_argument("--ckpt_path", default=None, help="load from checkpoint")

    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    parser.add_argument("--enable_lora_spatial", action="store_true")
    parser.add_argument("--frozen_spatial", action="store_true")
    parser.add_argument("--enable_multitask_collate", action="store_true")
    parser.add_argument(
        "--sample_type", default="uniform", choices=["uniform", "fixfps"]
    )
    parser.add_argument("--freeze_text_encoder", action="store_true")
    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--save_ceph_args", type=str, default=None)

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--test_best", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    if args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig

            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            ds_init = None
            print("Please 'pip install deepspeed'")
    else:
        ds_init = None

    return args, ds_init


def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def main(args, ds_init):
    utils.init_distributed_mode(args)
    if ds_init is not None:
        utils.create_internvideo2_ds_config(args)

    print(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.save_ceph_args is not None:
        ceph_args = yaml.load(open(args.save_ceph_args, "r"), Loader=yaml.FullLoader)
        print("Saving checkpoint to ceph with args:", ceph_args)
    else:
        ceph_args = {"use_ceph_checkpoint": False}

    with open(args.multi_tasks_metadata, "r") as f:
        multitask_metadata = yaml.load(f, Loader=yaml.FullLoader)
    dataset_metadata = multitask_metadata["datasets"]
    dataset_train, dataset_eval, multi_task_config = build_multi_task_dataset(
        dataset_metadata, args
    )

    if args.enable_multitask_collate:
        task_input_template = construct_task_input_template(dataset_train)
        collate_func = partial(
            multiple_tasks_samples_collate, task_input_template=task_input_template
        )
    elif args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    if args.sampler_type == "balanced":
        sampler_train = DistributedBatchTaskBalancedSampler(
            dataset_train,
            args.batch_size,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
    elif args.sampler_type == "unique":
        sampler_train = DistributedBatchTaskUniqueSampler(
            dataset_train,
            args.batch_size,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
    else:
        raise ValueError(
            f"Unknown sampler type: {args.sampler_type}. Please choose from ['balanced', 'unique']"
        )
    # args.update_freq = 3 #17 + 5 + 1
    if args.do_eval:
        sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            sampler=sampler_eval,
            # batch_sampler=sampler_eval,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=collate_func,
            pin_memory=args.pin_mem,
            persistent_workers=True,
        )
        print(
            "DataLoader for evaluation set is built with {} samples".format(
                len(data_loader_eval)
            )
        )

        loc_eval = None
    else:
        data_loader_eval = None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(args.log_dir)
        # save args
        with open(os.path.join(args.log_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        with open(os.path.join(args.log_dir, "multi_task_config.json"), "w") as f:
            json.dump(multi_task_config, f, default=convert_to_json_serializable)

    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_func,
        persistent_workers=True,
    )
    if args.init_vit == "clip":
        raise NotImplementedError("Not implemented yet")
    elif args.init_vit == "siglip":
        model = TimesformerForMultiTaskingSigLIP.from_pretrained(
            args.pretrained_model, multi_task_config, ignore_mismatched_sizes=True
        )
    elif args.init_vit == "sam":
        raise NotImplementedError("Not implemented yet")
    elif args.init_vit == "siglip_sam":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Unknown init_vit: {}".format(args.init_vit))
    model.to(device)
    if args.bf16:
        model = model.bfloat16()
    model.prepare_for_multi_tasks()
    if args.frozen_spatial:
        model.frozen_spatial()
    if args.enable_lora_spatial:
        # add lora to spatial attention
        model.add_lora_spatial()
    if args.ckpt_path:
        print(f"Loading ckpt at {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path)
        res = model.load_state_dict(ckpt["model"], strict=False)
        print(res)

    if args.freeze_text_encoder:
        print("Freezing text encoder:")
        for n, p in model.text_encoder.named_parameters():
            print("freeze:", n, p.shape)
            p.requires_grad = False

    n_parameters = sum(p.numel() for p in model.parameters())
    task_heads_parameters = sum(p.numel() for p in model.task_heads.parameters())
    print("Number of parameters: {}".format(n_parameters))
    print(
        "Number of task heads parameters: {} | {:.2f}%".format(
            task_heads_parameters, 100 * task_heads_parameters / n_parameters
        )
    )
    print(
        "Trainable parameters:{} | {}%".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            100
            * sum(p.numel() for p in model.parameters() if p.requires_grad)
            / n_parameters,
        )
    )

    model_without_ddp = model
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    args.lr = args.lr * total_batch_size * args.num_sample / 256
    args.min_lr = args.min_lr * total_batch_size * args.num_sample / 256
    args.warmup_lr = args.warmup_lr * total_batch_size * args.num_sample / 256
    print(
        "LR = {}, min_lr = {}, warmup_lr = {}".format(
            args.lr, args.min_lr, args.warmup_lr
        )
    )
    print("Total batch size: {}".format(total_batch_size))

    assigner = None

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model,
            args.weight_decay,
            (),
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print(
            "model.gradient_accumulation_steps() = %d"
            % model.gradient_accumulation_steps()
        )
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True
            )  # deviceids=[args.local_rank]
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=(),
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.warmup_lr,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    criterion = torch.nn.CrossEntropyLoss()
    print("Use bf16: {}".format(args.bf16))

    print(f"Start training from {args.start_epoch} for {args.epochs} epochs")
    max_accuracy = 0.0

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # data_loader_train.sampler.set_epoch(epoch)
            data_loader_train.batch_sampler.set_epoch(
                epoch
            )  # TODO check if this is correct
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch_multi_task(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            update_freq=args.update_freq,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            start_steps=0,
            log_writer=log_writer,
            bf16=args.bf16,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            multi_task_config=multi_task_config,
        )
        if (
            args.output_dir
            and args.save_ckpt
            and (epoch > 0 and epoch % 10 == 0 or epoch == args.epochs - 1)
            and utils.is_main_process()
        ):
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_name=f"epoch_{epoch}",
            )

        if args.output_dir and args.save_ckpt and utils.is_main_process():
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_name=f"epoch",
                ceph_args=ceph_args,
            )

        if data_loader_eval is not None:
            test_stats = validation_one_epoch_multi_task(
                data_loader_eval,
                model,
                device,
                multi_task_config,
                bf16=args.bf16,
                loc_eval=loc_eval,
            )
            if "acc1" in test_stats:
                if max_accuracy < test_stats["acc1"]:
                    max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        model_name="best",
                        ceph_args=ceph_args,
                    )
            print(f"Max accuracy: {max_accuracy:.2f}%")
            if log_writer is not None:
                if "acc1" in test_stats:
                    log_writer.update(
                        val_acc1=test_stats["acc1"], head="perf", step=epoch
                    )
                    log_writer.update(
                        val_acc5=test_stats["acc5"], head="perf", step=epoch
                    )
                    log_writer.update(
                        val_loss=test_stats["loss"], head="perf", step=epoch
                    )
                if "retrieval_top1" in test_stats:
                    log_writer.update(
                        val_R_1=test_stats["retrieval_top1"], head="perf", step=epoch
                    )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            # Convert stats to JSON serializable format
            json_serializable_stats = {
                k: convert_to_json_serializable(v) for k, v in log_stats.items()
            }
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(json_serializable_stats) + "\n")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Total training time: {total_time_str}")


if __name__ == "__main__":
    args, ds_init = get_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args, ds_init)
