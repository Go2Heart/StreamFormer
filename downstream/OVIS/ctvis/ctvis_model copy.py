import logging
import math

import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList, BitMasks

from ..mask2former import MaskFormer,TimesformerMaskFormer
from .utils import retry_if_cuda_oom
from .modeling.cl_plugin import build_cl_plugin
from .modeling.tracker import build_tracker

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class CTVISModel(MaskFormer):
    @configurable
    def __init__(
            self,
            num_frames,
            num_topk,
            num_clip_frames,
            to_cpu_frames,
            test_interpolate_chunk_size,
            test_instance_chunk_size,
            cl_plugin,
            tracker,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.cl_plugin = cl_plugin
        self.tracker = tracker

        self.num_frames = num_frames
        self.num_topk = num_topk
        self.num_clip_frames = num_clip_frames
        self.to_cpu_frames = to_cpu_frames
        self.test_interpolate_chunk_size = test_interpolate_chunk_size
        self.test_instance_chunk_size = test_instance_chunk_size

    @classmethod
    def from_config(cls, cfg):
        rets = TimesformerMaskFormer.from_config(cfg)  # noqa

        cl_plugin = build_cl_plugin(cfg)  # train
        tracker = build_tracker(cfg)  # inference

        num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        num_topk = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        num_clip_frames = cfg.TEST.NUM_CLIP_FRAMES
        to_cpu_frames = cfg.TEST.TO_CPU_FRAMES
        test_interpolate_chunk_size = cfg.TEST.TEST_INTERPOLATE_CHUNK_SIZE
        test_instance_chunk_size = cfg.TEST.TEST_INSTANCE_CHUNK_SIZE

        rets.update(
            num_frames=num_frames,
            num_topk=num_topk,
            num_clip_frames=num_clip_frames,
            to_cpu_frames=to_cpu_frames,
            test_interpolate_chunk_size=test_interpolate_chunk_size,
            test_instance_chunk_size=test_instance_chunk_size,
            cl_plugin=cl_plugin,
            tracker=tracker
        )

        return rets

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if self.training:
            return self.train_model(batched_inputs)
        else:
            return self.inference_model(batched_inputs)

    def pre_process(self, batched_inputs):
        images = []  # noqa
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images
 
    def train_model(self, batched_inputs):
        images = self.pre_process(batched_inputs)  # noqa

        # features = self.backbone(images.tensor)
        pixel_values = images.tensor
        pixel_values = pixel_values.reshape(-1, self.num_frames, *pixel_values.shape[1:]) #(B, T, 3, H, W)
        features = self.backbone(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        ) 


        det_outputs = self.sem_seg_head(features)

        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = []
            for video in batched_inputs:
                for frame in video["instances"]:
                    gt_instances.append(frame.to(self.device))
            targets = self.prepare_targets(gt_instances, images)
        else:
            targets = None

        # bipartite matching-based loss
        losses = self.criterion(det_outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        losses.update(self.cl_plugin.train_loss(
            det_outputs, gt_instances, self.criterion.matcher))

        return losses

    def inference_model(self, batched_inputs):
        images = self.pre_process(batched_inputs)

        # Avoid Out-of-Memory
        num_frames = len(images)
        to_store = self.device if num_frames <= self.to_cpu_frames else "cpu"
        import ipdb; ipdb.set_trace()
        # with torch.no_grad():
        #     # features = self.backbone(images.tensor)
        #     pixel_values = images.tensor
        #     pixel_values = pixel_values.reshape(-1, self.num_frames, *pixel_values.shape[1:]) #(B, T, 3, H, W)
        #     features = self.backbone(
        #         pixel_values=pixel_values,
        #         output_attentions=False,
        #         output_hidden_states=True,
        #         return_dict=True,
        #     )
        #     det_outputs = self.sem_seg_head(features)
        if num_frames <= self.num_clip_frames:  # noqa
            with torch.no_grad():
                # features = self.backbone(images.tensor)
                pixel_values = images.tensor
                pixel_values = pixel_values.reshape(-1, num_frames, *pixel_values.shape[1:]) #(B, T, 3, H, W)
                print(pixel_values.shape)
                features = self.backbone(
                    pixel_values=pixel_values,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                det_outputs = self.sem_seg_head(features)
        else:
            pred_logits, pred_masks, pred_embeds, pred_queries = [], [], [], []
            num_clips = math.ceil(
                num_frames / self.num_clip_frames)  # math.ceil up
            for i in range(num_clips):
                start_idx = i * self.num_clip_frames
                end_idx = (i + 1) * self.num_clip_frames
                clip_images_tensor = images.tensor[start_idx:end_idx, ...]
                with torch.no_grad():
                    # clip_features = self.backbone(clip_images_tensor)
                    clip_pixel_values = clip_images_tensor.reshape(1, -1, *clip_images_tensor.shape[1:]) #(B, T, 3, H, W)
                    clip_features = self.backbone(
                        pixel_values=clip_pixel_values,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    clip_outputs = self.sem_seg_head(clip_features)

                pred_logits.append(clip_outputs['pred_logits'])
                pred_masks.append(clip_outputs['pred_masks'])
                pred_embeds.append(clip_outputs['pred_embeds'])
                pred_queries.append(clip_outputs['pred_queries'])

            det_outputs = {
                'pred_logits': torch.cat(pred_logits, dim=0),
                'pred_masks': torch.cat(pred_masks, dim=0),
                'pred_embeds': torch.cat(pred_embeds, dim=0),
                'pred_queries': torch.cat(pred_queries, dim=0)
            }

        class_embed = self.sem_seg_head.predictor.class_embed
        outputs = self.tracker.inference(det_outputs, hybrid_embed=class_embed)

        if len(outputs['pred_logits']) == 0:
            video_output = {
                "image_size": (images.image_sizes[0], images.image_sizes[1]),
                "pred_scores": [],
                "pred_labels": [],
                "pred_masks": []
            }
            return video_output

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        input_per_image = batched_inputs[0]
        image_tensor_size = images.tensor.shape
        # image size without padding after data augmentation
        image_size = images.image_sizes[0]

        # raw image size before data augmentation
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])

        del outputs, batched_inputs, images, det_outputs

        video_output = self.inference_video(mask_cls_results, mask_pred_results, image_tensor_size, image_size,
                                            height, width, to_store)

        return video_output

    def inference_video(self, mask_cls_results, mask_pred_results, image_tensor_size, image_size, height, width,
                        to_store):
        mask_cls_result = mask_cls_results[0]
        mask_pred_result = mask_pred_results[0]

        if len(mask_cls_result) > 0:
            scores = F.softmax(mask_cls_result, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
                len(mask_cls_result), 1).flatten(0, 1)  # noqa
            scores_per_video, topk_indices = scores.flatten(
                0, 1).topk(self.num_topk, sorted=True)

            labels_per_video = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes

            mask_pred_result = mask_pred_result[topk_indices]

            num_instance, num_frame = mask_pred_result.shape[:2]

            out_scores = []
            out_labels = []
            out_masks = []

            for k in range(math.ceil(num_instance / self.test_instance_chunk_size)):
                _mask_pred_result = mask_pred_result[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]
                _scores_per_video = scores_per_video[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]
                _labels_per_video = labels_per_video[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]

                masks_list = []  # noqa
                numerator = torch.zeros(
                    _mask_pred_result.shape[0], dtype=torch.float, device=self.device)
                denominator = torch.zeros(
                    _mask_pred_result.shape[0], dtype=torch.float, device=self.device)
                for i in range(math.ceil(num_frame / self.test_interpolate_chunk_size)):
                    temp_pred_mask = _mask_pred_result[:,
                                     i * self.test_interpolate_chunk_size:(i + 1) * self.test_interpolate_chunk_size,
                                     ...]  # noqa
                    temp_pred_mask = retry_if_cuda_oom(F.interpolate)(
                        temp_pred_mask,
                        size=(image_tensor_size[-2], image_tensor_size[-1]),
                        mode="bilinear",
                        align_corners=False)
                    temp_pred_mask = temp_pred_mask[:, :,
                                                    : image_size[0], : image_size[1]]

                    temp_pred_mask = retry_if_cuda_oom(F.interpolate)(temp_pred_mask, size=(height, width),
                                                                      mode="bilinear", align_corners=False)  # noqa
                    masks = (temp_pred_mask > 0.).float()
                    numerator += (temp_pred_mask.sigmoid()
                                  * masks).flatten(1).sum(1)
                    denominator += masks.flatten(1).sum(1)

                masks_list.append(masks.bool().to(to_store))
                _scores_per_video *= (numerator / (denominator + 1e-6))
                masks = torch.cat(masks_list, dim=1)

                out_scores.extend(_scores_per_video.tolist())
                out_labels.extend(_labels_per_video.tolist())
                out_masks.extend([m for m in masks.cpu()])
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (height, width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }
        return video_output

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            if isinstance(targets_per_image.gt_masks, BitMasks):
                gt_masks = targets_per_image.gt_masks.tensor
            else:
                gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1],
                         : gt_masks.shape[2]] = gt_masks

            # filter empty instances
            gt_instance_ids = targets_per_image.gt_ids
            valid_index = gt_instance_ids != -1

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes[valid_index],
                    "masks": padded_masks[valid_index],
                    "ids": gt_instance_ids[valid_index],
                }
            )
        return new_targets
