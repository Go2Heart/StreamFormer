import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from detectron2.config import configurable
from detectron2.utils.registry import Registry

from ...utils import mask_nms
from .memory_bank import MemoryBank

TRACKER_REGISTRY = Registry("Tracker")
TRACKER_REGISTRY.__doc__ = """Registry for Tracker for Online Video Instance Segmentation Models."""


def build_tracker(cfg):
    """
    Build a tracker for online video instance segmentation models
    """
    name = cfg.MODEL.TRACKER.TRACKER_NAME
    return TRACKER_REGISTRY.get(name)(cfg)


@TRACKER_REGISTRY.register()
class SimpleTracker(nn.Module):
    """
    Simple Tracker for Online Video Instance Segmentation.
    Follow IDOL.
    """

    @configurable
    def __init__(self,
                 *,
                 num_classes,
                 match_metric,
                 frame_weight,
                 match_score_thr,
                 temporal_score_type,
                 match_type,
                 inference_select_thr,
                 init_score_thr,
                 mask_nms_thr,
                 num_dead_frames,
                 embed_type,
                 maximum_cache,
                 noise_frame_num,
                 noise_frame_ratio,
                 suppress_frame_num,
                 none_frame_num):
        super().__init__()
        self.num_classes = num_classes
        self.match_metric = match_metric
        self.match_score_thr = match_score_thr
        self.temporal_score_type = temporal_score_type
        self.temporal_score_type = temporal_score_type
        assert self.temporal_score_type in ['mean', 'max', 'hybrid']
        self.match_type = match_type  # greedy hungarian

        self.inference_select_thr = inference_select_thr
        self.init_score_thr = init_score_thr
        self.mask_nms_thr = mask_nms_thr
        self.frame_weight = frame_weight

        self.num_dead_frames = num_dead_frames
        self.embed_type = embed_type
        self.maximum_cache = maximum_cache

        self.noise_frame_num = noise_frame_num
        self.noise_frame_ratio = noise_frame_ratio
        self.suppress_frame_num = suppress_frame_num
        self.none_frame_num = none_frame_num

    @classmethod
    def from_config(cls, cfg):
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES  # noqa
        match_metric = cfg.MODEL.TRACKER.MATCH_METRIC
        frame_weight = cfg.MODEL.TRACKER.FRAME_WEIGHT
        match_score_thr = cfg.MODEL.TRACKER.MATCH_SCORE_THR
        temporal_score_type = cfg.MODEL.TRACKER.TEMPORAL_SCORE_TYPE
        match_type = cfg.MODEL.TRACKER.MATCH_TYPE
        inference_select_thr = cfg.MODEL.TRACKER.INFERENCE_SELECT_THR
        init_score_thr = cfg.MODEL.TRACKER.INIT_SCORE_THR
        mask_nms_thr = cfg.MODEL.TRACKER.MASK_NMS_THR

        num_dead_frames = cfg.MODEL.TRACKER.MEMORY_BANK.NUM_DEAD_FRAMES
        embed_type = cfg.MODEL.TRACKER.MEMORY_BANK.EMBED_TYPE
        maximum_cache = cfg.MODEL.TRACKER.MEMORY_BANK.maximum_cache

        noise_frame_num = cfg.MODEL.TRACKER.NOISE_FRAME_NUM
        noise_frame_ratio = cfg.MODEL.TRACKER.NOISE_FRAME_RATIO
        suppress_frame_num = cfg.MODEL.TRACKER.SUPPRESS_FRAME_NUM
        none_frame_num = cfg.MODEL.TRACKER.NONE_FRAME_NUM

        ret = {
            "num_classes": num_classes,
            "match_metric": match_metric,
            "frame_weight": frame_weight,
            "match_score_thr": match_score_thr,
            "temporal_score_type": temporal_score_type,
            "match_type": match_type,
            "inference_select_thr": inference_select_thr,
            "init_score_thr": init_score_thr,
            "mask_nms_thr": mask_nms_thr,
            # memory bank & tracklet
            "num_dead_frames": num_dead_frames,
            "embed_type": embed_type,
            "maximum_cache": maximum_cache,
            "noise_frame_num": noise_frame_num,
            "noise_frame_ratio": noise_frame_ratio,
            "suppress_frame_num": suppress_frame_num,
            "none_frame_num": none_frame_num
        }
        return ret

    @property
    def device(self):
        return torch.device('cuda')

    @property
    def empty(self):
        return self.num_tracklets == 0

    def reset(self):
        self.num_tracklets = 0  # noqa
        self.memory_bank = MemoryBank(self.embed_type,  # noqa
                                      self.num_dead_frames,
                                      self.maximum_cache)

    def update(self, ids, pred_scores, pred_logits, pred_masks, pred_reid_embeds, frame_id):
        assert ids.shape[0] == pred_logits.shape[0], 'Shape must match.'  # noqa

        num_instances = ids.shape[0]

        for instance_index in range(num_instances):

            instance_id = int(ids[instance_index].item())
            instance_score = pred_scores[instance_index]
            instance_logit = pred_logits[instance_index]
            instance_mask = pred_masks[instance_index]
            instance_reid_embed = pred_reid_embeds[instance_index]

            if instance_id in self.memory_bank.exist_ids:
                self.memory_bank.update(instance_id, instance_score, instance_logit,
                                        instance_mask, instance_reid_embed, frame_id)
            else:
                self.memory_bank.add(instance_id)
                self.memory_bank.update(instance_id, instance_score, instance_logit,
                                        instance_mask, instance_reid_embed, frame_id)

    def inference(self, det_outputs, hybrid_embed):
        num_frames, num_queries = det_outputs['pred_logits'].shape[:2]

        video_dict = dict()
        for frame_id in range(num_frames):
            if frame_id == 0:
                self.reset()

            pred_logits = det_outputs['pred_logits'][frame_id]
            pred_masks = det_outputs['pred_masks'][frame_id]
            pred_embeds = det_outputs['pred_embeds'][frame_id]
            pred_queries = det_outputs['pred_queries'][frame_id]

            scores = F.softmax(pred_logits, dim=-1)[:, :-1]
            max_scores, max_indices = torch.max(scores, dim=1)
            _, sorted_indices = torch.sort(max_scores, dim=0, descending=True)

            pred_scores = max_scores[sorted_indices]
            pred_logits = pred_logits[sorted_indices]
            pred_masks = pred_masks[sorted_indices]
            pred_embeds = pred_embeds[sorted_indices]
            pred_queries = pred_queries[sorted_indices]

            valid_indices = pred_scores > self.inference_select_thr
            if valid_indices.sum() == 0:
                valid_indices[0] = 1
            pred_scores = pred_scores[valid_indices]
            pred_logits = pred_logits[valid_indices]
            pred_masks = pred_masks[valid_indices]
            pred_embeds = pred_embeds[valid_indices]
            pred_queries = pred_queries[valid_indices]

            # NMS: can bring a slight improvement
            valid_nms_indices = mask_nms(pred_masks[:, None, ...], pred_scores, nms_thr=self.mask_nms_thr)
            pred_scores = pred_scores[valid_nms_indices]
            pred_logits = pred_logits[valid_nms_indices]
            pred_masks = pred_masks[valid_nms_indices]
            pred_embeds = pred_embeds[valid_nms_indices]
            pred_queries = pred_queries[valid_nms_indices]

            ids, pred_logits, pred_masks, pred_queries = \
                self.track(pred_scores, pred_logits, pred_masks, pred_embeds, pred_queries, frame_id)

            for index in range(ids.shape[0]):
                instance_id = ids[index]
                pred_logit = pred_logits[index]
                pred_mask = pred_masks[index]
                pred_query = pred_queries[index]

                if instance_id.item() in video_dict.keys():
                    video_dict[instance_id.item()]['masks'].append(pred_mask)
                    video_dict[instance_id.item()]['scores'].append(pred_logit)
                    video_dict[instance_id.item()]['queries'].append(pred_query)
                else:
                    video_dict[instance_id.item()] = {
                        'masks': [None for _ in range(frame_id)],
                        'scores': [None for _ in range(frame_id)],
                        'queries': [None for _ in range(frame_id)]}
                    video_dict[instance_id.item()]['masks'].append(pred_mask)
                    video_dict[instance_id.item()]['scores'].append(pred_logit)
                    video_dict[instance_id.item()]['queries'].append(pred_query)

            for k, v in video_dict.items():
                if len(v['masks']) < frame_id + 1:
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['queries'].append(None)

            # filter sequences that are too short in video_dict (noise)，
            # the rule is: if the first two frames are None and valid is less than 3
            # stolen from IDOL
            # noise_frame_num = math.floor(num_frames * self.noise_frame_ratio)
            if frame_id > self.noise_frame_num:
                del_list = []
                for k, v in video_dict.items():
                    valid = sum([1 if _ is not None else 0 for _ in v['masks']])
                    none_frame = 0
                    for m in v['masks'][::-1]:
                        if m is None:
                            none_frame += 1
                        else:
                            break
                    if none_frame >= self.none_frame_num and valid < self.suppress_frame_num:
                        del_list.append(k)
                for del_k in del_list:
                    video_dict.pop(del_k)
                    # self.memory_bank.delete_tracklet(del_k)  uncomment will drop 0.24 AP

        logits_list = []
        masks_list = []
        mask_h, mask_w = det_outputs['pred_masks'].shape[-2:]
        for inst_id, m in enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            query_list_ori = video_dict[m]['queries']
            scores_temporal = []
            queries_temporal = []
            for t, k in zip(query_list_ori, score_list_ori):
                if k is not None:
                    scores_temporal.append(k)
                    queries_temporal.append(t)
            logits_i = torch.stack(scores_temporal)
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            elif self.temporal_score_type == 'hybrid':
                raise NotImplementedError
            logits_list.append(logits_i)

            masks_list_i = []
            for n in range(num_frames):
                mask_i = video_dict[m]['masks'][n]
                if mask_i is None:
                    zero_mask = det_outputs['pred_masks'].new_zeros(mask_h, mask_w)
                    masks_list_i.append(zero_mask)
                else:
                    masks_list_i.append(mask_i)
            masks_list_i = torch.stack(masks_list_i, dim=0)
            masks_list.append(masks_list_i)
        if len(logits_list) > 0:
            pred_cls = torch.stack(logits_list, dim=0)[None, ...]
            pred_masks = torch.stack(masks_list, dim=0)[None, ...]
        else:
            pred_cls = []

        return {
            'pred_logits': pred_cls,
            'pred_masks': pred_masks
        }

    def track(self, pred_scores, pred_logits, pred_masks, pred_embeds, pred_queries, frame_id):
        ids = pred_logits.new_full((pred_logits.shape[0],), -1, dtype=torch.long)

        if self.empty:
            valid_init_indices = pred_scores > self.init_score_thr
            num_new_tracklets = valid_init_indices.sum()
            ids[valid_init_indices] = torch.arange(self.num_tracklets, self.num_tracklets + num_new_tracklets,
                                                   dtype=torch.long).to(self.device)
            self.num_tracklets += num_new_tracklets

        else:
            num_instances = pred_logits.shape[0]
            exist_tracklet_ids, exist_reid_embeds, exist_frames = self.memory_bank.exist_reid_embeds(frame_id)

            if self.match_metric == 'bisoftmax':
                # d2t: 45.6
                # t2d: 45.7
                # bio: (t2d + d2t) / 2 : 48.3  good
                similarity = torch.mm(pred_embeds, exist_reid_embeds.t())
                d2t_scores = similarity.softmax(dim=1)
                t2d_scores = similarity.softmax(dim=0)
                match_scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'cosine':
                key = F.normalize(pred_embeds, p=2, dim=1)
                query = F.normalize(exist_reid_embeds, p=2, dim=1)
                match_scores = torch.mm(key, query.t())
            else:
                raise NotImplementedError

            if self.match_type == 'greedy':
                for idx in range(num_instances):
                    if self.frame_weight:
                        valid_indices = match_scores[idx, :] > self.match_score_thr
                        if (match_scores[idx, valid_indices] > self.match_score_thr).sum() > 1:
                            wighted_scores = match_scores.clone()
                            frame_weight = exist_frames[valid_indices].to(wighted_scores)
                            wighted_scores[idx, valid_indices] = wighted_scores[idx, valid_indices] * frame_weight
                            wighted_scores[idx, ~valid_indices] = wighted_scores[
                                                                      idx, ~valid_indices] * frame_weight.mean()
                            match_score, max_indices = torch.max(wighted_scores[idx, :], dim=0)
                        else:
                            match_score, max_indices = torch.max(match_scores[idx, :], dim=0)
                    else:
                        match_score, max_indices = torch.max(match_scores[idx, :], dim=0)

                    match_tracklet_id = exist_tracklet_ids[max_indices]
                    assert match_tracklet_id > -1
                    if match_score > self.match_score_thr:
                        ids[idx] = match_tracklet_id
                        match_scores[:idx, max_indices] = 0
                        match_scores[idx + 1:, max_indices] = 0
            elif self.match_type == 'hungarian':
                # drop 3 AP
                match_cost = - match_scores.cpu()
                indices = linear_sum_assignment(match_cost)

                for i, (instance_id, _id) in enumerate(zip(*indices)):
                    if match_scores[instance_id, _id] < self.match_score_thr:
                        indices[1][i] = -1

                ids[indices[0]] = ids.new_tensor(exist_tracklet_ids[indices[1]])
                ids[indices[0][indices[1] == -1]] = -1
            else:
                raise NotImplementedError

            new_instance_indices = (ids == -1) & (pred_scores > self.init_score_thr)
            num_new_tracklets = new_instance_indices.sum().item()
            ids[new_instance_indices] = torch.arange(self.num_tracklets,
                                                     self.num_tracklets + num_new_tracklets,
                                                     dtype=torch.long).to(self.device)
            self.num_tracklets += num_new_tracklets

        valid_inds = ids > -1
        ids = ids[valid_inds]
        pred_scores = pred_scores[valid_inds]
        pred_logits = pred_logits[valid_inds]
        pred_masks = pred_masks[valid_inds]
        pred_embeds = pred_embeds[valid_inds]
        pred_queries = pred_queries[valid_inds]

        self.update(
            ids=ids,
            pred_scores=pred_scores,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            pred_reid_embeds=pred_embeds,
            frame_id=frame_id)
        self.memory_bank.clean_dead_tracklets(frame_id)

        return ids, pred_logits, pred_masks, pred_queries
