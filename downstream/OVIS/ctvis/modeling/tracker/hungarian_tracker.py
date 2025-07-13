import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable

from online_vis.utils import mask_nms
from .memory_bank import MemoryBank
from .simple_tracker import TRACKER_REGISTRY


@TRACKER_REGISTRY.register()
class HungarianTracker(nn.Module):
    """
    Hungarian Tracker for Online Video Instance Segmentation.

    """
    @configurable
    def __init__(self,
                 *,
                 num_classes,
                 match_metric,
                 frame_weight,
                 match_score_thr,
                 temporal_score_type,
                 inference_select_thr,
                 init_score_thr,
                 mask_nms_thr,
                 num_dead_frames,
                 embed_type,
                 maximum_cache):
        super().__init__()
        self.num_classes = num_classes
        self.match_metric = match_metric
        self.match_score_thr = match_score_thr
        self.temporal_score_type = temporal_score_type  # 之前设置了 max，IDOL里面设置了 mean
        self.temporal_score_type = temporal_score_type
        assert self.temporal_score_type in ['mean', 'max', 'hybrid']

        self.inference_select_thr = inference_select_thr  # 对应 IDOL 里面第一个阈值
        self.init_score_thr = init_score_thr  # 设置为 0.1 点数最高,可能未来还需要再跳调一调
        self.mask_nms_thr = mask_nms_thr
        self.frame_weight = frame_weight

        # Memory Bank 的参数
        self.num_dead_frames = num_dead_frames
        self.embed_type = embed_type
        self.maximum_cache = maximum_cache

    @classmethod
    def from_config(cls, cfg):
        # head 部分的参数
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES  # noqa
        match_metric = cfg.MODEL.TRACKER.MATCH_METRIC
        frame_weight = cfg.MODEL.TRACKER.FRAME_WEIGHT
        match_score_thr = cfg.MODEL.TRACKER.MATCH_SCORE_THR
        temporal_score_type = cfg.MODEL.TRACKER.TEMPORAL_SCORE_TYPE
        inference_select_thr = cfg.MODEL.TRACKER.INFERENCE_SELECT_THR
        init_score_thr = cfg.MODEL.TRACKER.INIT_SCORE_THR
        mask_nms_thr = cfg.MODEL.TRACKER.MASK_NMS_THR

        # memory bank 以及 tracklet 部分的参数 only for test stage
        num_dead_frames = cfg.MODEL.TRACKER.MEMORY_BANK.NUM_DEAD_FRAMES
        embed_type = cfg.MODEL.TRACKER.MEMORY_BANK.EMBED_TYPE
        maximum_cache = cfg.MODEL.TRACKER.MEMORY_BANK.maximum_cache

        ret = {
            "num_classes": num_classes,
            "match_metric": match_metric,
            "frame_weight": frame_weight,
            "match_score_thr": match_score_thr,
            "temporal_score_type": temporal_score_type,
            "inference_select_thr": inference_select_thr,
            "init_score_thr": init_score_thr,
            "mask_nms_thr": mask_nms_thr,
            # memory bank & tracklet
            "num_dead_frames": num_dead_frames,
            "embed_type": embed_type,
            "maximum_cache": maximum_cache
        }
        return ret

    @property
    def device(self):
        return torch.device('cuda')

    @property
    def empty(self):
        return self.num_tracklets == 0

    def reset(self):
        """用于初始化 tracker """
        self.num_tracklets = 0  # noqa
        self.memory_bank = MemoryBank(self.embed_type,    # noqa
                                      self.num_dead_frames,
                                      self.maximum_cache)

    def update(self, ids, pred_scores, pred_logits, pred_masks, pred_reid_embeds, frame_id):
        assert ids.shape[0] == pred_logits.shape[0], 'Shape must match.'

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
        # 每帧单独进行
        for frame_id in range(num_frames):
            if frame_id == 0:
                self.reset()

            pred_logits = det_outputs['pred_logits'][frame_id]
            pred_masks = det_outputs['pred_masks'][frame_id]
            pred_embeds = det_outputs['pred_embeds'][frame_id]
            pred_queries = det_outputs['pred_queries'][frame_id]
            # pred_

            # 第一步 先过滤低置信度的 目标instance
            scores = F.softmax(pred_logits, dim=-1)[:, :-1]
            max_scores, max_indices = torch.max(scores, dim=1)
            _, sorted_indices = torch.sort(max_scores, dim=0, descending=True)
            # 根据 置信度 高低进行排序
            pred_scores = max_scores[sorted_indices]
            pred_logits = pred_logits[sorted_indices]
            pred_masks = pred_masks[sorted_indices]
            pred_embeds = pred_embeds[sorted_indices]
            pred_queries = pred_queries[sorted_indices]

            valid_indices = pred_scores > self.inference_select_thr  # 0.01 效果比较好，0.1 和 0.001(-0.1) 效果都会变差
            if valid_indices.sum() == 0:  # 如果没有符合的instance的话 这里面如果啥都不拿的话会掉点(-0.4)
                valid_indices[0] = 1  # 这里直接拿置信度最高的 instance
            pred_scores = pred_scores[valid_indices]
            pred_logits = pred_logits[valid_indices]
            pred_masks = pred_masks[valid_indices]
            pred_embeds = pred_embeds[valid_indices]
            pred_queries = pred_queries[valid_indices]

            # 第二步 用NMS去重
            valid_nms_indices = mask_nms(pred_masks[:, None, ...], pred_scores, nms_thr=self.mask_nms_thr)  # 这里设置0.6
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
                    #  如果 存在 的 话
                    video_dict[instance_id.item()]['masks'].append(pred_mask)
                    video_dict[instance_id.item()]['scores'].append(pred_logit)
                    video_dict[instance_id.item()]['queries'].append(pred_query)
                else:
                    # 不存在 的 话
                    video_dict[instance_id.item()] = {
                        'masks': [None for _ in range(frame_id)],
                        'scores': [None for _ in range(frame_id)],
                        'queries': [None for _ in range(frame_id)]}
                    video_dict[instance_id.item()]['masks'].append(pred_mask)
                    video_dict[instance_id.item()]['scores'].append(pred_logit)
                    video_dict[instance_id.item()]['queries'].append(pred_query)

            # 有一些tracklet 在当前帧没有匹配的 话，那么 在下一帧就会有 匹配
            for k, v in video_dict.items():
                if len(v['masks']) < frame_id + 1:
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['queries'].append(None)

            # filter sequences that are too short in video_dict (noise)，
            # the rule is: if the first two frames are None and valid is less than 3
            # stolen from IDOL 这个部分的代码需要仔细调一调 这个会删除掉噪音视频，但是会不会删除相应的 tracklet
            if frame_id > 8:  # 一套 规则 用于 删除 那些 noise mask sequence
                del_list = []
                for k, v in video_dict.items():
                    valid = sum([1 if _ is not None else 0 for _ in v['masks']])
                    if v['masks'][-1] is None and v['masks'][-2] is None and valid < 3:
                        del_list.append(k)
                for del_k in del_list:
                    video_dict.pop(del_k)
                    # self.memory_bank.delete_tracklet(del_k)  加了会掉点（-0.24）

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
            queries_i = torch.stack(queries_temporal, dim=0)
            # todo: 这两种方式都有一点不对
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            elif self.temporal_score_type == 'hybrid':
                hybrid_query = torch.mean(queries_i, dim=0)
                logits_i = hybrid_embed(hybrid_query).squeeze()
            else:
                raise NotImplementedError
            logits_list.append(logits_i)

            # category_id = np.argmax(logits_i.mean(0))
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
        ids = pred_logits.new_full((pred_logits.shape[0],), -1, dtype=torch.long)  # 先将所有的 id 设置为 -1,方便区分

        if self.empty:
            # 如果是空的话，直接将其添加到 memory_bank 里面
            valid_init_indices = pred_scores > self.init_score_thr
            num_new_tracklets = valid_init_indices.sum()
            ids[valid_init_indices] = torch.arange(self.num_tracklets, self.num_tracklets + num_new_tracklets,
                                                   dtype=torch.long).to(self.device)
            self.num_tracklets += num_new_tracklets  # 只有这里的 num_tracklets会用到，memory bank 里面的 num_tracklets 不会用到

        else:
            # memory 如果非空的话 就进行 匹配
            num_instances = pred_logits.shape[0]
            exist_tracklet_ids, exist_reid_embeds, exist_frames = self.memory_bank.exist_reid_embeds

            if self.match_metric == 'bisoftmax':
                # d2t: 45.6
                # t2d: 45.7
                # (t2d + d2t) / 2 : 48.3
                similarity = torch.mm(pred_embeds, exist_reid_embeds.t())
                d2t_scores = similarity.softmax(dim=1)  # 这个方法有一点不好的 就是 会有一个大于0.9
                t2d_scores = similarity.softmax(dim=0)
                match_scores = (d2t_scores + t2d_scores) / 2  # 这个 score 双向 的
            elif self.match_metric == 'cosine':
                key = F.normalize(pred_embeds, p=2, dim=1)
                query = F.normalize(exist_reid_embeds, p=2, dim=1)
                match_scores = torch.mm(key, query.t())
            else:
                raise NotImplementedError

            # 对每一个 预测 的 instance 进行 匹配
            # 这里这个地方用的是贪心算法
            # 考虑做成一个匈牙利匹配的过程

            for idx in range(num_instances):
                if self.frame_weight:
                    # 这个模块有效（+1.6）
                    # 求取 相似度 时 会考虑 每一个 tracklet存在 的 时间
                    valid_indices = match_scores[idx, :] > self.match_score_thr
                    if (match_scores[idx, valid_indices] > self.match_score_thr).sum() > 1:  # 合适的 tracklet 如果说大于一个
                        wighted_scores = match_scores.clone()
                        frame_weight = exist_frames[valid_indices].to(wighted_scores)
                        # 看谁 出现的 次数多，出现 较多的 更适合 作为 匹配成功的 对象
                        wighted_scores[idx, valid_indices] = wighted_scores[idx, valid_indices] * frame_weight
                        wighted_scores[idx, ~valid_indices] = wighted_scores[idx, ~valid_indices] * frame_weight.mean()
                        match_score, max_indices = torch.max(wighted_scores[idx, :], dim=0)
                    else:
                        match_score, max_indices = torch.max(match_scores[idx, :], dim=0)  # 少于 一个 的话 直接 选择 最合适 的 那个 即可
                else:
                    match_score, max_indices = torch.max(match_scores[idx, :], dim=0)

                match_tracklet_id = exist_tracklet_ids[max_indices]
                assert match_tracklet_id > -1
                if match_score > self.match_score_thr:
                    ids[idx] = match_tracklet_id
                    match_scores[:idx, max_indices] = 0  # 这么做可以让那个 tracklet 不
                    match_scores[idx + 1:, max_indices] = 0  # 再去 匹配 其他 的 instances

            # 可能会有一部分 的 instance 没有匹配上 任何 的 instance
            new_instance_indices = (ids == -1) & (pred_scores > self.init_score_thr)  # 有一些 id 并没有 匹配 上
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
