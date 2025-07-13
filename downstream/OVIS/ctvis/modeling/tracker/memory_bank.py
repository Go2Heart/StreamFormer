import torch
import torch.nn.functional as F


class Tracklet(object):
    """
    用于定义一个track 片段， 表示一个 小的跟踪片段（轨迹），是组成一个memory bank的基本单元
    """

    def __init__(self, instance_id, maximum_cache=10, linear=None):
        self.instance_id = instance_id
        self.logits = []
        self.masks = []
        self.reid_embeds = []
        self.long_scores = []
        self.frame_ids = []
        self.last_reid_embed = torch.zeros((256,), device='cuda')
        self.similarity_guided_reid_embed = torch.zeros((256,), device='cuda')
        self.fusion_reid_embed = torch.zeros((256,), device='cuda')
        self.exist_frames = 0
        self.maximum_cache = maximum_cache
        self.momentum = 0.75  # 这个效果是最好的
        self.linear = linear  # todo: 使用线性层代替 动量更新

    def update(self, score, logit, mask, reid_embed, frame_id):
        # 用于更新 track时使用
        self.long_scores.append(score)
        self.logits.append(logit)
        self.masks.append(mask)
        self.reid_embeds.append(reid_embed)
        self.frame_ids.append(frame_id)

        if self.exist_frames == 0:
            # 第一次出现的话直接使用使用
            # 加不加差不多 (+ 0.02)
            self.last_reid_embed = reid_embed
            self.similarity_guided_reid_embed = reid_embed
            self.fusion_reid_embed = reid_embed
        else:
            self.last_reid_embed = (1 - self.momentum) * self.last_reid_embed + self.momentum * reid_embed

            # Similarity-Guided Feature Fusion
            # https://arxiv.org/abs/2203.14208v1
            all_reid_embed = torch.stack(self.reid_embeds[:-1], dim=0)
            similarity = torch.sum(torch.einsum("bc,c->b",
                                                F.normalize(all_reid_embed, dim=-1),
                                                F.normalize(reid_embed, dim=-1))) / (len(self.reid_embeds) - 1)
            beta = max(0, similarity)
            self.similarity_guided_reid_embed = (1 - beta) * self.similarity_guided_reid_embed + beta * reid_embed
            # 用 可学习的优化来进行更新
            if self.linear:
                self.fusion_reid_embed = self.linear(self.fusion_reid_embed + reid_embed)

        self.exist_frames += 1

        if len(self.long_scores) > self.maximum_cache:
            self.long_scores.pop(0)
            self.reid_embeds.pop(0)


class MemoryBank:
    """
    众所周知，就是一个memory bank，主要用来存取 tracklet，与 CL 中的 Memory Bank 有一定的区别
    """

    def __init__(self,
                 embed_type='temporally_weighted_softmax',
                 num_dead_frames=10,
                 maximum_cache=10):
        self.tracklets = dict()
        self.num_tracklets = 0

        self.embed_type = embed_type  # last | temporally_weighted_softmax | momentum | similarity_guided
        self.num_dead_frames = num_dead_frames
        self.maximum_cache = maximum_cache

    def add(self, instance_id):
        self.tracklets[instance_id] = Tracklet(instance_id, self.maximum_cache)
        self.num_tracklets += 1
        # self.num_tracklets += 0  # bug fixed

    def update(self, instance_id, score, logit, mask, reid_embed, frame_id):
        self[instance_id].update(score, logit, mask, reid_embed, frame_id)

    def __getitem__(self, instance_id):
        return self.tracklets[instance_id]

    def __len__(self):
        return self.num_tracklets

    @property
    def exist_ids(self):
        return self.tracklets.keys()

    def clean_dead_tracklets(self, cur_frame_id):
        # 删除已经死亡的 tracklets
        dead_ids = []
        for instance_id, tracklet in self.tracklets.items():
            if cur_frame_id - tracklet.frame_ids[-1] > self.num_dead_frames:
                dead_ids.append(instance_id)
        for dead_id in dead_ids:
            del self.tracklets[dead_id]
            self.num_tracklets -= 1

    def exist_reid_embeds(self, frame_id):
        memory_bank_embeds = []
        memory_bank_ids = []
        memory_bank_exist_frames = []
        for instance_id, tracklet in self.tracklets.items():
            if self.embed_type == 'temporally_weighted_softmax':
                score_weights = torch.stack(tracklet.long_scores)
                length = score_weights.shape[0]
                temporal_weights = torch.range(0., 1, 1 / length)[1:].to(score_weights)
                weights = score_weights + temporal_weights
                weighted_sum_embed = (torch.stack(tracklet.reid_embeds) * weights.unsqueeze(1)).sum(0) / weights.sum()
                memory_bank_embeds.append(weighted_sum_embed)
            elif self.embed_type == 'last':
                memory_bank_embeds.append(tracklet.reid_embeds[-1])
            elif self.embed_type == 'momentum':  # 动量更新
                memory_bank_embeds.append(tracklet.last_reid_embed)
            elif self.embed_type == 'similarity_guided':
                memory_bank_embeds.append(tracklet.similarity_guided_reid_embed)
            else:
                raise NotImplementedError

            memory_bank_ids.append(instance_id)
            # num_frame_disappear = float(frame_id - tracklet.frame_ids[-1])
            # memory_bank_exist_frames.append(max(tracklet.exist_frames / num_frame_disappear, 0.8))
            memory_bank_exist_frames.append(tracklet.exist_frames)

        memory_bank_embeds = torch.stack(memory_bank_embeds, dim=0)
        memory_bank_ids = memory_bank_embeds.new_tensor(memory_bank_ids).to(dtype=torch.long)
        # memory_bank_exist_frames = memory_bank_embeds.new_tensor(memory_bank_exist_frames).to(dtype=torch.float32)
        memory_bank_exist_frames = memory_bank_embeds.new_tensor(memory_bank_exist_frames).to(dtype=torch.long)

        return memory_bank_ids, memory_bank_embeds, memory_bank_exist_frames
