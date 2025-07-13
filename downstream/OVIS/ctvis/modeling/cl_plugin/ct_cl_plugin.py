import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.config import configurable
from detectron2.structures import BitMasks
from detectron2.utils.registry import Registry

from ...utils import box_xyxy_to_cxcywh

CL_PLUGIN_REGISTRY = Registry("CL_PLUGIN")
CL_PLUGIN_REGISTRY.__doc__ = """Registry for CL PLUGIN for Discriminative Representation Learning."""


def build_cl_plugin(cfg):
    name = cfg.MODEL.CL_PLUGIN.CL_PLUGIN_NAME
    return CL_PLUGIN_REGISTRY.get(name)(cfg)


class TrainTracklet(object):
    def __init__(self, instance_id, maximum_cache=10, momentum_embed=True, noise_embed=False):
        self.instance_id = instance_id
        self.logits = []
        self.masks = []
        self.reid_embeds = []
        self.negative_embeds = []
        self.long_scores = []
        self.frame_ids = []
        self.last_reid_embed = torch.zeros((256,), device='cuda')
        self.similarity_guided_reid_embed = None
        self.similarity_guided_reid_embed_list = []
        self.positive_embed_list = []
        self.exist_frames = 0
        self.maximum_cache = maximum_cache
        self.momentum = 0.75
        self.momentum_embed = momentum_embed
        self.noise_embed = noise_embed

    def update(self, positive_embed, negative_embed, frame_id=0):
        # update with noise
        if self.noise_embed and positive_embed is None:
            if np.random.rand() < 0.999:
                positive_embed = None
            else:
                index = random.randint(0, negative_embed.shape[0] - 1)
                positive_embed = negative_embed[index][None, ...]
        else:
            pass

        # if self.noise_embed and noise_embed is not None:
        #     positive_embed = noise_embed

        self.reid_embeds.append(positive_embed)
        self.negative_embeds.append(negative_embed)

        if positive_embed is not None:
            self.positive_embed_list.append(positive_embed)
            if self.exist_frames == 0:
                self.similarity_guided_reid_embed = positive_embed
                self.similarity_guided_reid_embed_list.append(self.similarity_guided_reid_embed)
            else:
                # Similarity-Guided Feature Fusion
                # https://arxiv.org/abs/2203.14208v1
                all_reid_embed = []
                for embedding in self.reid_embeds[:-1]:
                    if embedding is not None:
                        all_reid_embed.append(embedding)
                all_reid_embed = torch.cat(all_reid_embed, dim=0)

                similarity = torch.sum(torch.einsum("bc,c->b",
                                                    F.normalize(all_reid_embed, dim=-1),
                                                    F.normalize(positive_embed.squeeze(),
                                                                dim=-1))) / self.exist_frames  # noqa
                beta = max(0, similarity)
                self.similarity_guided_reid_embed = (1 - beta) * self.similarity_guided_reid_embed + beta * positive_embed  # noqa
                self.similarity_guided_reid_embed_list.append(
                    self.similarity_guided_reid_embed)
            self.exist_frames += 1
        else:
            # no instance in the current frame
            self.similarity_guided_reid_embed_list.append(self.similarity_guided_reid_embed)

    def exist_before(self, frame_id):
        return frame_id != sum([1 if _ is None else 0 for _ in self.reid_embeds[:frame_id]])

    def exist_after(self, frame_id):
        return frame_id != sum([1 if _ is None else 0 for _ in self.reid_embeds[frame_id + 1:]])

    def get_positive_negative_embeddings(self, frame_id):
        anchor_embedding = self.reid_embeds[frame_id]
        positive_embedding = None
        if self.exist_before(frame_id):
            if self.momentum_embed and np.random.rand() > 0.5:
                positive_embedding = self.similarity_guided_reid_embed_list[frame_id - 1]
            else:
                for embedding in self.reid_embeds[:frame_id][::-1]:
                    if embedding is not None:
                        positive_embedding = embedding
                        break
        else:
            if self.exist_after(frame_id):
                for embedding in self.reid_embeds[frame_id + 1:]:
                    if embedding is not None:
                        positive_embedding = embedding
                        break
        negative_embedding = self.negative_embeds[frame_id - 1]

        return anchor_embedding, positive_embedding, negative_embedding


class SimpleTrainMemory:
    def __init__(self,
                 embed_type='temporally_weighted_softmax',
                 num_dead_frames=10,
                 maximum_cache=10,
                 momentum_embed=True,
                 noise_embed=False):
        self.tracklets = dict()
        self.num_tracklets = 0

        # last | temporally_weighted_softmax | momentum | similarity_guided
        self.embed_type = embed_type
        self.num_dead_frames = num_dead_frames
        self.maximum_cache = maximum_cache
        self.momentum_embed = momentum_embed
        self.noise_embed = noise_embed

    def update(self, instance_id, reid_embed, negative_embed):
        if instance_id not in self.exist_ids:
            self.tracklets[instance_id] = TrainTracklet(
                instance_id, self.maximum_cache, momentum_embed=self.momentum_embed, noise_embed=self.noise_embed)
            self.num_tracklets += 1
        self[instance_id].update(reid_embed, negative_embed)

    def __getitem__(self, instance_id):
        return self.tracklets[instance_id]

    def __len__(self):
        return self.num_tracklets

    def empty(self):
        self.tracklets = dict()
        self.num_tracklets = 0

    @property
    def exist_ids(self):
        return self.tracklets.keys()

    def valid(self, instance_id, frame_id):
        return self[instance_id].reid_embeds[frame_id] is not None

    def exist_reid_embeds(self):
        memory_bank_embeds = []
        memory_bank_ids = []

        for instance_id, tracklet in self.tracklets.items():
            memory_bank_embeds.append(tracklet.similarity_guided_reid_embed)

            memory_bank_ids.append(instance_id)

        memory_bank_embeds = torch.stack(memory_bank_embeds, dim=0)
        memory_bank_ids = memory_bank_embeds.new_tensor(memory_bank_ids).to(dtype=torch.long)

        return memory_bank_ids, memory_bank_embeds

    def get_training_samples(self, instance_id, frame_id):
        anchor_embedding, positive_embedding, negative_embedding = self[instance_id].get_positive_negative_embeddings(frame_id)  # noqa

        return anchor_embedding, positive_embedding, negative_embedding


@CL_PLUGIN_REGISTRY.register()
class CTCLPlugin(nn.Module):
    @configurable
    def __init__(self,
                 *,
                 weight_dict,
                 num_negatives,
                 sampling_frame_num,
                 bio_cl,
                 momentum_embed,
                 noise_embed):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_negatives = num_negatives
        self.sampling_frame_num = sampling_frame_num

        self.bio_cl = bio_cl
        self.momentum_embed = momentum_embed
        self.noise_embed = noise_embed
        self.train_memory_bank = SimpleTrainMemory(momentum_embed=self.momentum_embed, noise_embed=self.noise_embed)

    @classmethod
    def from_config(cls, cfg):
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        reid_weight = cfg.MODEL.CL_PLUGIN.REID_WEIGHT
        aux_reid_weight = cfg.MODEL.CL_PLUGIN.AUX_REID_WEIGHT

        weight_dict = {"loss_reid": reid_weight,
                       "loss_aux_reid": aux_reid_weight}

        num_negatives = cfg.MODEL.CL_PLUGIN.NUM_NEGATIVES

        bio_cl = cfg.MODEL.CL_PLUGIN.BIO_CL

        momentum_embed = cfg.MODEL.CL_PLUGIN.MOMENTUM_EMBED
        noise_embed = cfg.MODEL.CL_PLUGIN.NOISE_EMBED

        ret = {"weight_dict": weight_dict,
               "num_negatives": num_negatives,
               "sampling_frame_num": sampling_frame_num,
               "bio_cl": bio_cl,
               "momentum_embed": momentum_embed,
               "noise_embed": noise_embed}
        return ret

    @property
    def device(self):
        return torch.device('cuda')

    def get_key_ref_outputs(self, det_outputs):
        outputs_keys = det_outputs.keys()  # noqa
        outputs_list = [dict() for _ in range(self.sampling_frame_num)]

        num_images = det_outputs['pred_logits'].shape[0]
        index_list = []
        for i in range(self.sampling_frame_num):
            index_list.append(torch.arange(i, num_images, step=self.sampling_frame_num, device=self.device))

        for key in outputs_keys:
            if key in ['aux_outputs', 'interm_outputs']:
                pass
            else:
                for i in range(self.sampling_frame_num):
                    outputs_list[i][key] = det_outputs[key][index_list[i]]

        return outputs_list

    def train_loss(self, det_outputs, gt_instances, matcher):
        targets_list = self.prepare_targets(gt_instances)
        outputs_list = self.get_key_ref_outputs(det_outputs)

        indices_list = []
        for i in range(self.sampling_frame_num):
            outputs = outputs_list[i]
            targets = targets_list[i]
            outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
            # [matched_row, matched_colum]
            indices = matcher(outputs_without_aux, targets)
            indices_list.append(indices)

        losses = dict()

        losses.update(self.get_reid_loss(targets_list, outputs_list, indices_list))

        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
            else:
                losses.pop(k)
        return losses

    def get_reid_loss(self, targets_list, outputs_list, indices_list):
        contrastive_items = []

        batch_size = len(targets_list[0])
        for i in range(batch_size):  # per batch
            # empty memory bank
            self.train_memory_bank.empty()
            indice_list = [indices[i] for indices in indices_list]
            target_list = [targets[i] for targets in targets_list]

            gt2query_id_list = [indice[0][torch.sort(
                indice[1])[1]] for indice in indice_list]

            reid_embedding_list = [outputs[f'pred_embeds'][i]
                                    for outputs in outputs_list]
            num_instances = target_list[0]['valid'].shape[0]  # num of instances in this frame
            
            # Step 1: Store and update the embeddings into memory bank first
            for j in range(self.sampling_frame_num):
                anchor_embeddings = reid_embedding_list[j]
                anchor_target = target_list[j]

                for instance_i in range(num_instances):
                    if anchor_target['valid'][instance_i]:  # instance exists
                        anchor_query_id = gt2query_id_list[j][instance_i]  # query id 
                        anchor_embedding = anchor_embeddings[anchor_query_id][None, ...]

                        negative_query_id = sorted(
                            random.sample(set(range(self.num_negatives + 1)) - set([anchor_query_id.item()]), self.num_negatives))  # noqa
                        negative_embedding = anchor_embeddings[negative_query_id]
                    else:  # not exists
                        anchor_embedding = None
                        negative_embedding = anchor_embeddings

                    self.train_memory_bank.update(
                        instance_i, anchor_embedding, negative_embedding)  # update the memory bank

            # Step 2: build contrastive items frame by frame 
            for frame_id in range(self.sampling_frame_num):
                if frame_id == 0:
                    continue
                else:
                    # query -> memory_bank
                    for instance_i in range(num_instances):
                        if self.train_memory_bank.valid(instance_i, frame_id):
                            anchor_embedding, positive_embedding, negative_embedding = self.train_memory_bank.get_training_samples(
                                instance_i, frame_id)  # noqa
                            if positive_embedding is None:
                                # No valid positive embedding
                                continue
                            num_positive = positive_embedding.shape[0]

                            pos_neg_embedding = torch.cat([positive_embedding, negative_embedding], dim=0)

                            pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                        dtype=torch.int64)  # noqa
                            pos_neg_label[:num_positive] = 1.

                            # dot product
                            dot_product = torch.einsum('ac,kc->ak', [pos_neg_embedding, anchor_embedding])
                            aux_normalize_pos_neg_embedding = nn.functional.normalize(pos_neg_embedding, dim=1)
                            aux_normalize_anchor_embedding = nn.functional.normalize(anchor_embedding, dim=1)

                            aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                               aux_normalize_anchor_embedding])
                            contrastive_items.append({
                                'dot_product': dot_product,
                                'cosine_similarity': aux_cosine_similarity,
                                'label': pos_neg_label})

                    # if self.bio_cl:
                    #     # memory_bank -> query
                    #     for instance_i in range(num_instances):
                    #         if self.train_memory_bank[instance_i].similarity_guided_reid_embed_list[frame_id - 1] is not None and \
                    #                 self.train_memory_bank[instance_i].reid_embeds[frame_id] is not None:
                    #             anchor_embedding = self.train_memory_bank[
                    #                 instance_i].similarity_guided_reid_embed_list[frame_id - 1]
                    #             positive_embedding = self.train_memory_bank[instance_i].reid_embeds[frame_id]
                    #             negative_embedding = self.train_memory_bank[
                    #                 instance_i].negative_embeds[frame_id]

                    #             num_positive = positive_embedding.shape[0]

                    #             pos_neg_embedding = torch.cat(
                    #                 [positive_embedding, negative_embedding], dim=0)

                    #             pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                    #                                                         dtype=torch.int64)  # noqa
                    #             pos_neg_label[:num_positive] = 1.

                    #             # dot product
                    #             dot_product = torch.einsum(
                    #                 'ac,kc->ak', [pos_neg_embedding, anchor_embedding])
                    #             aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    #                 pos_neg_embedding, dim=1)
                    #             aux_normalize_anchor_embedding = nn.functional.normalize(
                    #                 anchor_embedding, dim=1)

                    #             aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                    #                                                                aux_normalize_anchor_embedding])
                    #             contrastive_items.append({
                    #                 'dot_product': dot_product,
                    #                 'cosine_similarity': aux_cosine_similarity,
                    #                 'label': pos_neg_label})

        # we follow the losses in IDOL
        losses = loss_reid(contrastive_items, outputs_list[0])

        return losses

    def prepare_targets(self, targets):
        # prepare for track part
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            if isinstance(targets_per_image.gt_masks, BitMasks):
                gt_masks = targets_per_image.gt_masks.tensor
            else:
                gt_masks = targets_per_image.gt_masks
            inst_ids = targets_per_image.gt_ids
            valid_id = inst_ids != -1  # if an object is disappeared，its gt_ids is -1

            new_targets.append(
                {"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, 'inst_id': inst_ids, 'valid': valid_id})
        bz = len(new_targets) // self.sampling_frame_num
        ids_list = []
        for i in range(self.sampling_frame_num):
            ids_list.append(
                list(range(i, bz * self.sampling_frame_num, self.sampling_frame_num)))

        targets_list = []
        for i in range(self.sampling_frame_num):
            targets_list.append([new_targets[j] for j in ids_list[i]])

        return targets_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def loss_reid(qd_items, outputs, reduce=False):
    contras_loss = 0
    aux_loss = 0

    num_qd_items = len(qd_items)
    if reduce:  # it seems worse when reduce is True
        num_qd_items = torch.as_tensor(
            [num_qd_items], dtype=torch.float, device=outputs['pred_embeds'].device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_qd_items)
        num_qd_items = torch.clamp(
            num_qd_items / get_world_size(), min=1).item()

    if len(qd_items) == 0:
        losses = {'loss_reid': outputs['pred_embeds'].sum() * 0,
                    'loss_aux_reid': outputs['pred_embeds'].sum() * 0}
        return losses

    for qd_item in qd_items:
        pred = qd_item['dot_product'].permute(1, 0)
        label = qd_item['label'].unsqueeze(0)
        # contrastive loss
        pos_inds = (label == 1)
        neg_inds = (label == 0)
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, pred.shape[1])
        # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
        x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0)
        contras_loss += torch.logsumexp(x, dim=1)

        aux_pred = qd_item['cosine_similarity'].permute(1, 0)
        aux_label = qd_item['label'].unsqueeze(0)

        aux_loss += (torch.abs(aux_pred - aux_label) ** 2).mean()

    losses = {'loss_reid': contras_loss.sum() / num_qd_items,
                'loss_aux_reid': aux_loss / num_qd_items}
    return losses
