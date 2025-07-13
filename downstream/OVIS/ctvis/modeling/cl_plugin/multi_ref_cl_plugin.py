import random

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import BitMasks

from ...utils import box_xyxy_to_cxcywh

from .ct_cl_plugin import CL_PLUGIN_REGISTRY


@CL_PLUGIN_REGISTRY.register()
class MultiRefCLPlugin(nn.Module):
    @configurable
    def __init__(self,
                 *,
                 weight_dict,
                 num_negatives,
                 sampling_frame_num,
                 one_direction):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_negatives = num_negatives

        self.ref_first = False
        self.sampling_frame_num = sampling_frame_num

        self.one_direction = one_direction

    @classmethod
    def from_config(cls, cfg):
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        one_direction = cfg.MODEL.CL_PLUGIN.ONE_DIRECTION
        reid_weight = cfg.MODEL.CL_PLUGIN.REID_WEIGHT
        aux_reid_weight = cfg.MODEL.CL_PLUGIN.AUX_REID_WEIGHT

        weight_dict = {"loss_reid": reid_weight, "loss_aux_reid": aux_reid_weight}
        num_negatives = cfg.MODEL.CL_PLUGIN.NUM_NEGATIVES

        ret = {"weight_dict": weight_dict,
               "num_negatives": num_negatives,
               "sampling_frame_num": sampling_frame_num,
               "one_direction": one_direction}
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
            if key == 'aux_outputs':
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
            indices = matcher(outputs_without_aux, targets)  # [matched_row, matched_colum]
            indices_list.append(indices)

        contrastive_items = []

        batch_size = len(targets_list[0])
        for i in range(batch_size):
            indice_list = [indices[i] for indices in indices_list]
            target_list = [targets[i] for targets in targets_list]

            gt2query_id_list = [indice[0][torch.sort(indice[1])[1]] for indice in indice_list]
            reid_embedding_list = [outputs['pred_embeds'][i] for outputs in outputs_list]

            for j in range(self.sampling_frame_num):
                anchor_embeddings = reid_embedding_list[j]
                anchor_target = target_list[j]

                ref_ids = list(range(self.sampling_frame_num))
                ref_ids.pop(j)

                num_instances = anchor_target['valid'].shape[0]  # key frame 

                for instance_i in range(num_instances):
                    if anchor_target['valid'][instance_i]:
                        anchor_embedding = anchor_embeddings[gt2query_id_list[j][instance_i]][None, ...]
                        positive_list = []
                        negative_list = []
                        for ref_id in ref_ids:
                            ref_target = target_list[ref_id]
                            if ref_target['valid'][instance_i]:
                                query_id = gt2query_id_list[ref_id][instance_i]
                                positive_list.append(reid_embedding_list[ref_id][query_id][None, ...])
                                negative_query_id = sorted(
                                    random.sample(set(range(self.num_negatives + 1)) - set([query_id.item()]),
                                                  self.num_negatives))  # noqa
                                negative_list.append(reid_embedding_list[ref_id][negative_query_id])
                            else:
                                negative_list.append(reid_embedding_list[ref_id])

                        if len(positive_list) == 0:
                            continue

                        num_positive = len(positive_list)
                        pos_neg_embedding = torch.cat(positive_list + negative_list, dim=0)

                        pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],), dtype=torch.int64)
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
                if self.one_direction:
                    break
        losses = loss_reid(contrastive_items, outputs_list[0])

        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
            else:
                losses.pop(k)
        return losses

    def prepare_targets(self, targets):
        # prepare for track part
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
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
            ids_list.append(list(range(i, bz * self.sampling_frame_num, self.sampling_frame_num)))

        targets_list = []
        for i in range(self.sampling_frame_num):
            targets_list.append([new_targets[j] for j in ids_list[i]])

        return targets_list


def loss_reid(qd_items, outputs):
    contras_loss = 0
    aux_loss = 0
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

    losses = {'loss_reid': contras_loss.sum() / len(qd_items),
              'loss_aux_reid': aux_loss / len(qd_items)}
    return losses
