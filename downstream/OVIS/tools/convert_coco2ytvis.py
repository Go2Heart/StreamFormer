
import sys
import os

sys.path.insert(0, './')

import mmcv

from ..ctvis.data.vis.builtin import (
    COCO_TO_YTVIS_2019,
    COCO_TO_YTVIS_2021,
    COCO_TO_OVIS
)

_root = os.getenv("DETECTRON2_DATASETS", "./datasets")

convert_list = {
    "coco2ytvis2019": {
        "annos": os.path.join(_root, "coco/annotations/instances_train2017.json"),
        "label_map": COCO_TO_YTVIS_2019,
        "out_path": os.path.join(_root, "coco/annotations/coco2ytvis2019_train.json")
    },
    "coco2ytvis2021": {
        "annos": os.path.join(_root, "coco/annotations/instances_train2017.json"),
        "label_map": COCO_TO_YTVIS_2021,
        "out_path": os.path.join(_root, "coco/annotations/coco2ytvis2021_train.json")
    },
    "coco2ovis": {
        "annos": os.path.join(_root, "coco/annotations/instances_train2017.json"),
        "label_map": COCO_TO_OVIS,
        "out_path": os.path.join(_root, "coco/annotations/coco2ovis_train.json")
    },
}


def merge(coco_list):
    merge_coco_annos = {
        'images': [], 'categories': coco_list[0]['categories'], 'annotations': []}
    image_id = 0
    anno_id = 0
    for i, anns in enumerate(coco_list):
        image_id_dict = dict()
        for image in anns['images']:
            image_id_ = image['id']
            if image_id_ in image_id_dict.keys():
                raise NotImplementedError
            image_id_dict[image_id_] = image_id
            image['id'] = image_id
            image_id += 1
        for anno in anns['annotations']:
            anno['image_id'] = image_id_dict[anno['image_id']]
            anno['id'] = anno_id
            anno_id += 1

        merge_coco_annos['images'].extend(anns['images'])
        merge_coco_annos['annotations'].extend(anns['annotations'])
    return merge_coco_annos


def process(anno_path, label_map):
    coco_annos = mmcv.load(anno_path)

    out_json = {}
    for k, v in coco_annos.items():
        if k != 'annotations':
            out_json[k] = v

    converted_item_num = 0
    exist_image_key = []
    out_json['annotations'] = []
    for anno in coco_annos['annotations']:
        if anno["category_id"] not in label_map:
            continue

        out_json['annotations'].append(anno)
        exist_image_key.append(anno['image_id'])
        converted_item_num += 1
    exist_image_key = list(set(exist_image_key))

    # filter unused images
    new_images_list = []
    print('Start deleting images.')
    for image in out_json['images']:
        if image['id'] in exist_image_key:
            new_images_list.append(image)
    out_json['images'] = new_images_list

    return out_json


def main():
    for key, value_dict in convert_list.items():
        label_map = value_dict["label_map"]
        anno_path = value_dict["annos"]
        out_path = value_dict["out_path"]
        anno = process(anno_path, label_map)

        mmcv.dump(anno, out_path)
        print(f"{key} is finished!")


if __name__ == '__main__':
    main()
