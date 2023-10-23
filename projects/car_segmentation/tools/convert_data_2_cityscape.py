#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import glob
import json
import os
import os.path as osp
import numpy as np
import shutil
import glob
import cv2
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm
import mmcv
import mmengine.fileio as fileio


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def deal_json(json_file):
    data_cs = {}
    objects = []
    num = -1
    num = num + 1
    if not json_file.endswith('.json'):
        print('Cannot generating dataset from:', json_file)
        return None
    with open(json_file) as f:
        print('Generating dataset from:', json_file)
        data = json.load(f)
        data_cs['imgHeight'] = data['imageHeight']
        data_cs['imgWidth'] = data['imageWidth']
        for shapes in data['shapes']:
            obj = {}
            label = shapes['label']
            obj['label'] = label
            points = shapes['points']
            p_type = shapes['shape_type']
            if p_type == 'polygon':
                obj['polygon'] = points
            objects.append(obj)
        data_cs['objects'] = objects
    return data_cs


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument('--json_input_dir', default='label', help='input annotated directory')
    parser.add_argument(
        '--output_dir', default='cityscape', help='output dataset directory', )

    args = parser.parse_args()
    try:
        assert os.path.exists(args.json_input_dir)
    except AssertionError as e:
        print('The json folder does not exist!')
        os._exit(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Deal with the json files.
    total_num = len(glob.glob(osp.join(args.json_input_dir, '*.json')))
    for json_name in os.listdir(args.json_input_dir):
        data_cs = deal_json(osp.join(args.json_input_dir, json_name))
        if data_cs is None:
            continue
        json.dump(
            data_cs,
            open(osp.join(args.output_dir, json_name), 'w'),
            indent=4,
            cls=MyEncoder, )


def debug():
    import os
    import logging
    import cv2
    import xml
    import xml.dom.minidom
    import numpy as np
    import glob
    import shutil
    from PIL import Image
    import random
    import json

    logging.basicConfig(level=logging.INFO, )
    logger = logging.getLogger(__name__)

    def create_img(height, width, bgr_color):
        image = np.zeros((height, width, 3), np.uint8)
        # color = tuple(reversed(rgb_color))
        image[:] = bgr_color
        return image

    def write_label_list(output_dir):
        # write label list
        images_sets_dir = os.path.join(output_dir, "ImageSets")
        segmentation_dir = os.path.join(images_sets_dir, "Segmentation")
        for dir_path in [images_sets_dir, segmentation_dir]:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

        train_label_txt = os.path.join(segmentation_dir, "train.txt")
        val_label_txt = os.path.join(segmentation_dir, "val.txt")
        test_label_txt = os.path.join(segmentation_dir, "test.txt")
        trainval_label_txt = os.path.join(segmentation_dir, "trainval.txt")

        img_names = list(filter(lambda x: x[-3:] == "jpg", os.listdir(os.path.join(output_dir, "JPEGImages"))))
        train_ratio = 0.9
        train_num = int(len(img_names) * train_ratio)
        with open(train_label_txt, "w") as f:
            for img_name in img_names[:train_num]:
                f.write(img_name[:-4] + "\n")

        with open(val_label_txt, "w") as f:
            for img_name in img_names[train_num:]:
                f.write(img_name[:-4] + "\n")

        with open(test_label_txt, "w") as f:
            for img_name in img_names[train_num:]:
                f.write(img_name[:-4] + "\n")

        with open(trainval_label_txt, "w") as f:
            for img_name in img_names:
                f.write(img_name[:-4] + "\n")

        print(len(img_names))
        print(len(img_names[:train_num]))
        print(len(img_names[train_num:]))

    def generate_voc_seg_from_labelme():
        logger.info("Start proc")
        input_dir_all = ["/home/liyongjing/Egolee_2021/data/src_prison/door_seg_labelme"]
        output_dir = "/home/liyongjing/Egolee_2021/data/TrainData/seg_prison/VOCdevkit/prison_wall"

        bg_color = (255, 0, 0)
        # cls_name = ['wall', 'fence']
        # cls_color = [(0, 0, 255), (0, 0, 255)]
        # cls_type = ['area', 'area']

        cls_name = ['h_line']
        cls_color = [(0, 0, 255)]
        cls_type = ['line']

        line_thickness = 20
        gt_img_width = 384
        gt_img_height = 640

        img_dir_target = os.path.join(output_dir, "JPEGImages")
        seg_dir_target = os.path.join(output_dir, "SegmentationClass")
        for dir_path in [img_dir_target, seg_dir_target]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.mkdir(dir_path)

        # start process.
        for input_dir in input_dir_all:
            img_names = list(filter(lambda x: x[-3:] == "jpg", os.listdir(input_dir)))
            for i, img_name in enumerate(img_names):
                logger.info("Process Img:{}".format(img_name))

                img_path = os.path.join(input_dir, img_name)
                img = cv2.imread(img_path)

                # anno_path = img_path.replace(".jpg", ".json")
                anno_path = os.path.splitext(img_path)[0] + '.json'

                if os.path.exists(anno_path):
                    f = open(anno_path)
                    coor = json.load(f)
                else:
                    coor = {"shapes": []}

                image_gt = create_img(img.shape[0], img.shape[1], bg_color)
                if coor['shapes']:
                    for shape in coor['shapes']:
                        if shape['label'] in cls_name:
                            cls_idx = cls_name.index(shape['label'])
                            draw_color = cls_color[cls_idx]
                            draw_type = cls_type[cls_idx]
                            points = np.array(shape['points'], dtype=np.int32)
                            if draw_type == 'line':
                                cv2.polylines(image_gt, [points], isClosed=False, \
                                              color=tuple(reversed(draw_color)), thickness=line_thickness)
                            if draw_type == 'area':
                                # print(points)
                                points = np.array([shape['points']], dtype=np.int32)
                                cv2.fillPoly(image_gt, points, tuple(reversed(draw_color)))

                # resize
                img = cv2.resize(img, (gt_img_width, gt_img_height), interpolation=cv2.INTER_CUBIC)
                image_gt = cv2.resize(image_gt, (gt_img_width, gt_img_height), interpolation=cv2.INTER_NEAREST)

                # save img
                img_name_new = ''
                for s in os.path.splitext(img_name)[0]:
                    if s in ['[', ']', '.', '=', '&', ',']:
                        img_name_new = img_name_new + '_'
                    else:
                        img_name_new = img_name_new + s
                img_name_new = img_name_new + '.jpg'

                img_path_target = os.path.join(img_dir_target, img_name_new)
                cv2.imwrite(img_path_target, img)

                # save gt img
                seg_img_path_target = os.path.join(seg_dir_target, os.path.splitext(img_name_new)[0] + '.png')
                seg_img_target = np.zeros((image_gt.shape[0], image_gt.shape[1]))

                cls_color_set = list(set(cls_color))
                for ii, color in enumerate(cls_color_set):
                    seg_img_target[np.all(image_gt == color[::-1], axis=2)] = ii + 1  # cls start from 1

                pil_img = Image.fromarray(seg_img_target.astype(np.uint8), mode="P")

                pil_img.putpalette([0, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0])
                pil_img.save(seg_img_path_target)

        # write label list
        write_label_list(output_dir)


def convert_data_2_cityscape():
    root = "/home/dell/liyongjing/dataset/seg_task_car_20230818/car"
    target_root = "/home/dell/liyongjing/dataset/seg_task_car_20230818/car_cityscape"
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    os.mkdir(target_root)

    all_num = 421

    train_val_test_name = ["train", "val", "test"]
    train_val_test_ratio = [0.8, 0.1, 0.1]
    train_val_test_num =[int(all_num * ratio) for ratio in train_val_test_ratio]
    train_val_test_cumsum = np.cumsum(np.array(train_val_test_num))

    for _sub_dir_0 in ["gtFine", "leftImg8bit"]:
        sub_dir_0 = os.path.join(target_root, _sub_dir_0)
        if not os.path.exists(sub_dir_0):
            os.mkdir(sub_dir_0)

        for _sub_dir_1 in train_val_test_name:
            sub_dir_1 = os.path.join(sub_dir_0, _sub_dir_1)
            if not os.path.exists(sub_dir_1):
                os.mkdir(sub_dir_1)

            default_dir = os.path.join(sub_dir_1, "default")
            if not os.path.exists(default_dir):
                os.mkdir(default_dir)

    for i in tqdm(range(1, all_num)):
        sub_set_index = np.sum(train_val_test_cumsum < i)
        sub_set_name = train_val_test_name[sub_set_index]
        img_path = os.path.join(root, "{}.image.png".format(str(i).zfill(3)))

        mask_path_format = os.path.join(root, "{}.mask.*.png".format(str(i).zfill(3)))
        mask_paths = glob.glob(mask_path_format)

        # img = cv2.imread(img_path)
        img = mpimg.imread(img_path)

        # 将读取图像转为取值为0-255的RGB图像
        img = img * 255
        img = img.astype(np.uint8)

        s_name = str(i).zfill(3)

        # save img
        img_path = os.path.join(target_root, "leftImg8bit", sub_set_name, "default",
                                s_name + "_leftImg8bit.png")
        cv2.imwrite(img_path, img[:, :, ::-1])

        img_h, img_w, _ = img.shape

        img_color = np.zeros((img_h, img_w, 3), np.uint8)
        img_label = np.zeros((img_h, img_w), np.uint8)
        img_instance = np.zeros((img_h, img_w), np.uint8)

        for i, mask_path in enumerate(mask_paths):
            mask = cv2.imread(mask_path, 0)

            mask_h, mask_w = mask.shape
            assert img_h == mask_h, "img_h == mask_h"
            assert img_w == mask_w, "img_w == mask_w"
            img_label[mask > 0] = 1             # 语义分割, 这里只有1类
            # img_label[mask > 0] = mask        # 语义分割, 如果有多个类别
            img_instance[mask > 0] = i
            img_color[mask > 0] = (255, 255, 0)

        # save gt img
        img_color_path = os.path.join(target_root, "gtFine", sub_set_name, "default",
                                      s_name + "_gtFine_color.png")
        img_instance_path = os.path.join(target_root, "gtFine", sub_set_name, "default",
                                      s_name + "_gtFine_instanceIds.png")
        img_label_path = os.path.join(target_root, "gtFine", sub_set_name, "default",
                                         s_name + "_gtFine_labelIds.png")

        cv2.imwrite(img_color_path, img_color)
        cv2.imencode('.png', img_label)[1].tofile(img_label_path)
        cv2.imencode('.png', img_instance)[1].tofile(img_instance_path)

        # check label
        img = Image.open(img_label_path)
        img = np.array(img)
        print(np.unique(img))

        img_bytes = fileio.get(
            img_label_path, backend_args=None)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend="pillow").squeeze().astype(np.uint8)

        print(np.unique(gt_semantic_seg))


        # plt.imshow(img)
        # plt.show()
        #
        # exit(1)


if __name__ == '__main__':
    # main()
    convert_data_2_cityscape()
