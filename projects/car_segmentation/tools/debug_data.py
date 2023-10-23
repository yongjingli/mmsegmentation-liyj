import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import mmcv
import mmengine.fileio as fileio
import glob


def load_img_and_seg_mask():
    root = "/home/dell/liyongjing/dataset/seg_task_car_20230818/car"

    all_mask_idx = []
    for i in range(1, 423):
        img_path = os.path.join(root, "{}.image.png".format(str(i).zfill(3)))

        # img_path = "/home/dell/liyongjing/dataset/seg_task_car_20230818/001.png"
        mask_path = os.path.join(root, "{}.mask.0.png".format(str(i).zfill(3)))
        print(img_path)
        print(mask_path)
        # mask_path2 = os.path.join(root, "{}.mask.1.png".format(str(i).zfill(3)))

        # img_path = "/home/dell/liyongjing/dataset/seg_task_car_20230818/car/001.image.png"
        # mask_path = "/home/dell/liyongjing/dataset/seg_task_car_20230818/car/001.mask.0.png"

        if not os.path.exists(img_path):
            print("{} not exit".format(img_path))

        if not os.path.exists(mask_path):
            print("{} not exit".format(mask_path))

        # 采用matplotlib可以读取,但是采用opencv读取不了
        # img = cv2.imread(img_path)
        img = mpimg.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        print(img.shape)

        all_mask_idx = all_mask_idx + np.unique(mask).tolist()

        plt.imshow(img)
        # plt.imshow(mask)
        plt.show()
        exit(1)

    print(set(all_mask_idx))


def test_cityscape_format():
    color_img_path = "/home/dell/liyongjing/dataset/mmseg/grassland_overfit_20230721/gtFine/test/default/1689848873551644470_gtFine_color.png"
    label_img_path = "/home/dell/liyongjing/dataset/mmseg/grassland_overfit_20230721/gtFine/test/default/1689848873551644470_gtFine_labelIds.png"
    instance_img_path = "/home/dell/liyongjing/dataset/mmseg/grassland_overfit_20230721/gtFine/test/default/1689848873551644470_gtFine_instanceIds.png"


def debug_label():
    # label_path = "/home/dell/liyongjing/dataset/mmseg/grassland_overfit_20230721/gtFine/train/default/1689848678804013195_gtFine_labelIds.png"
    # img = Image.open(label_path)
    # img = np.array(img)
    # print(np.unique(img))

    # label_path = "/home/dell/liyongjing/dataset/seg_task_car_20230818/car_cityscape/gtFine/test/default/379_gtFine_labelIds.png"
    root = "/home/dell/liyongjing/dataset/seg_task_car_20230818/car_cityscape/gtFine/test/default"
    img_names = glob.glob(root + "/*_gtFine_labelIds.png")
    for img_name in img_names:
        label_path = img_name
        img_bytes = fileio.get(
            label_path, backend_args=None)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend="pillow").squeeze().astype(np.uint8)

        print(np.unique(gt_semantic_seg))


if __name__ == "__main__":
    print("Start")
    # load_img_and_seg_mask()
    debug_label()
    print("End")

