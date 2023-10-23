# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CarSegDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    # METAINFO = dict(
    #     classes=('background', 'bush', 'grassland', 'person', 'pole', 'tree'),
    #     palette=[[248, 246, 248], [221, 255, 51], [250, 50, 83], [102, 255, 102],
    #              [61, 61, 245], [51, 221, 255]])
    METAINFO = dict(
        classes=('car'),
        palette=[0, 255, 255])
        # classes=('bush', 'grassland', 'person', 'pole', 'tree'),
        # palette=[[221, 255, 51], [250, 50, 83], [102, 255, 102],
        #          [61, 61, 245], [51, 221, 255]])

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
