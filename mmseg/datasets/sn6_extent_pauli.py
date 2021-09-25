'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-09-14
	content: 
'''
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image
import re
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SN6PauliDataset(CustomDataset):
    """SpaceNet6 pauliRGB dataset.
    """
    CLASSES = (
        'background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                seg_map_prefix = r'geoms_',
                img_prefix = r'slc_',
                **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        
        self.seg_map_prefix = seg_map_prefix
        self.img_prefix = img_prefix
        

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name)
                    if ann_dir is not None:
                        crs = re.findall(r'\d{6}\_\d{6}', img_name)
                        assert len(crs)==2, \
                                f'len of found cfs should be 2, but got {len(crs)}'
                        seg_map = self.seg_map_prefix + crs[1] \
                                    + self.seg_map_suffix
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            raise NotImplemented

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos