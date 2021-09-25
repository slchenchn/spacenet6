'''
Author: Shuailin Chen
Created Date: 2021-09-16
Last Modified: 2021-09-16
	content: 
'''

import os
import os.path as osp
import cv2
import mylib.image_utils as iu
import mylib.labelme_utils as lu
import numpy as np
import sys

from ..builder import PIPELINES


@PIPELINES.register_module()
class Visualize():
    ''' View the images in the pipeline '''

    def __init__(self, save_dir=r'./tmp') -> None:
        self.save_dir = save_dir

    def __call__(self, results):
        img = results['img']
        gt = results['gt_semantic_seg']
        filename = results['filename']

        iu.save_image_by_cv2(img, osp.join(self.save_dir, 
                            f'img.png'), is_bgr=False)
        lu.lblsave(osp.join(self.save_dir, f'gt.png'), gt, np.array([[0, 0, 0], [255, 255, 255]]))
        # iu.save_image_by_cv2(gt, osp.join(self.save_dir, f'gt.png'),
                            # is_bgr=False, if_norm=True)
        print(f"file name: {filename}\ngt: {osp.join(results['seg_prefix'],    results['ann_info']['seg_map'])}")
        # sys.exit()
        print()
