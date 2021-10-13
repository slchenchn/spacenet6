'''
Author: Shuailin Chen
Created Date: 2021-10-13
Last Modified: 2021-10-13
	content: transforms for SpaceNet6
'''

import mmcv
from ..builder import PIPELINES


@PIPELINES.register_module()
class FlipAccodingToOrien():
    ''' Vertical flip accoding to orientation of images in SpaceNet6 datasets'''

    def __init__(self) -> None:
        pass

    def __call__(self, results):
        if results['orientation']:
            results['img'] = mmcv.imflip(results['img'], direction='vertical')
            results['gt_semantic_seg'] = mmcv.imflip(results['gt_semantic_seg'], direction='gt_semantic_seg')
        return results