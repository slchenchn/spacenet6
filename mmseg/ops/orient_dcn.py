'''
Author: Shuailin Chen
Created Date: 2021-10-13
Last Modified: 2021-10-13
	content: orientation-aware deformable convolution
    NOTE: undone
'''

from mmcv.ops import DeformConv2dPack
from mmcv.cnn import CONV_LAYERS
import torch
from torch import tensor


# @CONV_LAYERS.register_module()
# class OrientDeformConv2dPack(DeformConv2dPack):
    