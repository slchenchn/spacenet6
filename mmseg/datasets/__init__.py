'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-09-25
	content: 
'''
# Copyright (c) OpenMMLab. All rights reserved.
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .stare import STAREDataset
from .voc import PascalVOCDataset

from .sn6_extent_pauli import SN6PauliDataset
from .sn6_sar_pro import SN6SARProDataset
from .sn6_ps_rgb import SN6PSRGBDataset

# __all__ = [
#     'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
#     'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
#     'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
#     'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
#     'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset'
# ]
