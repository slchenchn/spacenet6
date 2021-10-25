'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-25
	content: fine tuning
'''

_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/sn6_sar_pro_rotated.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

model = dict(
    pretrained='/home/csl/code/PolSAR_SelfSup/work_dirs/pbyol_r18_sn6_sar_pro_ul_ep400_lr03/20211009_142911/mmseg_epoch_400.pth',
)
