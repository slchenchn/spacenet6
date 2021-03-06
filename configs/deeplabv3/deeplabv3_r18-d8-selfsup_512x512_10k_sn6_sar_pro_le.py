'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-03
	content: linear evaluation
'''

_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/sn6_sar_pro.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10k_le.py'
]

model = dict(
    pretrained='/home/csl/code/PolSAR_SelfSup/work_dirs/pbyol_r18_sn6_sar_pro_ul_ep200_lr00375/20210930_225502/epoch_200_mmseg.pth',
    
    backbone=dict(
        depth=18,
        frozen_stages=4,    # freeze the whole backbone
        ),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=2)
)
