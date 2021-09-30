'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-09-30
	content: 
'''
_base_ = [
    './deeplabv3_r50-d8_512x512_20k_sn6_ps_rgb.py'
]

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
