'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-12
	content: enable deformable convolutions
'''

_base_ = [
    './deeplabv3_r50-d8_512x512_20k_sn6_sar_pro.py'
]

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

