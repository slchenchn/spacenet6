'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-23
	content: 
'''
_base_ = [
    './deeplabv3_r50-d8_512x512_20k_sn6_sar_pro_rotated.py'
]

model = dict(
    pretrained=None,
    )
