'''
Author: Shuailin Chen
Created Date: 2021-12-06
Last Modified: 2021-12-24
	content: 
'''
_base_ = [
    './upernet_vit-b16_mln_512x512_40k_sn6_sar_pro_rotated.py'
]

model = dict(
    pretrained=None
)
