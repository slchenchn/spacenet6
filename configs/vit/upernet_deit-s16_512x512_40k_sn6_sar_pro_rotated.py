'''
Author: Shuailin Chen
Created Date: 2021-12-07
Last Modified: 2021-12-07
	content: 
'''
_base_ = './upernet_vit-b16_mln_512x512_40k_sn6_sar_pro_rotated.py'

model = dict(
    pretrained='pretrain/deit_small_patch16_224-cd65a155.pth',
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    decode_head=dict(num_classes=2, in_channels=[384, 384, 384, 384]),
    neck=None,
    auxiliary_head=dict(num_classes=2, in_channels=384))
