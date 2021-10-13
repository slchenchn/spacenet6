'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-10-13
	content: 
'''

# dataset settings
dataset_type = 'SN6SARProDataset'
data_root = '/data/csl'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
crop_size = (512, 512)
img_scale = (900, 900)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='Visualize'),
    # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale = (900, 900),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='SN6_full/SAR-PRO',
        ann_dir='SN6_sup/label_mask',
        split='SN6_sup/split/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='SN6_full/SAR-PRO',
        ann_dir='SN6_sup/label_mask',
        split='SN6_sup/split/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='SN6_full/SAR-PRO',
        ann_dir='SN6_sup/label_mask',
        split='SN6_sup/split/test.txt',
        pipeline=test_pipeline))
