_base_ = [
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/sn6_sar_pro_rotated_dbes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 2
model = dict(
    decode_head=dict(
        type='FCNDBES',
        input_transform='multiple_select',
        _delete_=True,
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        channels=1,
        norm_cfg=norm_cfg,
        num_classes=num_classes,
        loss_decode=dict(
            type='JointEdgeSegLoss', 
            seg_body_weight=0.1,
            num_classes=num_classes)
    )
)
find_unused_parameters = True