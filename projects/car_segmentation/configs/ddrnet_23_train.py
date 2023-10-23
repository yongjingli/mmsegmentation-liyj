_base_ = [
    './car_seg_dataset.py',
    './default_runtime.py',
]

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
# class_weight = [
#     0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
#     1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
#     1.0507
# ]

# ['background', 'bush', 'grassland', 'person', 'pole', 'tree']
# class_weight = [
#     1, 1.5, 1, 2, 2, 2
# ]
class_weight = None

crop_size = (480, 640)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,  # 输入图像小于这个尺寸时，对图像和gt图加padding，并组合成batch
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,  # 图像padding值
    seg_pad_val=255)  # gt图padding值
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DDRNet',
        in_channels=3,
        channels=64,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='work_dirs/cityscapes_model/pretrained/ddrnet23-in1kpre_3rdparty-9ca29f62.pth',
            checkpoint='../../projects/car_segmentation/checkpoints/pretrained/ddrnet23-in1kpre_3rdparty-9ca29f62.pth')
    ),
    decode_head=dict(
        type='DDRHead',
        in_channels=64 * 4,
        channels=128,
        dropout_ratio=0.,
        num_classes=2,  #19,  # 类别数量加上背景
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.4),
        ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=8, num_workers=8)

# iters = 120000
iters = 30000

# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
paramwise_cfg = dict(custom_keys={'decode_head': dict(lr_mult=5.)})  # head的学习率乘以5
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None, paramwise_cfg=paramwise_cfg)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))

randomness = dict(seed=304)
