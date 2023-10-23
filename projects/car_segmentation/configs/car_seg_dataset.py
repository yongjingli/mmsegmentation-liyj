# custom_imports = dict(imports=['projects.xw_segmentation.datasets.xw_segmentation'])
custom_imports = dict(imports=['projects.car_segmentation.libs'])

# dataset settings
dataset_type = 'CarSegDataset'
# data_root = 'data/cityscapes/'
# data_root = './data/dataset/grassland_overfit_20230721'
# data_root = '../../data/seg_dataset/grassland_overfit_20230721'
data_root = '/home/dell/liyongjing/dataset/seg_task_car_20230818/car_cityscape'
# crop_size = (1024, 1024)
crop_size = (480, 640)   # h, w
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        # scale=(2048, 1024),
        scale=(640, 480),  # 被32整除的数值   w, h
        ratio_range=(0.5, 2.0),  # 在0.5到2的范围内随机缩放
        keep_ratio=False),  # 由于原图是2880x1860，保持比例的话不能被32整除
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),  # 随机裁剪，图像小于crop_size时则不裁剪，cat_max_ratio是某一类别的标注区域占整图的最大比例
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations'),  # 没有gt时注释掉
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

reduce_zero_label = False
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=reduce_zero_label,
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=reduce_zero_label,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=val_pipeline))

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         reduce_zero_label=reduce_zero_label,
#         data_prefix=dict(
#             img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
#         pipeline=test_pipeline))

test_dataloader = val_dataloader

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root='/data/ros_bag',
#         reduce_zero_label=reduce_zero_label,
#         img_suffix='.jpg',
#         data_prefix=dict(
#             img_path='grassland_2023_07_20-18_24_24_0_image'),
#         pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
# test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], format_only=True, output_dir='work_dirs/format_results')  # format_only不做评测，没有gt时设置为True