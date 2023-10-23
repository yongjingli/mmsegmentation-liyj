_base_ = [
    './ddrnet_23_train.py',
]

dataset_type = 'CarSegDataset'
# data_root = '/data/ros_bag'
# data_root = '../../data/seg_dataset/grassland_overfit_20230721'
data_root = '/home/dell/liyongjing/dataset/seg_task_car_20230818/car_cityscape'
reduce_zero_label = False

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations'),  # 没有gt时注释掉
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=reduce_zero_label,
        # img_suffix='.jpg',
        img_suffix='.png',
        data_prefix=dict(
            # img_path='grassland_2023_07_20-18_24_24_0_image', seg_map_path='',
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'
        ),
        pipeline=test_pipeline))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], format_only=True, output_dir='work_dirs/format_results')  # format_only不做评测，没有gt时设置为True
