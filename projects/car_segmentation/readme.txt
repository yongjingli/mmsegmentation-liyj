1. 安装环境
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

cd mmsegmentation-car
pip install -v -e .


2. 车辆分割的项目在.mmsegmentation-car/projects/car_segmentation

2.1 将数据生成训练、验证和测试的脚本为
./mmsegmentation-car/projects/car_segmentation/tools/convert_data_2_cityscape.py

2.2 训练的脚本为
cd ./mmsegmentation-car/projects/car_segmentation
sh train.sh

#训练的结果保存在./mmsegmentation-car/projects/car_segmentation/work_dirs/iter_3w
# 训练的loss和日志都在这里
# 采用tensorboard可视化训练的过程的监控变量
# tensorboard --logdir=./mmsegmentation-car/projects/car_segmentation/work_dirs/iter_3w/20230818_200119/vis_data


2.3 测试的脚本为
sh test.sh
#会打印测试的结果，同时会可视化gt和预测的结果，左边为gt右边为预测结果
./mmsegmentation-car/projects/car_segmentation/work_dirs/iter_3w/visual_3w_val

