cd ..

# (1)ddrnet
# official
#CONFIG_FILE="./configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py"
#CHECKPOINT_FILE="./checkpoints/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth"

CONFIG_FILE="./work_dirs/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py"
CHECKPOINT_FILE="./work_dirs/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024/iter_120000.pth"

python demo/image_demo.py demo/demo.png $CONFIG_FILE $CHECKPOINT_FILE --device cuda:0 --out-file result.jpg
