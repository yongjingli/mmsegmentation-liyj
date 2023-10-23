#https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/4_train_test.md
#python tools/train.py  ${CONFIG_FILE} [optional arguments]

cd ../
CONFIG_FILE="./configs/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py"
python tools/train.py $CONFIG_FILE