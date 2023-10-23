#!/usr/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/../..

work_dir=${SCRIPT_DIR}/work_dirs/iter_3w
checkpoint=${work_dir}/iter_3000.pth
show_dir=${work_dir}/visual_3w

# val
python ${ROOT_DIR}/tools/test.py \
    ${SCRIPT_DIR}/configs/ddrnet_23_train.py \
    ${checkpoint} \
    --work-dir ${work_dir} \
    --show-dir ${show_dir}_val \
    --task seg

# convert to video
${SCRIPT_DIR}/tools/merge_img_to_video.sh ${show_dir}_val/vis_data/vis_image 1

# test
python ${ROOT_DIR}/tools/test.py \
    ${SCRIPT_DIR}/configs/ddrnet_23_test.py \
    ${checkpoint} \
    --work-dir ${work_dir} \
    --show-dir ${show_dir}_test \
    --cfg-options test_evaluator.output_dir=${show_dir}_test/format_results \
    --task seg

# convert to vidio
${SCRIPT_DIR}/tools/merge_img_to_video.sh ${show_dir}_test/vis_data/vis_image 10
