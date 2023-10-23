#!/usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/../..

work_dir=${SCRIPT_DIR}/work_dirs/iter_3w

python ${ROOT_DIR}/tools/train.py \
    ${SCRIPT_DIR}/configs/ddrnet_23_train.py \
    --work-dir ${work_dir}
