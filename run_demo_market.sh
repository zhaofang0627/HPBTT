#!/usr/bin/env bash
NAME=$1
EPOCH=$2
INPUT=$3


CUDA_VISIBLE_DEVICES=0 python -m cmr_py3.demo_market --name ${NAME} --num_train_epoch ${EPOCH} \
  --img_path ${INPUT} --img_size 256 --notest

CUDA_VISIBLE_DEVICES=0 python -m cmr_py3.demo_market --name ${NAME} --num_train_epoch ${EPOCH} \
  --img_path ${INPUT} --img_size 256 --notest --nohmr
