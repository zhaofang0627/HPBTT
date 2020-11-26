#!/usr/bin/env bash
NAME=$1
EPOCH=$2
INPUT=$3


python -m HPBTT.demo_market --name ${NAME} --num_train_epoch ${EPOCH} --img_path ${INPUT}

python -m HPBTT.demo_market --name ${NAME} --num_train_epoch ${EPOCH} --img_path ${INPUT} --nohmr
