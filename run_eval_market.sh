#!/usr/bin/env bash
NAME=$1
EPOCH=$2
OUTPUT=$3


python -m HPBTT.eval_market

python -m HPBTT.eval_market --name ${NAME} --num_train_epoch ${EPOCH} --nohmr --img_path ${OUTPUT}

python -m HPBTT.ssim_score_market ${OUTPUT}
