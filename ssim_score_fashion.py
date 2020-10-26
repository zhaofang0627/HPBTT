import numpy as np
import os
import cv2
import pickle
import sys
import torch

from .external import pytorch_ssim


def fun(pred_path, gt_path, query_list):
    query_num = len(query_list)
    scores = []

    for i in range(query_num):
        if i % 100 == 0:
            print(i)

        img_rel_path = query_list[i]
        l = img_rel_path.split('/')
        rel_path = os.path.join(l[1], l[2], l[3], l[4])

        if os.path.isfile(os.path.join(pred_path, rel_path.split('.')[0]+'.png')):
            img_pred = cv2.imread(os.path.join(pred_path, rel_path.split('.')[0]+'.png'))
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
            img_pred = torch.from_numpy(np.rollaxis(img_pred, 2)).float().unsqueeze(0) / 255.0

            p = os.path.join(gt_path, rel_path.split('.')[0]+'.jpg')

            img_oth = cv2.imread(p)
            img_oth = cv2.cvtColor(img_oth, cv2.COLOR_BGR2RGB)
            img_oth = torch.from_numpy(np.rollaxis(img_oth, 2)).float().unsqueeze(0) / 255.0

            ssim_loss = pytorch_ssim.SSIM(window_size=11)

            scores.append(ssim_loss(img_oth, img_pred))

    print(len(scores))
    return np.mean(scores)


pred_path = sys.argv[1]

gt_path = '/raid/fangzhao/dataset/DeepFashion/In-shop_Clothes_Retrieval_Benchmark/img'

img_data = pickle.load(open(os.path.join('/raid/fangzhao/cmr_py3/cachedir/deepfashion/data', 'train_test_list.pkl'), 'rb'))
query_list = img_data['query_list']

result = fun(pred_path, gt_path, query_list)

print(result)

