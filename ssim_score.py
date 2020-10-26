import numpy as np
import os
import cv2
import sys
import torch

from .external import pytorch_ssim


def fun(pred_path, gt_path):
    scores = []
    scores_mask = []
    i = 0

    for item in os.listdir(pred_path):
        if item.split('.')[-1] == 'jpg':
            if i % 100 == 0:
                print(i)
            img_pred = cv2.imread(os.path.join(pred_path, item))
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
            img_pred = torch.from_numpy(np.rollaxis(img_pred, 2)).float().unsqueeze(0) / 255.0

            mask_pred = cv2.imread(os.path.join(pred_path, item.split('.')[0]+'_mask.png'))
            mask_pred = cv2.cvtColor(mask_pred, cv2.COLOR_BGR2RGB)
            mask_pred = torch.from_numpy(np.rollaxis(mask_pred, 2)).float().unsqueeze(0) / 255.0

            p = os.path.join(gt_path, item)

            img_oth = cv2.imread(p)
            img_oth = cv2.cvtColor(img_oth, cv2.COLOR_BGR2RGB)
            img_oth = torch.from_numpy(np.rollaxis(img_oth, 2)).float().unsqueeze(0) / 255.0

            ssim_loss = pytorch_ssim.SSIM(window_size=11)
            mask_ssim_loss = pytorch_ssim.SSIM(window_size=11)

            scores.append(ssim_loss(img_oth, img_pred))
            scores_mask.append(mask_ssim_loss(img_oth*mask_pred, img_pred*mask_pred))
            i += 1

    return np.mean(scores), np.mean(scores_mask)


pred_path = sys.argv[1]
print(pred_path)

gt_path = '/raid/fangzhao/dataset/Market-1501-v15.09.15/query'

result, result_mask = fun(pred_path, gt_path)

print(result)
print(result_mask)

