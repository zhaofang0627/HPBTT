"""
Loss Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def texture_dt_loss(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 1 x F x T*T
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid, align_corners=True)

    if vis_rend is not None:
        # Visualize the error!
        # B x 3 x F x T*T
        dts = dist_transf.repeat(1, 3, 1, 1)
        # B x 3 x F x T x T
        dts = dts.view(-1, 3, F, T, T)
        # B x F x T x T x 3
        dts = dts.permute(0, 2, 3, 4, 1)
        dts = dts.unsqueeze(4).repeat(1, 1, 1, 1, T, 1) / dts.max()

        from ..utils import bird_vis
        for i in range(dist_transf.size(0)):
            rend_dt = vis_rend(verts[i], cams[i], dts[i])
            rend_img = bird_vis.tensor2im(tex_pred[i].data)            
            import matplotlib.pyplot as plt
            plt.ion()
            fig=plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(121)
            ax.imshow(rend_dt)
            ax = fig.add_subplot(122)
            ax.imshow(rend_img)
            import ipdb; ipdb.set_trace()

    return dist_transf.mean()


def tv_loss(tex):
    tv = torch.mean(torch.abs(tex[:, :, :, :-1] - tex[:, :, :, 1:])) + \
               torch.mean(torch.abs(tex[:, :, :-1, :] - tex[:, :, 1:, :]))
    return tv


class PerceptualTextureLoss(object):
    def __init__(self):
        from ..nnutils.perceptual_loss import PerceptualLoss
        self.perceptual_loss = PerceptualLoss()

    def __call__(self, img_pred, img_gt):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        """
        mask_sq = torch.zeros_like(img_gt)
        mask_sq[:, :, :, 64:192] = 1.0

        dist = self.perceptual_loss(img_pred * mask_sq, img_gt * mask_sq)
        return dist.mean()


class PerceptualTextureFashionLoss(object):
    def __init__(self):
        from ..nnutils.perceptual_loss import PerceptualLoss
        self.perceptual_loss = PerceptualLoss()

    def __call__(self, img_pred, img_gt, mask_gt):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
          mask_gt: B x H x W
        """
        mask_gt = mask_gt.unsqueeze(1)

        dist = self.perceptual_loss(img_pred, img_gt * mask_gt)
        return dist.mean()
