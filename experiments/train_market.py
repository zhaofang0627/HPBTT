"""
Training for Market-1501

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

import os.path as osp
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from collections import OrderedDict
import pickle

from ..data import market1501 as market_data
from ..data import background_pose as background_data
from ..utils import image as image_utils
from ..utils.SMPL_D import SMPL
from ..nnutils import train_utils
from ..nnutils import loss_utils
from ..nnutils import mesh_net
from ..nnutils.nmr import NeuralRendererWOCAM
from ..nnutils import geom_utils
from .get_body_mesh import get_body_mesh

flags.DEFINE_string('dataset', 'market1501')
# Weights:
flags.DEFINE_float('tex_loss_wt', 1., 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', .5, 'weights to tex dt loss')
flags.DEFINE_float('tex_tv_loss_wt', .5, 'weights to tex total variation loss')

opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        # ----------
        # Options
        # ----------
        self.face_labels = pickle.load(open('/raid/fangzhao/HPBTT/cachedir/market1501/data/face_labels.pkl', 'rb'), encoding='latin1')
        region_colors = np.array([[0, 0, 0], [1.0, 0, 0], [1.0, 0, 1.0], [0, 1.0, 0], [1.0, 1.0, 0], [0, 0, 1.0], [0, 1.0, 1.0]])
        # F(13776) x 3
        face_colors = torch.tensor(region_colors[(self.face_labels + 1).astype(np.int)]).type(torch.FloatTensor).cuda()
        # B x F(13776) x T x T x T x 3
        self.textures_seg = face_colors.unsqueeze(0).repeat(opts.batch_size, 1, 1).unsqueeze(2).repeat(1, 1, opts.tex_size**2, 1).\
            view(opts.batch_size, 13776, opts.tex_size, opts.tex_size, 3).unsqueeze(4).repeat(1, 1, 1, 1, opts.tex_size, 1)

        faces = np.load('/raid/fangzhao/HPBTT/external/hmr/src/tf_smpl/smpl_faces.npy')
        self.m = get_body_mesh('/raid/fangzhao/HPBTT/external/hmr/models/body.obj', trans=np.array([0, 0, 4]),
                               rotation=np.array([np.pi / 2, 0, 0]))

        img_size_scale = (opts.img_size, opts.img_size // 2)
        self.model_scale = mesh_net.MeshScaleNet(img_size_scale, opts, verts=self.m.v, faces=faces, nz_feat=opts.nz_feat, num_classes=751, num_shape_param=1)

        img_size_flow = (opts.img_size, opts.img_size)
        self.model_flow = mesh_net.MeshMP3FlowPoseNet(img_size_flow, opts, verts=self.m.v, faces=faces, nz_feat=opts.nz_feat, n_upconv=6, nc_init=128, predict_flow=True)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model_scale, 'scale', opts.num_pretrain_epochs)
            self.load_network(self.model_flow, 'flow', opts.num_pretrain_epochs)

        self.model_scale = self.model_scale.cuda()
        self.model_flow = self.model_flow.cuda()

        faces = self.model_flow.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)

        self.num_cam = 3
        self.num_pose = 72
        self.smpl = SMPL('/raid/fangzhao/HPBTT/external/hmr/models/neutral_smpl_with_cocoplus_reg.pkl', opts, obj_saveable=False).cuda()

        # For renderering.
        print(opts.batch_size)
        self.renderer = NeuralRendererWOCAM(opts.img_size, device=0)

        self.renderer_seg = NeuralRendererWOCAM(opts.img_size, device=0)
        self.renderer_seg.ambient_light_only()

        # Need separate NMR for each fwd/bwd call.
        self.tex_renderer = NeuralRendererWOCAM(opts.img_size, device=0)
        # Only use ambient light for tex renderer
        self.tex_renderer.ambient_light_only()

        self.tex_ex_renderer = NeuralRendererWOCAM(opts.img_size, device=0)
        # Only use ambient light for tex renderer
        self.tex_ex_renderer.ambient_light_only()

        return


    def init_dataset(self):
        opts = self.opts
        if opts.dataset == 'market1501':
            self.data_module = market_data
            self.background_data_module = background_data
        else:
            print('Unknown dataset %d!' % opts.dataset)

        self.dataloader = self.data_module.data_id_loader(opts)
        self.bg_dataloader = self.background_data_module.data_loader(opts)

        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])


    def define_criterion(self):
        self.texture_loss = loss_utils.PerceptualTextureLoss()
        self.texture_ex_loss = loss_utils.PerceptualTextureLoss()
        self.texture_dt_loss_fn = loss_utils.texture_dt_loss
        self.tv_loss = loss_utils.tv_loss


    def set_input(self, batch, bg_image_batch):
        input_img_tensor = batch['img'][:, :, :, 64:192].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.normalize(input_img_tensor[b])

        img_tensor = batch['img'].type(torch.FloatTensor)
        bg_img_tensor = bg_image_batch['bg_img'].type(torch.FloatTensor)
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        theta_tensor = batch['theta'].type(torch.FloatTensor)

        self.input_imgs = Variable(
            input_img_tensor.cuda(), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(), requires_grad=False)
        self.bg_imgs = Variable(
            bg_img_tensor.cuda(), requires_grad=False)
        self.masks = Variable(
            mask_tensor.cuda(), requires_grad=False)
        self.theta = Variable(
            theta_tensor.cuda(), requires_grad=False)


    def forward(self):
        opts = self.opts

        with torch.no_grad():
            self.delta_shape, _, _, _ = self.model_scale(self.input_imgs)

        cams = self.theta[:, :self.num_cam]
        poses = self.theta[:, self.num_cam:(self.num_cam + self.num_pose)]
        shapes = self.theta[:, (self.num_cam + self.num_pose):]
        scales = self.delta_shape.unsqueeze(2).detach()
        offsets = torch.zeros(self.delta_shape.size(0), 6890, 3).cuda()
        verts, _, _ = self.smpl(shapes, poses, scales, offsets, get_skin=True)

        proc_param = {
            'scale': 1.0*224/128,
            'img_size': 256
        }

        cam_s = cams[:, 0].view(-1, 1)
        cam_pos = cams[:, 1:]
        flength = 500.
        tz = flength / (0.5 * proc_param['img_size'] * cam_s)
        trans = torch.cat([cam_pos, tz], dim=1)
        # print(trans)
        vert_shifted = verts + trans.unsqueeze(1).repeat(1, verts.size(1), 1)

        self.pred_v = vert_shifted

        # Render mask.
        self.mask_pred, self.face_index = self.renderer(self.pred_v, self.faces)

        # Render mask segments.
        self.mask_seg_pred, _ = self.renderer_seg(self.pred_v.detach(), self.faces, textures=self.textures_seg)

        mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in (self.masks > 0).float().cpu().numpy()])
        dt_tensor = torch.FloatTensor(mask_dts).cuda()
        # B x 1 x N x N
        self.dts_barrier = Variable(dt_tensor, requires_grad=False).unsqueeze(1)

        mask_pred_01 = (self.mask_pred.detach() > 0).float().unsqueeze(1)
        x = torch.arange(0, opts.img_size).repeat(opts.img_size, 1).float().cuda()
        coord = torch.cat([x.T.unsqueeze(0), x.unsqueeze(0)], dim=0).unsqueeze(0).repeat(opts.batch_size, 1, 1, 1)
        coord = coord * mask_pred_01
        parsing = (self.mask_seg_pred.detach() - 0.5) * 2.0

        self.textures, self.uv_pred = self.model_flow(coord, parsing, poses)
        self.texture_flow = self.textures
        self.textures = geom_utils.sample_textures(self.texture_flow, self.imgs)
        tex_size = self.textures.size(2)
        self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1, tex_size, 1)

        self.textures_ex = self.textures.unsqueeze(1)
        self.textures_ex = torch.cat([self.textures_ex[1::2], self.textures_ex[::2]], 1).view(-1, 1, self.textures.size(1),
                                                                                              self.textures.size(2), self.textures.size(3), self.textures.size(4), self.textures.size(5))
        self.textures_ex = torch.squeeze(self.textures_ex, 1)

        self.texture_pred, _ = self.tex_renderer(self.pred_v.detach(), self.faces, textures=self.textures)
        self.texture_ex_pred, _ = self.tex_ex_renderer(self.pred_v.detach(), self.faces, textures=self.textures_ex)

        self.texture_pred = self.texture_pred * mask_pred_01 + self.bg_imgs * (1 - mask_pred_01)
        self.texture_ex_pred = self.texture_ex_pred * mask_pred_01 + self.bg_imgs * (1 - mask_pred_01)

        # Compute losses for this instance.
        self.tex_loss = self.texture_loss(self.texture_pred, self.imgs)
        self.tex_ex_loss = self.texture_ex_loss(self.texture_ex_pred, self.imgs)
        self.tex_dt_loss = self.texture_dt_loss_fn(self.texture_flow, self.dts_barrier)
        self.tex_tv_loss = self.tv_loss(self.uv_pred)

        # Finally sum up the loss.
        self.total_loss = opts.tex_loss_wt * self.tex_loss
        self.total_loss += opts.tex_loss_wt * self.tex_ex_loss
        self.total_loss += opts.tex_dt_loss_wt * self.tex_dt_loss
        self.total_loss += opts.tex_tv_loss_wt * self.tex_tv_loss


    def get_current_scalars(self):
        sc_dict = OrderedDict([
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss.data.item()),
        ])
        sc_dict['tex_loss'] = self.tex_loss.data.item()
        sc_dict['tex_ex_loss'] = self.tex_ex_loss.data.item()
        sc_dict['tex_dt_loss'] = self.tex_dt_loss.data.item()
        sc_dict['tex_tv_loss'] = self.tex_tv_loss.data.item() * 10

        return sc_dict


def main(_):
    torch.manual_seed(0)
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
