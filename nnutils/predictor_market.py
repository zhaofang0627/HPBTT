"""
Predictor for Market-1501

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle

import torch
import torchvision
from torch.autograd import Variable

from ..nnutils import mesh_net
from ..nnutils import geom_utils
from ..nnutils.nmr import NeuralRendererWOCAM
from ..utils.SMPL_D import SMPL
from ..utils import bird_vis
from ..experiments.get_body_mesh import get_body_mesh


class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts

        self.face_labels = pickle.load(open('./HPBTT/external/hmr/src/tf_smpl/face_labels.pkl', 'rb'), encoding='latin1')
        region_colors = np.array([[0, 0, 0], [1.0, 0, 0], [1.0, 0, 1.0], [0, 1.0, 0], [1.0, 1.0, 0], [0, 0, 1.0], [0, 1.0, 1.0]])
        parsing_colors = np.array([[0, 0, 0], [1.0, 0.8, 0.8], [1.0, 0.6, 1.0], [0.698, 1.0, 0.4], [1.0, 1.0, 0.4], [0.6, 0.8, 1.0], [0.6, 1.0, 1.0]])
        # F(13776) x 3
        face_colors = torch.tensor(region_colors[(self.face_labels + 1).astype(np.int)]).type(torch.FloatTensor).cuda()
        face_colors_parsing = torch.tensor(parsing_colors[(self.face_labels + 1).astype(np.int)]).type(torch.FloatTensor).cuda()
        # B x F(13776) x T x T x T x 3
        self.textures_seg = face_colors.unsqueeze(0).repeat(opts.batch_size, 1, 1).unsqueeze(2).repeat(1, 1, opts.tex_size**2, 1).\
            view(opts.batch_size, 13776, opts.tex_size, opts.tex_size, 3).unsqueeze(4).repeat(1, 1, 1, 1, opts.tex_size, 1)

        self.textures_par = face_colors_parsing.unsqueeze(0).repeat(opts.batch_size, 1, 1).unsqueeze(2).repeat(1, 1, opts.tex_size**2, 1).\
            view(opts.batch_size, 13776, opts.tex_size, opts.tex_size, 3).unsqueeze(4).repeat(1, 1, 1, 1, opts.tex_size, 1)

        print('Setting up model..')
        faces = np.load('./HPBTT/external/hmr/src/tf_smpl/smpl_faces.npy')
        self.m = get_body_mesh('./HPBTT/external/hmr/models/body.obj', trans=np.array([0, 0, 4]), rotation=np.array([np.pi / 2, 0, 0]))

        img_size_scale = (opts.img_size, opts.img_size // 2)
        self.model_scale = mesh_net.MeshScaleNet(img_size_scale, opts, verts=self.m.v, faces=faces, nz_feat=opts.nz_feat, num_classes=751, num_shape_param=1)

        img_size_flow = (opts.img_size, opts.img_size)
        self.model_flow = mesh_net.MeshMP3FlowPoseNet(img_size_flow, opts, verts=self.m.v, faces=faces, nz_feat=opts.nz_feat, n_upconv=6, nc_init=128, predict_flow=True)

        self.load_network(self.model_scale, 'scale', self.opts.num_train_epoch)
        self.load_network(self.model_flow, 'flow', self.opts.num_train_epoch)

        self.model_scale.eval()
        self.model_scale = self.model_scale.cuda(device=self.opts.gpu_id)
        self.model_flow.eval()
        self.model_flow = self.model_flow.cuda(device=self.opts.gpu_id)

        self.num_cam = 3
        self.num_pose = 72
        self.smpl = SMPL('./HPBTT/external/hmr/models/neutral_smpl_with_cocoplus_reg.pkl', opts,
                         obj_saveable=False).cuda(device=opts.gpu_id)
        self.smpl.eval()

        self.renderer = NeuralRendererWOCAM(opts.img_size)

        self.renderer_seg = NeuralRendererWOCAM(opts.img_size, device=0)
        self.renderer_seg.ambient_light_only()

        self.renderer_parsing = NeuralRendererWOCAM(opts.img_size, device=0)
        self.renderer_parsing.set_bgcolor([1., 1., 1.])
        self.renderer_parsing.set_light_dir([0, 1, -1], 0.4)

        if opts.texture:
            self.tex_renderer = NeuralRendererWOCAM(opts.img_size)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        faces = self.model_flow.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.vis_rend = bird_vis.VisRenderer(opts.img_size,
                                             faces.data.cpu().numpy())
        self.vis_rend.set_bgcolor([1., 1., 1.])

        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.totensor = torchvision.transforms.ToTensor()
        self.resnet_transform = torchvision.transforms.Compose([self.totensor, self.normalize])


    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        network.load_state_dict(torch.load(save_path))

        return


    def set_input(self, batch):
        opts = self.opts
        theta_batch = batch['theta']
        input_img_crop_batch = torch.zeros((opts.batch_size, 3, opts.img_size, opts.img_size//2))
        img_crop_batch = torch.zeros((opts.batch_size, 3, opts.img_size, opts.img_size))
        for i in range(opts.batch_size):
            input_img_crop_batch[i] = self.resnet_transform(batch['img_crop'][i, :, 64:192, :])
            img_crop_batch[i] = self.totensor(batch['img_crop'][i])

        self.input_img_crop_batch = Variable(
            input_img_crop_batch.cuda(device=opts.gpu_id), requires_grad=False)

        self.img_crop_batch = Variable(
            img_crop_batch.cuda(device=opts.gpu_id), requires_grad=False)

        self.theta = Variable(torch.tensor(theta_batch.astype(np.float32)).cuda(device=opts.gpu_id), requires_grad=False)


    def predict(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward()
        return self.collect_outputs()


    def forward(self):
        opts = self.opts

        with torch.no_grad():
            self.delta_shape, _, _, _ = self.model_scale(self.input_img_crop_batch)

            cams = self.theta[:, :self.num_cam]  # + self.delta_cam
            poses = self.theta[:, self.num_cam:(self.num_cam + self.num_pose)]
            shapes = self.theta[:, (self.num_cam + self.num_pose):]  # + self.delta_shape

            scales = self.delta_shape.unsqueeze(1)
            offsets = torch.zeros(self.delta_shape.size(0), 6890, 3).cuda(device=self.opts.gpu_id)
            verts, Js, _ = self.smpl(shapes, poses, scales, offsets, get_skin=True)

            proc_param = {
                'scale': 1.0 * 224 / 128,
                'img_size': 256
            }

            cam_s = cams[:, 0].view(-1, 1)
            cam_pos = cams[:, 1:]
            flength = 500.
            tz = flength / (0.5 * proc_param['img_size'] * cam_s)
            trans = torch.cat([cam_pos, tz], dim=1)
            vert_shifted = verts + trans.unsqueeze(1).repeat(1, verts.size(1), 1)

            self.pred_v = vert_shifted

            pose = np.load('./HPBTT/external/hmr/src/tf_smpl/pose_standing.npy')
            theta = np.concatenate((np.array([np.pi, 0, 0]), pose, np.zeros(10)))

            theta_standing = torch.tensor(theta).view(1, -1).repeat(self.delta_shape.size(0), 1).float().cuda(device=self.opts.gpu_id)

            cams = theta_standing[:, :self.num_cam]
            poses_standing = theta_standing[:, self.num_cam:(self.num_cam + self.num_pose)]
            shapes_standing = theta_standing[:, (self.num_cam + self.num_pose):]
            scales_standing = torch.tensor([[[1.0]]]).repeat(self.delta_shape.size(0), 1, 1).cuda(device=self.opts.gpu_id)

            verts_standing, Js, _ = self.smpl(shapes_standing, poses_standing, scales_standing, offsets, get_skin=True)

            cam_s = cams[:, 0].view(-1, 1)
            cam_pos = cams[:, 1:]
            flength = 500.
            tz = flength / (0.5 * proc_param['img_size'] * cam_s)
            trans = torch.cat([cam_pos, tz], dim=1)
            self.vert_standing_shifted = verts_standing + trans.unsqueeze(1).repeat(1, verts_standing.size(1), 1)

            self.mask_pred, _ = self.renderer(self.pred_v, self.faces)

            # Render mask segments.
            self.mask_seg_pred, _ = self.renderer_seg(self.pred_v, self.faces, textures=self.textures_seg)

            # Render texture.
            x = torch.arange(0, opts.img_size).repeat(opts.img_size, 1).float().cuda()
            coord = torch.cat([x.T.unsqueeze(0), x.unsqueeze(0)], dim=0).unsqueeze(0).repeat(opts.batch_size, 1, 1, 1)
            coord = coord * (self.mask_pred.detach() > 0).float().unsqueeze(1)
            parsing = (self.mask_seg_pred.detach() - 0.5) * 2.0
            self.textures, _ = self.model_flow(coord, parsing, poses)

            if self.textures.size(-1) == 2:
                # Flow texture
                self.texture_flow = self.textures
                self.textures = geom_utils.sample_textures(self.textures, self.img_crop_batch)
            if self.textures.dim() == 5:  # B x F x T x T x 3
                tex_size = self.textures.size(2)
                self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1, tex_size, 1)

            # Render texture:
            self.texture_pred, _ = self.tex_renderer(self.pred_v, self.faces, textures=self.textures)

            # B x 2 x H x W
            uv_images = self.model_flow.texture_predictor.uvimage_pred
            # B x H x W x 2
            self.uv_flows = uv_images.permute(0, 2, 3, 1)
            self.uv_images = torch.nn.functional.grid_sample(self.img_crop_batch, self.uv_flows)


    def collect_outputs(self):
        outputs = {
            'verts': self.pred_v.data,
            'verts_stand': self.vert_standing_shifted,
            'mask_pred': self.mask_pred.data,
        }
        outputs['texture'] = self.textures
        outputs['texture_pred'] = self.texture_pred.data

        return outputs
