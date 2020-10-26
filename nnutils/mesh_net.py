"""
Mesh net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init

from ..utils import mesh
from . import net_blocks as nb

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

flags.DEFINE_float('span_range_height', 0.9, 'span_range_height')
flags.DEFINE_float('span_range_width', 0.9, 'span_range_width')
flags.DEFINE_integer('grid_height', 5, 'grid_height')
flags.DEFINE_integer('grid_width', 5, 'grid_width')


#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4, pretrained=True):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class ResNetCoord(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetCoord, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True, input_coord=False, pretrained=True):
        super(Encoder, self).__init__()
        if input_coord:
            self.resnet_conv = ResNetCoord(n_blocks=4)
        else:
            self.resnet_conv = ResNetConv(n_blocks=4, pretrained=pretrained)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv(img)

        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc(out_enc_conv1)

        return feat


class TexturePredictorMP2UV3(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=True):
        super(TexturePredictorMP2UV3, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)

        self.enc = nb.fc_stack(nz_feat, (self.nc_init//2)*self.feat_H*self.feat_W, 2)
        self.enc_p = nb.fc_stack(nz_feat, (self.nc_init//2)*self.feat_H*self.feat_W, 2)
        self.enc_pose = nb.fc_stack(72, (self.nc_init//2)*self.feat_H*self.feat_W, 2)
        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        self.decoder = nb.decoder2d(n_upconv, None, (nc_init//2)*3, init_fc=False, nc_final=nc_final, use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat, feat_p, pose):
        # pdb.set_trace()
        uvimage_pred = self.enc(feat)
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init//2, self.feat_H, self.feat_W)
        uvimage_pred_p = self.enc_p(feat_p)
        uvimage_pred_p = uvimage_pred_p.view(uvimage_pred_p.size(0), self.nc_init//2, self.feat_H, self.feat_W)
        uvimage_pred_pose = self.enc_pose(pose)
        uvimage_pred_pose = uvimage_pred_pose.view(uvimage_pred_pose.size(0), self.nc_init//2, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        self.uvimage_pred = self.decoder(torch.cat([uvimage_pred, uvimage_pred_p, uvimage_pred_pose], 1))

        if self.predict_flow:
            self.uvimage_pred = torch.tanh(self.uvimage_pred)
        else:
            self.uvimage_pred = torch.sigmoid(self.uvimage_pred)

        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, self.uv_sampler, align_corners=True)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)
        # print(self.uv_sampler)

        # Contiguous Needed after the permute..
        return tex_pred.contiguous(), self.uvimage_pred


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

        init.constant_(self.pred_layer.weight, 0.0)
        init.constant_(self.pred_layer.bias, 0.0)

    def forward(self, feat):
        trans = self.pred_layer(feat)
        return trans


class Classifier(nn.Module):
    def __init__(self, nz_feat, num_classes):
        super(Classifier, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, num_classes)

        # Initialize pred_layer weights to be small so initial def aren't so big
        init.normal_(self.pred_layer.weight, std=0.001)
        init.constant_(self.pred_layer.bias, 0)

    def forward(self, feat):
        class_pred = self.pred_layer(feat)
        return class_pred


class ShapeParamPredictor(nn.Module):
    """
    Outputs mesh shape parameters
    """

    def __init__(self, nz_feat, num_param):
        super(ShapeParamPredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, num_verts)
        self.pred_layer = nn.Linear(nz_feat, num_param)

        # Initialize pred_layer weights to be small so initial def aren't so big
        # self.pred_layer.weight.data.normal_(0, 0.0001)
        init.constant_(self.pred_layer.weight, 0.0)
        init.constant_(self.pred_layer.bias, 1.0)

    def forward(self, feat):
        delta_s = self.pred_layer(feat)
        return delta_s


#------------ Mesh Net ------------#
#----------------------------------#

class MeshScaleNet(nn.Module):
    def __init__(self, input_shape, opts, verts, faces, nz_feat=100, num_classes=751, num_shape_param=1):
        # Input shape is H x W of the image.
        super(MeshScaleNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.texture
        self.num_classes = num_classes

        num_verts = verts.shape[0]

        self.num_output = num_verts

        self.faces = Variable(torch.LongTensor(faces.astype(np.float32)).cuda(), requires_grad=False)

        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat)
        self.code_predictor = ShapeParamPredictor(nz_feat=nz_feat, num_param=num_shape_param)
        self.cam_predictor = TransPredictor(nz_feat, orth=False)
        self.classifier = Classifier(nz_feat=nz_feat, num_classes=self.num_classes)

    def forward(self, img):
        img_feat = self.encoder(img)
        delta_s_pred = self.code_predictor(img_feat)
        # offset_pred = self.offset_predictor(img_feat)
        delta_c_pred = self.cam_predictor(img_feat)
        class_pred = self.classifier(img_feat)
        return delta_s_pred, delta_c_pred, class_pred, img_feat


class MeshMP3FlowPoseNet(nn.Module):
    def __init__(self, input_shape, opts, verts, faces, nz_feat=100, n_upconv=6, nc_init=128, predict_flow=True):
        # Input shape is H x W of the image.
        super(MeshMP3FlowPoseNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.texture

        num_verts = verts.shape[0]

        self.num_output = num_verts

        verts_np = verts
        faces_np = faces
        self.faces = Variable(torch.LongTensor(faces.astype(np.float32)).cuda(), requires_grad=False)

        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat, input_coord=True)
        self.encoder_parsing = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat, input_coord=False, pretrained=False)

        num_faces = faces.shape[0]

        uv_sampler = mesh.compute_uvsampler(verts_np, faces_np[:num_faces], tex_size=opts.tex_size)
        # F' x T x T x 2
        uv_sampler = Variable(torch.FloatTensor(uv_sampler).cuda(), requires_grad=False)
        # B x F' x T x T x 2
        uv_sampler = uv_sampler.unsqueeze(0).repeat(self.opts.batch_size, 1, 1, 1, 1)
        img_H = int(2**np.floor(np.log2(np.sqrt(num_faces) * opts.tex_size)))
        print(img_H)
        img_W = 2 * img_H

        self.texture_predictor = TexturePredictorMP2UV3(
          nz_feat, uv_sampler, opts, img_H=img_H, img_W=img_W, n_upconv=n_upconv, nc_init=nc_init, predict_flow=predict_flow)
        nb.net_init(self.texture_predictor)

    def forward(self, mask_coord, mask_parsing, pose):
        coord_feat = self.encoder(mask_coord)
        parsing_feat = self.encoder_parsing(mask_parsing)
        texture_pred, uv_pred = self.texture_predictor(coord_feat, parsing_feat, pose)
        return texture_pred, uv_pred

