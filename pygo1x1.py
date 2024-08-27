import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict
from non_local_helper import NLBlockND

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, use_norm=True, zero_mean=True):
      super().__init__()
      self.conv3d = nn.Conv3d(in_channels=in_channels,
         out_channels=out_channels,
         kernel_size=kernel_size,
         stride=stride,
         padding=padding, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
         bias=bias)
      self.use_norm = use_norm
      self.zero_mean = zero_mean
    def forward(self, x):
      weight = self.conv3d.weight
      if self.zero_mean and self.use_norm:
        x = (x - x.mean())
        weight = weight-weight.mean()
      val = torch.nn.functional.conv3d(x, weight, self.conv3d.bias, stride=self.conv3d.stride)
      if False and self.use_norm:
        xlen = torch.linalg.vector_norm(x)
        wlen = torch.linalg.vector_norm(weight)
        norm = (xlen * wlen + 0.000001)
        val = val / norm
      return val

class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.LeakyReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            #f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f_up = F.interpolate(feature_maps[i], feature_maps[i-1].shape, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        print("Pygo1x1 decoder", 'x.shape', x.shape, "upscale", mask.shape)
        return mask

class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.leaky_relu,
                 use_batch_norm=True, # True SethS 8/27/2024
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        #self.conv3d = nn.Conv3d(in_channels=in_channels,
        self.conv3d = Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias, use_norm = use_batch_norm, zero_mean=use_batch_norm) # TODOSethS add separate parameter
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x, 0.05)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name,non_local=False,use_bias=False, use_batch_norm=True, activation_fn=None):
        super(InceptionModule, self).__init__()
        #print("Creating PYGO inception module")
        self.non_local=non_local
        if self.non_local: 
            self.nl_block=NLBlockND(in_channels=in_channels, mode='gaussian', dimension=3)
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1', use_bias=use_bias, use_batch_norm=use_batch_norm, activation_fn=activation_fn) # Let it be. TODO configure all and finally replace logits layer!
        self.name = name

    def forward(self, x):
        if self.non_local:
            x=self.nl_block(x)
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)

#print("Defining Pygo I3d")
class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
       ''' 'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'FinalMaxPool',
        'Dropout',
        'Logits',
        'Predictions','''
    )
    FEATURE_ENDPOINTS=['Conv3d_2c_3x3', 'Mixed_3c'] #,'Mixed_4f','Mixed_5c',]
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,forward_features=True,non_local=False,complexity=2,down=64, is_main=False):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        #print("USING PYGO INCEPTION 3D")
        #if final_endpoint not in self.VALID_ENDPOINTS:
        #    raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self.iter = 0
        self.is_main = is_main
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.forward_features=forward_features
        self.non_local=non_local
        #if self._final_endpoint not in self.VALID_ENDPOINTS:
        #    raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        # Feature size starts at 64x64x64
        # Receptive field starts at 1x1x1
        c = complexity
        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=c*4, kernel_shape=[7, 1, 1],
                                            stride=(2, 1, 1), padding=(3,0,0),  name=name+end_point) # 32,32,32; 7,7,7
        down = down // 2
        if self._final_endpoint == end_point or down == 1: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 1, 1], stride=(1, 1, 1),
                                                             padding=0) # 16,16,32; 14,14,7
        down = down // 2
        if self._final_endpoint == end_point or down == 1: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=c*4, output_channels=c*4, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point) # 16,16,32; 14,14,7
        if self._final_endpoint == end_point: return

        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=c*4, output_channels=c*12, kernel_shape=[3, 1, 1], padding=(1,0,0),
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 1, 1], stride=(2, 1, 1),
                                                             padding=0) # 8,8,32; 28,28,7
        down = down // 2 # 8x downsampling so far, with only 2x channel downsampling (to 15)
        if self._final_endpoint == end_point or down == 1: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(c*12, [c*4,c*6,c*8,c,c*2,c*2], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(c*16, [c*8,c*8,c*12,c*2,c*6,c*4], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 1, 1], stride=(2, 1, 1),
                                                             padding=0)
        down = down // 2 # 16x downsampling so far, and 2x channelwise (8)
        if self._final_endpoint == end_point or down == 1: return # 4,4,8; 56,56,14

        end_point = 'Mixed_4b'
        #self.end_points[end_point] = InceptionModule(c*8+c*12+c*6+c*4, [c*12,c*6,c*13,c,c*3,c*4], name+end_point)
        #self.end_points[end_point] = InceptionModule(c*8+c*12+c*6+c*4, [1,2,4,8,16,32], name+end_point)
        self.end_points[end_point] = InceptionModule(c*8+c*12+c*6+c*4, [16,16,16,16,16,16], name+end_point, use_bias=True, use_batch_norm=False, activation_fn=None)
        if self._final_endpoint == end_point: return

        self.complexity = c
        self.build()

        return


        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(c*12+c*13+c*3+c*4, [c*10,c*7,c*14,c*3//2,c*4,c*4], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(c*10+c*14+c*4+c*4, [c*8,c*8,c*16,c*3//2,c*4,c*4], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(c*8+c*16+c*4+c*4, [c*7,c*9,c*18,c*2,c*4,c*4], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(c*7+c*18+c*4+c*4, [c*16,c*10,c*20,c*2,c*8,c*8], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 1, 1], stride=(2, 1, 1),
                                                             padding=0)
        down = down // 2 # 32x downsampling, and 4x channelwise 2,2,4
        if self._final_endpoint == end_point or down == 1: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(c*16+c*20+c*8+c*8, [c*16,c*10,c*20,c*2,c*8,c*8], name+end_point,non_local=self.non_local)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(c*16+c*20+c*8+c*8, [c*24,c*12,c*24,c*3,c*8,c*8], name+end_point,non_local=self.non_local)
        if self._final_endpoint == end_point: return

        end_point = 'FinalMaxPool'
        #self.end_points[end_point] = self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], # Avg Pool across 2x channels, 7x spatially. Maybe too aggressive pooling here...
        #                             stride=(1, 2, 2))
        self.end_points[end_point] = self.avg_pool = nn.AvgPool3d(kernel_size=[2, 1, 1], # Avg Pool across 2x channels, 7x spatially. Maybe too aggressive pooling here...
                                     stride=(2, 1, 1))
        #self.avg_pool = nn.MaxPool3d(kernel_size=[2, 1, 1], # Avg Pool across 2x channels, 7x spatially. Maybe too aggressive pooling here...
        end_point = "Dropout"
        self.end_points[end_point] = self.dropout = nn.Dropout(dropout_keep_prob) # It's missing dropout.

        end_point = 'Logits'
        down = down // 2 # 64x downsampling
        #self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
        self.logits = Unit3D(in_channels=c*24+c*24+c*8+c*8, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        self.end_points[end_point] = self.logits

        self.final_pool=self.avgpool = nn.AdaptiveMaxPool3d((15, 1, 1)) # THIS will bottleneck all of the predictions down to a single point anyway, so I don't have to worry about downsampling rate! Handy!
        self.complexity = c
        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        c = self.complexity
        #self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
        #self.logits = Unit3D(in_channels=384, output_channels=self._num_classes,
        self.logits = Unit3D(in_channels=c*24+c*24+c*8+c*8, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        if self.iter % 5000 == 0 and self.is_main:
          print("input shape", x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16")
        if self.forward_features:
            features=[]
            if self.iter % 5000 == 0 and self.is_main:
              print("x.shape", x.shape)
            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    x = self._modules[end_point](x) # use _modules to work with dataparallel
                    if self.iter % 5000 == 0 and self.is_main:
                      print("end_point", end_point, x.shape)
                    #print("I3D", end_point, self._modules[end_point].weight.shape, 2 * np.product(self._modules[end_point].weight.shape)/1000000, "MB weights", x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16 activations")
                    if self.iter % 5000 == 0 and self.is_main:
                      print("I3D", end_point, x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16")
                    if True or end_point in self.FEATURE_ENDPOINTS:
                        features.append(x)
            # x = self.logits(self.dropout(self.avg_pool(x)))
            # if self._spatial_squeeze:
            #     logits = x.squeeze(3).squeeze(3)
            # # logits is batch X time X classes, which is what we want to work with
            outshape = list(features[-1].shape)[-3:]
            outshape[0] = 1
            if self.iter % 5000 == 0 and self.is_main:
              print("adaptive max pool", features[-1].shape, outshape)
            x = torch.nn.functional.adaptive_max_pool3d(features[-1], outshape) #features
            if self.iter % 5000 == 0 and self.is_main:
              print("post pool", x.shape)
            x = x.squeeze(2)
            if self.iter % 5000 == 0 and self.is_main:
              print("post squeeze", x.shape)
            self.iter = 1 #+= 1
            return x
        else:
            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    x = self._modules[end_point](x) # use _modules to work with dataparallel
                    print("VE I3D", end_point, x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16")
            x = self.logits(self.dropout(self.avg_pool(x)))
            print("VE I3D", "logits", x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16")
            if self._spatial_squeeze:
                logits = x.squeeze(3).squeeze(3)
            x = self.final_pool(x)
            print("VE I3D", "final_pool", x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16")

            x = x.view(x.size(0), -1)
            print("VE I3D", "reshaped", x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16")
            return x
            # # logits is batch X time X classes, which is what we want to work with

    def extract_features(self, x):
        #print("input shape", x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16")
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
                print("EF I3D", end_point, self._modules[end_point].weight.shape, 2 * np.product(self._modules[end_point].weight.shape)/1000000, "MB weights", x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16 activations")
        x = self.avg_pool(x)
        print("EF I3D", "avg_pool", x.shape, int(2 * np.product(x.shape) / 1000000), "MB FP16")
        return x
