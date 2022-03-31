# from receptivefield.pytorch import PytorchReceptiveField
# from receptivefield.image import get_default_image
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# from torch import sigmoid
# from torch.nn.init import xavier_uniform_, zeros_
# import cv2
# import numpy as np


# def conv(in_planes, out_planes, kernel_size=3):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
#         Linear()
#     )


# def upconv(in_planes, out_planes):
#     return nn.Sequential(
#         nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
#         Linear()
#     )
# class Linear(nn.Module):
#     """An identity activation function"""
#     def forward(self, x):
#         return x
        

# class PoseExpNet(nn.Module):

#     def __init__(self, nb_ref_imgs=2, output_exp=False):
#         super(PoseExpNet, self).__init__()
#         self.nb_ref_imgs = nb_ref_imgs
#         self.output_exp = output_exp

#         conv_planes = [16, 32, 64, 128, 256, 256, 256]
#         self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
#         self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
#         self.conv3 = conv(conv_planes[1], conv_planes[2])
#         self.conv4 = conv(conv_planes[2], conv_planes[3])
#         self.conv5 = conv(conv_planes[3], conv_planes[4])
#         self.conv6 = conv(conv_planes[4], conv_planes[5])
#         self.conv7 = conv(conv_planes[5], conv_planes[6])

#         self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

#         if self.output_exp:
#             upconv_planes = [256, 128, 64, 32, 16]
#             self.upconv5 = upconv(conv_planes[4],   upconv_planes[0])
#             self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
#             self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
#             self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
#             self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

#             self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
#             self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
#             self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
#             self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

#         self.features = self._make_layers()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 uniform_(m.weight.data)
#                 if m.bias is not None:
#                     zeros_(m.bias)

#     def _make_layers(self):
#         #target_image = torch.cat((target_image, target_image, target_image), 1)
#         conv_planes = [16, 32, 64, 128, 256, 256, 256]
#         layers = [
#         conv(3*(1+2), conv_planes[0], kernel_size=7),
#         conv(conv_planes[0], conv_planes[1], kernel_size=5),
#         conv(conv_planes[1], conv_planes[2]),
#         conv(conv_planes[2], conv_planes[3]),
#         conv(conv_planes[3], conv_planes[4]),
#         conv(conv_planes[4], conv_planes[5]),
#         conv(conv_planes[5], conv_planes[6]),
#         nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)]
#         return nn.Sequential(*layers) 

#         # pose = self.pose_pred(out_conv7)
#         # pose = pose.mean(3).mean(2)
#         # pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

#         # return pose

#     def forward(self, x):
#         x = torch.cat((x, x, x), 1)
#         print(x.shape)
#         # index of layers with feature maps
#         select = [3, 5]
#         # self.feature_maps is a list of Tensors, PytorchReceptiveField looks for 
#         # this parameter and compute receptive fields for all Tensors inside it.
#         self.feature_maps = []
#         for l, layer in enumerate(self.features):
#             x = layer(x)
#             if l in select:
#                 self.feature_maps.append(x)
#         return x

# # define model functions
# def model_fn() -> nn.Module:
#     model = PoseExpNet()
#     model.eval()
#     return model

# input_shape = [128, 416, 3]
# rf = PytorchReceptiveField(model_fn)
# rf_params = rf.compute(input_shape = input_shape)
# # plot receptive fields
# rf.plot_rf_grids(
#     custom_image=cv2.imread(r'/media/serlini/data3/Datasets/kitti_odometry/dataset_formatted/00_3/000000.jpg'),
#     figsize=(20, 12), 
#     layout=(1, 2))
# plt.show()

#np.transpose(cv2.imread(r'/media/serlini/data3/Datasets/kitti_odometry/dataset_formatted/00_2/000000.jpg'), (1, 0, 2))
import matplotlib.pyplot as plt
import torch.nn as nn
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image

class Linear(nn.Module):
    """An identity activation function"""
    def forward(self, x):
        return x
# define some example feature extractor, here we compute RFs for two 
# feature maps
class SimpleVGG(nn.Module):
    def __init__(self, disable_activations: bool = False):
        """disable_activations: whether to generate network with Relus or not."""
        super(SimpleVGG, self).__init__()
        self.features = self._make_layers(disable_activations)

    def forward(self, x):
        # index of layers with feature maps
        select = [0,13]
        # self.feature_maps is a list of Tensors, PytorchReceptiveField looks for 
        # this parameter and compute receptive fields for all Tensors inside it.
        self.feature_maps = []
        for l, layer in enumerate(self.features):
            x = layer(x)
            if l in select:
                self.feature_maps.append(x)
        return x

    def _make_layers(self, disable_activations: bool):
        activation = lambda: Linear() if disable_activations else nn.ReLU()
        layers = [
            nn.Conv2d(3, 64, kernel_size=3),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3),
            activation(),
            
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            activation(),
            nn.Conv2d(128, 128, kernel_size=3),
            activation(), # 8
            
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3),
            activation(),
            nn.Conv2d(256, 256, kernel_size=3),
            activation(), # 13
        ]        
        return nn.Sequential(*layers)    

# define model functions
def model_fn() -> nn.Module:
    model = SimpleVGG(disable_activations=True)
    model.eval()
    return model

input_shape = [96, 96, 3]
rf = PytorchReceptiveField(model_fn)
rf_params = rf.compute(input_shape = input_shape)
# plot receptive fields
rf.plot_rf_grids(
    custom_image=get_default_image(input_shape, name='cat'), 
    figsize=(20, 12), 
    layout=(1, 2))

plt.show()