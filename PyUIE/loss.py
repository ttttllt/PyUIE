import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16, VGG16_Weights
from torchvision import models
from torchvision.models import vgg19

class DynamicLossWeighting(nn.Module):
    def __init__(self, num_losses):
        super(DynamicLossWeighting, self).__init__()
        self.num_losses = num_losses
        
        self.weights = nn.Parameter(torch.ones(num_losses, requires_grad=True, dtype=torch.float32))
        self.epoch_counter = 0

    def forward(self, losses):

        total_loss = sum(losses)
        normalized_losses = [loss / (total_loss + 1e-8) for loss in losses]
        weighted_losses = [1 / (normalized_loss + 1e-8) * loss for normalized_loss, loss in zip(normalized_losses, losses)]

        weights_sum = sum(self.weights)
        normalized_weights = [w / weights_sum for w in self.weights]

        final_losses = [w * l for w, l in zip(normalized_weights, weighted_losses)]
        total_loss = sum(final_losses)
        return total_loss






class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features[:23]  # 使用更深的层
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {'3': "relu1_2",
                                   '8': "relu2_2",
                                   '15': "relu3_3",
                                   '22': "relu4_2"}
        self.vgg_layers.eval()

    def output_features(self, x):
        output = {}
        with torch.no_grad():
            for name, module in self.vgg_layers.named_children():
                x = module(x)
                if name in self.layer_name_mapping:
                    output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        loss = sum(F.smooth_l1_loss(pred_im_feature, gt_feature) for pred_im_feature, gt_feature in
                   zip(pred_im_features, gt_features))
        loss = loss / (len(pred_im_features) + 1e-8)
        #print("PerceptualLoss:", loss.item())
        return loss

class ColorLosstwo(nn.Module):
    def __init__(self):
        super(ColorLosstwo, self).__init__()

    def forward(self, inp, tar):
        # 标准化输入和目标图像
        inp_mean = inp.mean(dim=[2, 3], keepdim=True)
        inp_std = inp.std(dim=[2, 3], keepdim=True)
        inp = (inp - inp_mean) / (inp_std + 1e-8)

        tar_mean = tar.mean(dim=[2, 3], keepdim=True)
        tar_std = tar.std(dim=[2, 3], keepdim=True)
        tar = (tar - tar_mean) / (tar_std + 1e-8)

        R_mu_inp = inp[:, 0, :, :].mean(dim=[1, 2], keepdim=True)
        G_mu_inp = inp[:, 1, :, :].mean(dim=[1, 2], keepdim=True)
        B_mu_inp = inp[:, 2, :, :].mean(dim=[1, 2], keepdim=True)

        loss = (R_mu_inp - G_mu_inp) ** 2 + (R_mu_inp - B_mu_inp) ** 2 + (G_mu_inp - B_mu_inp) ** 2
        loss = loss.mean()

        return loss + 1e-8



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class UnContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(UnContrastLoss, self).__init__()
        self.vgg = Vgg19()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, original_image, enhanced_image, reference_image):
        original_vgg, enhanced_vgg, reference_vgg = self.vgg(original_image), self.vgg(enhanced_image), self.vgg(reference_image)
        loss = 0
        for i in range(len(original_vgg)):
            d_ap = self.l1(original_vgg[i], reference_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(original_vgg[i], enhanced_vgg[i].detach())
                Uncontrastive = d_ap / (d_an + 1e-7)
            else:
                Uncontrastive = d_ap
            loss += self.weights[i] * Uncontrastive
        return loss










