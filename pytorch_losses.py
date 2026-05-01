import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Select layers (similar to the TF implementation)
        # block1_conv2, block2_conv2, block3_conv4, block4_conv4, block5_conv4
        # In torchvision VGG19 features:
        # conv1_2: 3, conv2_2: 8, conv3_4: 17, conv4_4: 26, conv5_4: 35
        self.slice1 = vgg[:4]   # block1_conv2
        self.slice2 = vgg[4:9]  # block2_conv2
        self.slice3 = vgg[9:18] # block3_conv4
        self.slice4 = vgg[18:27]# block4_conv4
        self.slice5 = vgg[27:36]# block5_conv4
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, y_true, y_pred):
        # inputs are in [-1, 1], need to be in [0, 1] for VGG (standard)
        # or we match the exact TF preprocess. TF was: (x + 1) * 127.5 then vgg_preprocess.
        # PyTorch standard is: (x + 1) / 2 then normalize with imagenet mean/std.
        
        def normalize(x):
            x = (x + 1.0) / 2.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x - mean) / std

        y_true = normalize(y_true)
        y_pred = normalize(y_pred)
        
        loss = 0
        
        # Pass through slices
        h1_true = self.slice1(y_true)
        h1_pred = self.slice1(y_pred)
        loss += F.l1_loss(h1_pred, h1_true)
        
        h2_true = self.slice2(h1_true)
        h2_pred = self.slice2(h1_pred)
        loss += F.l1_loss(h2_pred, h2_true)
        
        h3_true = self.slice3(h2_true)
        h3_pred = self.slice3(h2_pred)
        loss += F.l1_loss(h3_pred, h3_true)
        
        h4_true = self.slice4(h3_true)
        h4_pred = self.slice4(h3_pred)
        loss += F.l1_loss(h4_pred, h4_true)
        
        h5_true = self.slice5(h4_true)
        h5_pred = self.slice5(h4_pred)
        loss += F.l1_loss(h5_pred, h5_true)
        
        return loss

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        
    def forward(self, feat_real, feat_fake):
        loss = 0
        for r, f in zip(feat_real, feat_fake):
            loss += F.l1_loss(f, r.detach())
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel filters
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        
        # Tile for 3 channels
        self.register_buffer('filter_x', kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer('filter_y', kernel_y.repeat(3, 1, 1, 1))
        
    def forward(self, img1, img2):
        # Depthwise conv in PyTorch is achieved by setting groups=in_channels
        edge1_x = F.conv2d(img1, self.filter_x, padding=1, groups=3)
        edge1_y = F.conv2d(img1, self.filter_y, padding=1, groups=3)
        
        edge2_x = F.conv2d(img2, self.filter_x, padding=1, groups=3)
        edge2_y = F.conv2d(img2, self.filter_y, padding=1, groups=3)
        
        mag1 = torch.sqrt(edge1_x**2 + edge1_y**2 + 1e-6)
        mag2 = torch.sqrt(edge2_x**2 + edge2_y**2 + 1e-6)
        
        return F.l1_loss(mag1, mag2)

def gan_loss(out, is_real):
    target = torch.ones_like(out) if is_real else torch.zeros_like(out)
    return F.mse_loss(out, target)
