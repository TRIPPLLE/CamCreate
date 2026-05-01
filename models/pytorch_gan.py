import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return F.relu(x + self.block(x))

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise

class SpatialChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False)
        )
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(F.adaptive_avg_pool2d(x, 1))
        max_out = self.fc(F.adaptive_max_pool2d(x, 1))
        channel_out = torch.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # Spatial Attention
        avg_pool_spatial = torch.mean(x, dim=1, keepdim=True)
        max_pool_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)
        spatial_out = torch.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_out
        
        return x

class CustomGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(CustomGenerator, self).__init__()
        
        # Encoder
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 128
        
        self.e2 = nn.Sequential(
            nn.Conv2d(features, features*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 64
        
        self.e3 = nn.Sequential(
            nn.Conv2d(features*2, features*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 32
        
        # Bottleneck
        self.attention = SpatialChannelAttention(features*4)
        self.res1 = ResBlock(features*4)
        self.res2 = ResBlock(features*4)
        self.res3 = ResBlock(features*4)
        
        # Decoder 1 (32 -> 64)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features*4, features*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True)
        )
        self.noise1 = NoiseInjection(features*2)
        self.out64 = nn.Sequential(
            nn.Conv2d(features*2, out_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # Decoder 2 (64 -> 128)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features*2 * 2, features, 3, stride=1, padding=1, bias=False), # *2 due to concat
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.noise2 = NoiseInjection(features)
        self.out128 = nn.Sequential(
            nn.Conv2d(features, out_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # Decoder 3 (128 -> 256)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 2, features//2, 3, stride=1, padding=1, bias=False), # *2 due to concat
            nn.BatchNorm2d(features//2),
            nn.ReLU(inplace=True)
        )
        self.noise3 = NoiseInjection(features//2)
        self.out256 = nn.Sequential(
            nn.Conv2d(features//2, out_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        
        # Bottleneck
        a = self.attention(x3)
        r = self.res1(a)
        r = self.res2(r)
        r = self.res3(r)
        
        # Decoder
        d1 = self.up1(r)
        d1 = self.noise1(d1)
        out64 = self.out64(d1)
        
        d2 = self.up2(torch.cat([d1, x2], dim=1))
        d2 = self.noise2(d2)
        out128 = self.out128(d2)
        
        d3 = self.up3(torch.cat([d2, x1], dim=1))
        d3 = self.noise3(d3)
        out256 = self.out256(d3)
        
        return out64, out128, out256

def make_discriminator_net(max_features, n_layers):
    layers = []
    features = 64
    
    # Input layer
    layers.append(nn.utils.spectral_norm(nn.Conv2d(6, features, 4, stride=2, padding=1)))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    
    layer_outputs = [] # To store intermediate for feature matching if needed, but we'll return list
    
    curr_features = features
    for _ in range(1, n_layers):
        next_features = min(curr_features * 2, max_features)
        layers.append(nn.utils.spectral_norm(nn.Conv2d(curr_features, next_features, 4, stride=2, padding=1, bias=False)))
        layers.append(nn.BatchNorm2d(next_features))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        curr_features = next_features
        
    # Output layer
    layers.append(nn.utils.spectral_norm(nn.Conv2d(curr_features, 1, 4, stride=1, padding=1)))
    
    return nn.Sequential(*layers)

class DiscriminatorNet(nn.Module):
    def __init__(self, max_features, n_layers):
        super(DiscriminatorNet, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        
        features = 64
        # First layer
        self.layers.append(nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(6, features, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        curr_features = features
        for _ in range(1, n_layers):
            next_features = min(curr_features * 2, max_features)
            self.layers.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(curr_features, next_features, 4, stride=2, padding=1, bias=False)),
                nn.BatchNorm2d(next_features),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            curr_features = next_features
            
        self.final = nn.utils.spectral_norm(nn.Conv2d(curr_features, 1, 4, stride=1, padding=1))
        
    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        out = self.final(x)
        return out, features

class CustomDiscriminator(nn.Module):
    def __init__(self):
        super(CustomDiscriminator, self).__init__()
        self.global_D = DiscriminatorNet(256, 4)
        self.local_D = DiscriminatorNet(128, 2)
        
    def forward(self, sketch, photo):
        x = torch.cat([sketch, photo], dim=1)
        
        out_global, feat_global = self.global_D(x)
        out_local, feat_local = self.local_D(x)
        
        return out_global, out_local, feat_global + feat_local
