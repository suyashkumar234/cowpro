"""
UNet Backbone for Few-Shot Segmentation - Fixed Version
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

class UNetEncoder(nn.Module):
    """
    UNet Encoder with pretrained ResNet backbone
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], use_pretrained=True):
        super(UNetEncoder, self).__init__()
        self.features = features
        
        if use_pretrained:
            # Use pretrained ResNet34 as backbone
            resnet = models.resnet34(pretrained=True)
            
            # Modify first conv if input channels != 3
            if in_channels != 3:
                self.inc = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.inc.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            else:
                self.inc = resnet.conv1
            
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            
            # Use ResNet layers as encoder
            self.down1 = resnet.layer1  # 64 channels
            self.down2 = resnet.layer2  # 128 channels  
            self.down3 = resnet.layer3  # 256 channels
            self.down4 = resnet.layer4  # 512 channels
            
            print("###### UNet ENCODER: Using ResNet34 pretrained initialization ######")
        else:
            # Standard UNet encoder without pretrained weights
            self.inc = DoubleConv(in_channels, features[0])
            self.down1 = Down(features[0], features[1])
            self.down2 = Down(features[1], features[2])
            self.down3 = Down(features[2], features[3])
            self.down4 = Down(features[3], features[3])
            
            print("###### UNet ENCODER: Training from scratch ######")
        
        self.use_pretrained = use_pretrained

    def forward(self, x):
        if self.use_pretrained:
            # ResNet-based encoder
            x1 = self.relu(self.bn1(self.inc(x)))
            x1_pool = self.maxpool(x1)
            
            x2 = self.down1(x1_pool)  # 64 channels
            x3 = self.down2(x2)       # 128 channels
            x4 = self.down3(x3)       # 256 channels
            x5 = self.down4(x4)       # 512 channels
            
            return [x1, x2, x3, x4, x5]
        else:
            # Standard UNet encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            return [x1, x2, x3, x4, x5]

class UNetDecoder(nn.Module):
    """
    UNet Decoder for generating final segmentation mask - Fixed Version
    """
    def __init__(self, features=[512, 256, 128, 64], n_classes=2, bilinear=True):
        super(UNetDecoder, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        
        # Prototype integration layer
        self.proto_integration = nn.Sequential(
            nn.Conv2d(features[0] + 1, features[0], kernel_size=3, padding=1),  # +1 for prototype scores
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Fixed decoder layers with correct channel calculations
        # x5 (512) + x4 (256) = 768 -> 256
        self.up1 = Up(768, 256 // factor, bilinear)
        # x4 (256) + x3 (128) = 384 -> 128  
        self.up2 = Up(384, 128 // factor, bilinear)
        # x3 (128) + x2 (64) = 192 -> 64
        self.up3 = Up(192, 64 // factor, bilinear)
        # x2 (64) + x1 (64) = 128 -> 64
        self.up4 = Up(128, 64, bilinear)
        
        # Final classification layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, encoder_features, prototype_scores=None):
        """
        Args:
            encoder_features: List of feature maps from encoder [x1, x2, x3, x4, x5]
            prototype_scores: Prototype similarity scores from ALPModule [B, num_proto, H', W']
        """
        x1, x2, x3, x4, x5 = encoder_features
        
        # Print shapes for debugging
        print(f"Encoder feature shapes: x1:{x1.shape}, x2:{x2.shape}, x3:{x3.shape}, x4:{x4.shape}, x5:{x5.shape}")
        
        # Integrate prototype scores with deepest features if available
        if prototype_scores is not None:
            # Resize prototype scores to match x5 spatial dimensions
            proto_resized = F.interpolate(prototype_scores, size=x5.shape[-2:], mode='bilinear', align_corners=False)
            # Take max across prototype dimension to get single channel
            proto_max = torch.max(proto_resized, dim=1, keepdim=True)[0]
            # Concatenate with deepest features
            x5_with_proto = torch.cat([x5, proto_max], dim=1)
            # Integrate prototypes
            x5 = self.proto_integration(x5_with_proto)
            print(f"x5 after proto integration: {x5.shape}")
        
        # Standard UNet decoder path with fixed channel handling
        print(f"Before up1: x5:{x5.shape}, x4:{x4.shape}")
        x = self.up1(x5, x4)  # 512+256=768 -> 256
        print(f"After up1: {x.shape}")
        
        print(f"Before up2: x:{x.shape}, x3:{x3.shape}")
        x = self.up2(x, x3)   # 256+128=384 -> 128
        print(f"After up2: {x.shape}")
        
        print(f"Before up3: x:{x.shape}, x2:{x2.shape}")
        x = self.up3(x, x2)   # 128+64=192 -> 64
        print(f"After up3: {x.shape}")
        
        print(f"Before up4: x:{x.shape}, x1:{x1.shape}")
        x = self.up4(x, x1)   # 64+64=128 -> 64
        print(f"After up4: {x.shape}")
        
        # Final classification
        logits = self.outc(x)
        print(f"Final logits: {logits.shape}")
        
        return logits

class SimpleUNetDecoder(nn.Module):
    """
    Simplified UNet Decoder to avoid channel mismatch issues
    """
    def __init__(self, encoder_channels=[64, 64, 128, 256, 512], n_classes=2):
        super(SimpleUNetDecoder, self).__init__()
        self.n_classes = n_classes
        
        # Prototype integration
        self.proto_conv = nn.Conv2d(encoder_channels[-1] + 1, encoder_channels[-1], 1)
        
        # Simple upsampling blocks
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[-1], encoder_channels[-2], 2, stride=2),
            nn.Conv2d(encoder_channels[-2] * 2, encoder_channels[-2], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[-2]),
            nn.ReLU(inplace=True)
        )
        
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[-2], encoder_channels[-3], 2, stride=2),
            nn.Conv2d(encoder_channels[-3] * 2, encoder_channels[-3], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[-3]),
            nn.ReLU(inplace=True)
        )
        
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[-3], encoder_channels[-4], 2, stride=2),
            nn.Conv2d(encoder_channels[-4] * 2, encoder_channels[-4], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[-4]),
            nn.ReLU(inplace=True)
        )
        
        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[-4], encoder_channels[0], 2, stride=2),
            nn.Conv2d(encoder_channels[0] * 2, encoder_channels[0], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.final_conv = nn.Conv2d(encoder_channels[0], n_classes, 1)
        
    def forward(self, encoder_features, prototype_scores=None):
        x1, x2, x3, x4, x5 = encoder_features
        
        # Integrate prototype scores
        if prototype_scores is not None:
            proto_resized = F.interpolate(prototype_scores, size=x5.shape[-2:], mode='bilinear', align_corners=False)
            proto_max = torch.max(proto_resized, dim=1, keepdim=True)[0]
            x5 = self.proto_conv(torch.cat([x5, proto_max], dim=1))
        
        # Decoder path
        x = self.up_conv1[0](x5)  # Transpose conv
        x = torch.cat([x, x4], dim=1)  # Skip connection
        x = self.up_conv1[1:](x)  # Rest of the block
        
        x = self.up_conv2[0](x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2[1:](x)
        
        x = self.up_conv3[0](x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3[1:](x)
        
        x = self.up_conv4[0](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4[1:](x)
        
        return self.final_conv(x)

class UNetWithPrototypes(nn.Module):
    """
    Complete UNet architecture with prototype integration
    """
    def __init__(self, in_channels=3, n_classes=2, features=[64, 128, 256, 512], use_pretrained=True, use_simple_decoder=True):
        super(UNetWithPrototypes, self).__init__()
        self.encoder = UNetEncoder(in_channels, features, use_pretrained)
        
        if use_simple_decoder:
            # Use simplified decoder to avoid channel issues
            encoder_channels = [64, 64, 128, 256, 512] if use_pretrained else [features[0], features[0], features[1], features[2], features[3]]
            self.decoder = SimpleUNetDecoder(encoder_channels, n_classes)
        else:
            self.decoder = UNetDecoder(features, n_classes)
        
    def forward(self, x, prototype_scores=None):
        encoder_features = self.encoder(x)
        output = self.decoder(encoder_features, prototype_scores)
        return output, encoder_features