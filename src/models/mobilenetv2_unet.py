import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DecoderBlock(nn.Module):
    """
    Khối Decoder trong U-Net: Upsample -> Concat skip feature -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Xử lý trường hợp lệch size lẻ do padding
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV2UNet(nn.Module):
    """
    Kiến trúc MobileNetV2-UNet tuân thủ Interface chung:
    - Input: [B, 3, H, W]
    - Output: [B, 1, H, W] (Raw Logits)
    """
    def __init__(self, pretrained=True, out_channels=1):
        super(MobileNetV2UNet, self).__init__()
        
        # 1. Tải Encoder MobileNetV2
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights).features
        
        # Phân chia các Stage của MobileNetV2 để lấy Skip Connections:
        # Stage 0: /2  (kênh: 16)  -> features[0:2]
        # Stage 1: /4  (kênh: 24)  -> features[2:4]
        # Stage 2: /8  (kênh: 32)  -> features[4:7]
        # Stage 3: /16 (kênh: 96)  -> features[7:14]
        # Stage 4: /32 (kênh: 1280)-> features[14:19]
        self.stage0 = backbone[0:2]   # 1/2
        self.stage1 = backbone[2:4]   # 1/4
        self.stage2 = backbone[4:7]   # 1/8
        self.stage3 = backbone[7:14]  # 1/16
        self.stage4 = backbone[14:19] # 1/32

        # 2. Định nghĩa các Decoder Blocks
        self.dec4 = DecoderBlock(in_channels=1280, skip_channels=96, out_channels=256) # 1/32 -> 1/16
        self.dec3 = DecoderBlock(in_channels=256,  skip_channels=32, out_channels=128) # 1/16 -> 1/8
        self.dec2 = DecoderBlock(in_channels=128,  skip_channels=24, out_channels=64)  # 1/8  -> 1/4
        self.dec1 = DecoderBlock(in_channels=64,   skip_channels=16, out_channels=32)  # 1/4  -> 1/2
        self.dec0 = DecoderBlock(in_channels=32,   skip_channels=0,  out_channels=16)  # 1/2  -> 1/1 (Full size)

        # 3. Output Head: Trả về Raw Logits 1 kênh
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder Forward ---
        s0 = self.stage0(x)   # Shape: [B, 16, H/2, W/2]
        s1 = self.stage1(s0)  # Shape: [B, 24, H/4, W/4]
        s2 = self.stage2(s1)  # Shape: [B, 32, H/8, W/8]
        s3 = self.stage3(s2)  # Shape: [B, 96, H/16, W/16]
        s4 = self.stage4(s3)  # Shape: [B, 1280, H/32, W/32]

        # --- Decoder Forward ---
        x = self.dec4(s4, s3) # -> 1/16
        x = self.dec3(x, s2)  # -> 1/8
        x = self.dec2(x, s1)  # -> 1/4
        x = self.dec1(x, s0)  # -> 1/2
        x = self.dec0(x)      # -> 1/1 (Resize về gốc)

        # Output raw logits (Không dùng Sigmoid)
        logits = self.final_conv(x)
        return logits

    def set_encoder_freeze(self, freeze=True):
        """
        Quản lý Freeze/Unfreeze Encoder theo Protocol:
        Khi unfreeze, giữ nguyên BatchNorm ở trạng thái eval mode (frozen statistics).
        """
        encoder_stages = [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]
        for stage in encoder_stages:
            for module in stage.modules():
                # Đóng/mở gradient cho các trọng số
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = not freeze
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad = not freeze
                
                # Giữ nguyên BatchNorm frozen dù unfreeze backbone
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eval()