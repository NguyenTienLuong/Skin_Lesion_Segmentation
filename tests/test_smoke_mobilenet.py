import sys
import os
import torch

# Trỏ path ra thư mục gốc dự án để import được module trong src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mobilenetv2_unet import MobileNetV2UNet

def test_mobilenetv2_unet_smoke():
    print("==================================================")
    print("      RUNNING SMOKE TEST: MobileNetV2-UNet        ")
    print("==================================================")

    # 1. Khởi tạo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] Khoi tao model tren device: {device}")
    model = MobileNetV2UNet(pretrained=True, out_channels=1).to(device)
    print("     -> Done!")

    # 2. Test Shape @128
    print("[2/5] Test Forward Pass @128x128...")
    x128 = torch.randn(2, 3, 128, 128).to(device)
    out128 = model(x128)
    assert out128.shape == (2, 1, 128, 128), f"Loi Shape 128: {out128.shape}"
    print(f"     -> Input: {tuple(x128.shape)} | Output: {tuple(out128.shape)} [PASS]")

    # 3. Test Shape @256
    print("[3/5] Test Forward Pass @256x256...")
    x256 = torch.randn(2, 3, 256, 256).to(device)
    out256 = model(x256)
    assert out256.shape == (2, 1, 256, 256), f"Loi Shape 256: {out256.shape}"
    print(f"     -> Input: {tuple(x256.shape)} | Output: {tuple(out256.shape)} [PASS]")

    # 4. Test Backward & Loss
    print("[4/5] Test Backward Pass & Optimizer...")
    target = torch.randint(0, 2, (2, 1, 128, 128)).float().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss = criterion(out128, target)
    loss.backward()
    optimizer.step()
    print(f"     -> Loss: {loss.item():.4f} [PASS]")

    # 5. Test Freeze / Unfreeze
    print("[5/5] Test Freeze/Unfreeze Encoder...")
    model.set_encoder_freeze(freeze=True)
    assert not any(p.requires_grad for p in model.stage0.parameters()), "Loi Freeze Encoder!"
    print("     -> Freeze Encoder: [PASS]")

    model.set_encoder_freeze(freeze=False)
    assert all(p.requires_grad for p in model.stage0.parameters()), "Loi Unfreeze Encoder!"
    print("     -> Unfreeze Encoder: [PASS]")

    print("==================================================")
    print("         SMOKE TEST COMPLETED SUCCESSFULLY!       ")
    print("==================================================")

if __name__ == "__main__":
    test_mobilenetv2_unet_smoke()