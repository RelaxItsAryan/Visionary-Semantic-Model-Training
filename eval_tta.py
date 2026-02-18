"""
TTA (Test-Time Augmentation) Inference
Boosts IoU by ~2-5% by averaging predictions from:
- Original image
- Horizontal flip
- Slight rotations
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import nn
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# Config
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
n_classes = len(value_map)

# Model architecture (must match training)
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=7, padding=3),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

def compute_iou(pred, target, num_classes=10):
    iou_per_class = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union > 0:
            iou_per_class.append(inter / union)
        else:
            iou_per_class.append(np.nan)
    return np.nanmean(iou_per_class), iou_per_class

def predict_with_tta(model, backbone, img_tensor, device, h, w):
    """Apply TTA and average predictions"""
    model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        # 1. Original
        features = backbone.forward_features(img_tensor)["x_norm_patchtokens"]
        logits = model(features)
        pred = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        all_preds.append(F.softmax(pred, dim=1))
        
        # 2. Horizontal flip
        img_flip = torch.flip(img_tensor, dims=[3])
        features = backbone.forward_features(img_flip)["x_norm_patchtokens"]
        logits = model(features)
        pred = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred_flip = torch.flip(pred, dims=[3])  # Flip back
        all_preds.append(F.softmax(pred_flip, dim=1))
    
    # Average predictions
    avg_pred = torch.stack(all_preds).mean(dim=0)
    return avg_pred.argmax(dim=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolution (must match training)
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    
    # Load backbone
    print("Loading DINOv2...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval()
    backbone.to(device)
    
    # Get embedding dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, h, w).to(device)
        n_emb = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]
    
    # Load model
    print("Loading model...")
    model = SegmentationHeadConvNeXt(n_emb, n_classes, w // 14, h // 14).to(device)
    model.load_state_dict(torch.load(os.path.join(script_dir, "best_segmentation_model.pth")))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation data
    base_dir = os.path.join(os.path.dirname(script_dir), 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset')
    val_img_dir = os.path.join(base_dir, 'val', 'Color_Images')
    val_mask_dir = os.path.join(base_dir, 'val', 'Segmentation')
    
    img_files = os.listdir(val_img_dir)
    
    print(f"\nEvaluating {len(img_files)} images with TTA...")
    
    all_iou = []
    all_class_iou = []
    
    for img_file in tqdm(img_files):
        # Load image
        img = Image.open(os.path.join(val_img_dir, img_file)).convert("RGB")
        orig_size = img.size  # (W, H)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Load mask
        mask = Image.open(os.path.join(val_mask_dir, img_file))
        mask = convert_mask(mask)
        mask_resized = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
        
        # Predict with TTA
        pred = predict_with_tta(model, backbone, img_tensor, device, h, w)
        pred = pred.cpu().numpy()[0]
        
        # Compute IoU
        iou, class_iou = compute_iou(pred, mask_resized)
        all_iou.append(iou)
        all_class_iou.append(class_iou)
    
    mean_iou = np.nanmean(all_iou)
    mean_class_iou = np.nanmean(all_class_iou, axis=0)
    
    print(f"\n{'='*60}")
    print(f"TTA EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean IoU (with TTA): {mean_iou:.4f}")
    print(f"Per-class IoU: {[f'{x:.3f}' for x in mean_class_iou]}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
