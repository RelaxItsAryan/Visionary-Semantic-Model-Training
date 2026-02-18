"""
Team Visionary - Model Evaluation Script (test.py)
Evaluates the trained model on validation data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import json

# ============================================================================
# Model: Segmentation Head V2 (MUST MATCH TRAINING)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    """V2 Architecture - ConvNeXt-style with BatchNorm + Residual Connections"""
    def __init__(self, in_channels, out_channels, tokenW, tokenH, hidden_channels=256):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, padding=3, groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, padding=3, groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.classifier(x)

# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# ============================================================================
# Metrics
# ============================================================================

def calculate_iou(pred, target, n_classes):
    """Calculate per-class IoU and mean IoU."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(n_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        iou = (intersection / (union + 1e-6)).item()
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    return ious, mean_iou

# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("TEAM VISIONARY - Model Evaluation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Configuration
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, "best_segmentation_model.pth")
    
    # Find validation data
    possible_val_paths = [
        os.path.join(parent_dir, 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'val'),
        os.path.join(script_dir, 'val'),
        os.path.join(parent_dir, 'Offroad_Segmentation_Scripts', 'val'),
    ]
    
    val_dir = None
    for p in possible_val_paths:
        if os.path.exists(p):
            val_dir = p
            break
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    if not val_dir:
        print("[ERROR] Validation data not found!")
        return
    
    print(f"\nModel: {model_path}")
    print(f"Validation: {val_dir}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
    
    # Dataset
    val_dataset = MaskDataset(val_dir, transform, mask_transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"Validation images: {len(val_dataset)}")
    
    # Load backbone
    print("\nLoading DINOv2 backbone...")
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14")
    backbone.eval()
    backbone.to(device)
    
    # Get embedding dimension
    dummy_img = torch.randn(1, 3, h, w).to(device)
    with torch.no_grad():
        dummy_out = backbone.forward_features(dummy_img)["x_norm_patchtokens"]
    n_embedding = dummy_out.shape[2]
    
    # Load classifier
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14,
        hidden_channels=256
    )
    classifier.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    classifier.to(device)
    classifier.eval()
    print("Model loaded!\n")
    
    # Evaluation
    all_ious = []
    all_losses = []
    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.squeeze(1).to(device).long()
            
            features = backbone.forward_features(images)["x_norm_patchtokens"]
            logits = classifier(features)
            outputs = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            
            loss = criterion(outputs, masks)
            all_losses.append(loss.item())
            
            preds = torch.argmax(outputs, dim=1)
            ious, mean_iou = calculate_iou(preds, masks, n_classes)
            all_ious.append(ious)
    
    # Results
    avg_loss = np.mean(all_losses)
    avg_ious = np.mean(all_ious, axis=0)
    mean_iou = np.mean(avg_ious)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Loss: {avg_loss:.4f}")
    
    print(f"\nPer-Class IoU:")
    print("-" * 40)
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, avg_ious)):
        bar = "█" * int(iou * 20) + "░" * (20 - int(iou * 20))
        print(f"  {i}: {name:<15} {bar} {iou:.4f}")
    
    # Save results
    results = {
        "mean_iou": float(mean_iou),
        "loss": float(avg_loss),
        "per_class_iou": {name: float(iou) for name, iou in zip(CLASS_NAMES, avg_ious)}
    }
    
    with open(os.path.join(script_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: evaluation_results.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
