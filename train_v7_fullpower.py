"""
RTX 4060 FULL POWER - V7 AGGRESSIVE
===================================
Based on V2 (0.4421 IoU) + Maximum GPU utilization:
- Mixed Precision (FP16) = 2x FASTER
- OneCycleLR = Aggressive learning
- 50 Epochs = More training time
- Same proven architecture as V2
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random
from tqdm import tqdm

# MAXIMUM GPU PERFORMANCE
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Config
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

# ============================================================================
# Dataset - Same proven augmentation as V2
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask)

        if self.augment:
            # Horizontal Flip (50%)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Vertical Flip (20%)
            if random.random() > 0.8:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Rotation (-15 to 15)
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                image = TF.rotate(image, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=0)
            
            # Scale (0.8 to 1.2)
            if random.random() > 0.5:
                scale = random.uniform(0.8, 1.2)
                new_w = int(image.width * scale)
                new_h = int(image.height * scale)
                image = TF.resize(image, (new_h, new_w))
                mask = TF.resize(mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask) * 255

        return image, mask

# ============================================================================
# Model - EXACT SAME as V2 (proven to work!)
# ============================================================================

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
        x = x + self.block1(x)  # Residual
        x = self.block2(x)
        return self.classifier(x)

# ============================================================================
# Combined Loss - EXACT SAME as V2
# ============================================================================

class CombinedLoss(nn.Module):
    def __init__(self, class_weights, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        
    def dice_loss(self, pred, target, smooth=1e-6):
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return ce + self.dice_weight * dice

# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union > 0:
            iou_per_class.append((inter / union).cpu().numpy())
        else:
            iou_per_class.append(float('nan'))
    return np.nanmean(iou_per_class), iou_per_class

def evaluate(model, backbone, loader, device):
    model.eval()
    iou_scores = []
    all_class_iou = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(features)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            iou, class_iou = compute_iou(outputs, labels.squeeze(1).long())
            iou_scores.append(iou)
            all_class_iou.append(class_iou)
    model.train()
    return np.nanmean(iou_scores), np.nanmean(all_class_iou, axis=0)

# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("🔥 RTX 4060 FULL POWER - V7 AGGRESSIVE 🔥")
    print("="*70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # === CONFIG ===
    BATCH_SIZE = 8
    EPOCHS = 50  # More training
    MAX_LR = 5e-4  # Higher peak LR
    WEIGHT_DECAY = 1e-4
    
    w = int(((960 / 2) // 14) * 14)  # 476
    h = int(((540 / 2) // 14) * 14)  # 266
    
    print(f"\nConfig:")
    print(f"  Resolution: {w}x{h}")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Max LR: {MAX_LR}")
    print(f"  Mixed Precision: YES (FP16)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    mask_tf = transforms.Compose([
        transforms.Resize((h, w), interpolation=TF.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    # Data
    base_dir = os.path.join(os.path.dirname(script_dir), 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset')
    
    trainset = MaskDataset(os.path.join(base_dir, 'train'), train_tf, mask_tf, augment=True)
    valset = MaskDataset(os.path.join(base_dir, 'val'), val_tf, mask_tf, augment=False)
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\nDataset: {len(trainset)} train, {len(valset)} val")
    
    # Backbone
    print("\nLoading DINOv2 ViT-B/14...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval()
    backbone.to(device)
    
    with torch.no_grad():
        dummy = torch.randn(1, 3, h, w).to(device)
        n_emb = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]
    
    # Model - EXACT SAME as V2
    model = SegmentationHeadConvNeXt(n_emb, n_classes, w // 14, h // 14).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss - EXACT SAME weights as V2
    class_weights = torch.tensor([
        0.05,  # Background
        0.5,   # Trees
        1.0,   # Lush Bushes
        1.0,   # Dry Grass
        1.5,   # Dry Bushes
        3.0,   # Ground Clutter
        8.0,   # Logs (rare)
        4.0,   # Rocks
        0.1,   # Landscape
        0.1,   # Sky
    ]).to(device)
    
    loss_fn = CombinedLoss(class_weights, dice_weight=0.5)
    
    # Optimizer - AdamW like V2
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    
    # OneCycleLR for aggressive learning
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warmup 10%
        anneal_strategy='cos'
    )
    
    # Mixed Precision
    scaler = torch.amp.GradScaler('cuda')
    
    best_iou = 0.0
    best_loss = float('inf')
    
    print("\n" + "="*70)
    print("🚀 Starting FULL POWER Training...")
    print("="*70 + "\n")
    
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Mixed Precision Forward
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                labels = labels.squeeze(1).long()
                loss = loss_fn(outputs, labels)
            
            # Mixed Precision Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        # Validation
        val_iou, class_iou = evaluate(model, backbone, val_loader, device)
        train_loss = np.mean(losses)
        
        print(f"\n📊 Epoch {epoch+1}: Loss={train_loss:.4f}, Val IoU={val_iou:.4f}")
        print(f"   Per-class: {[f'{x:.3f}' for x in class_iou]}")
        
        # Save best
        improved = False
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(script_dir, "best_segmentation_model.pth"))
            print(f"   🏆 NEW BEST IoU: {val_iou:.4f} - Saved!")
            improved = True
        
        if train_loss < best_loss:
            best_loss = train_loss
            print(f"   📉 NEW BEST Loss: {train_loss:.4f}")
            improved = True
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou': best_iou,
            }, os.path.join(script_dir, f"checkpoint_ep{epoch+1}.pth"))
            print(f"   💾 Checkpoint saved")
    
    print("\n" + "="*70)
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Best IoU: {best_iou:.4f}")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Model: best_segmentation_model.pth")
    print("="*70)

if __name__ == "__main__":
    main()
