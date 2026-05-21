"""
NT-CCTV — Brand Classifier  (Local Training on Apple M4)
==========================================================
เทรน MobileNetV3-Small บนเครื่อง macOS M4 ด้วย MPS backend

2-Phase Training:
  Phase 1 — Freeze backbone, train head only   (10 ep,  LR=3e-3)
  Phase 2 — Unfreeze all, full fine-tune       (≤60 ep, LR=5e-4,
             ReduceLROnPlateau + early stopping patience=15)

ผลลัพธ์:
  final.onnx          ← Brand Classifier พร้อมใช้กับ nt_cctv_pipeline.py
  class_names.json    ← class names array

Usage:
    python train_brand_local.py
    python train_brand_local.py --brand_dir "Car Brand" --output final.onnx
    python train_brand_local.py --batch 32 --phase2_epochs 40

Estimated time on M4:
  Phase 1  ~6–8  min
  Phase 2  ~20–30 min  (early stopping ช่วยหยุดเร็วถ้า converge แล้ว)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise SystemExit("timm not found — run: pip install timm")

try:
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError:
    raise SystemExit("scikit-learn not found — run: pip install scikit-learn")


# ═══════════════════════════════════════════════════════════════════════
# Config defaults
# ═══════════════════════════════════════════════════════════════════════
IMG_SIZE     = 224
MODEL_NAME   = "mobilenetv3_small_100"

PHASE1_EPOCHS = 10
PHASE1_LR     = 3e-3

PHASE2_EPOCHS = 60      # max — early stopping จะหยุดเองถ้า converge
PHASE2_LR     = 5e-4
PATIENCE      = 15      # early stopping

# ImageNet normalization
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

SEP = "═" * 64


# ═══════════════════════════════════════════════════════════════════════
# Device
# ═══════════════════════════════════════════════════════════════════════
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: Apple MPS (M-series Neural Engine)")
        return torch.device("mps")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Device: CUDA — {name}")
        return torch.device("cuda")
    print("Device: CPU  (ช้ากว่า MPS/CUDA ~10×)")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════
def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    return train_tf, val_tf


def build_dataloaders(brand_dir: str, batch_size: int, num_workers: int = 2):
    """Stratified 80/20 split → (train_loader, val_loader, class_names)."""
    train_tf, val_tf = build_transforms()

    # โหลด dataset (ใช้ train_tf ก่อน เพื่อรู้ class indices)
    full_ds = ImageFolder(brand_dir, transform=train_tf)
    class_names = full_ds.classes
    targets = np.array(full_ds.targets)

    print(f"\nDataset: {len(full_ds)} images  |  {len(class_names)} classes")
    print(f"Classes: {class_names}")

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_ds = Subset(full_ds, train_idx)

    # val dataset ต้องใช้ val_tf (ไม่มี augmentation)
    val_ds_base = ImageFolder(brand_dir, transform=val_tf)
    val_ds = Subset(val_ds_base, val_idx)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=False, persistent_workers=(num_workers > 0))

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    return train_loader, val_loader, class_names


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {MODEL_NAME}  |  params={n_total:,}  |  classes={num_classes}")
    return model


# ═══════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════
def _one_epoch_train(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def _one_epoch_val(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total_n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item()
        correct    += (out.argmax(1) == labels).sum().item()
        total_n    += labels.size(0)
    return total_loss / len(loader), correct / total_n * 100


# ═══════════════════════════════════════════════════════════════════════
# Phase 1 — Head-only training
# ═══════════════════════════════════════════════════════════════════════
def phase1(model, train_loader, val_loader, device, epochs: int, lr: float,
           ckpt_path: Path) -> float:
    # Freeze backbone, train head only
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # MobileNetV3 มี conv_head ด้วย
    if hasattr(model, "conv_head"):
        for param in model.conv_head.parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer  = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    print(f"\n{SEP}")
    print(f"  Phase 1: Head-only  ({epochs} epochs | LR={lr} | trainable={n_trainable:,})")
    print(SEP)

    best_acc   = 0.0
    t_start    = time.time()

    for ep in range(1, epochs + 1):
        tl = _one_epoch_train(model, train_loader, optimizer, criterion, device)
        vl, acc = _one_epoch_val(model, val_loader, criterion, device)
        scheduler.step()

        mark = ""
        if acc > best_acc:
            best_acc = acc
            torch.save({"epoch": ep, "model_state": model.state_dict()}, str(ckpt_path))
            mark = " ← best ✓"

        elapsed = (time.time() - t_start) / 60
        print(f"  Ep {ep:2d}/{epochs}  train={tl:.4f}  val={vl:.4f}  "
              f"acc={acc:6.2f}%  [{elapsed:.1f}min]{mark}")

    total_min = (time.time() - t_start) / 60
    print(f"{SEP}")
    print(f"  Phase 1 done  best_acc={best_acc:.2f}%  time={total_min:.1f}min")
    return best_acc


# ═══════════════════════════════════════════════════════════════════════
# Phase 2 — Full fine-tune
# ═══════════════════════════════════════════════════════════════════════
def phase2(model, train_loader, val_loader, device, p1_epochs: int,
           epochs: int, lr: float, patience: int,
           p1_ckpt: Path, best_ckpt: Path) -> float:
    # โหลด Phase 1 best weights
    ckpt = torch.load(str(p1_ckpt), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded Phase 1 checkpoint (ep={ckpt['epoch']})")

    # Unfreeze ทุก layer
    for param in model.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Differential LR: backbone เล็กกว่า head
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and "conv_head" not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if "classifier" in n or "conv_head" in n]

    optimizer = AdamW([
        {"params": backbone_params, "lr": lr * 0.2},   # backbone: 1e-4
        {"params": head_params,     "lr": lr},          # head:     5e-4
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=5, min_lr=1e-7 
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n{SEP}")
    print(f"  Phase 2: Full fine-tune  (max {epochs} ep | "
          f"head_LR={lr} | backbone_LR={lr*0.2})")
    print(f"  ReduceLROnPlateau: factor=0.3  patience=5  |  "
          f"early-stop patience={patience}")
    print(f"  Trainable: {n_trainable:,} (all layers)")
    print(SEP)

    best_acc      = 0.0
    patience_cnt  = 0
    t_start       = time.time()

    for ep in range(1, epochs + 1):
        tl = _one_epoch_train(model, train_loader, optimizer, criterion, device)
        vl, acc = _one_epoch_val(model, val_loader, criterion, device)
        scheduler.step(acc)
        cur_lr = optimizer.param_groups[-1]["lr"]

        if acc > best_acc:
            best_acc     = acc
            patience_cnt = 0
            torch.save({"epoch": ep + p1_epochs, "model_state": model.state_dict()},
                       str(best_ckpt))
            mark = " ← best ✓"
        else:
            patience_cnt += 1
            mark = f" (patience {patience_cnt}/{patience})"

        elapsed = (time.time() - t_start) / 60
        print(f"  Ep {ep:2d}/{epochs}  train={tl:.4f}  val={vl:.4f}  "
              f"acc={acc:6.2f}%  lr={cur_lr:.1e}  [{elapsed:.1f}min]{mark}")

        if patience_cnt >= patience:
            print(f"\n  ⏹  Early stopping triggered (no improvement for {patience} epochs)")
            break

    total_min = (time.time() - t_start) / 60
    print(f"{SEP}")
    print(f"  Phase 2 done  best_acc={best_acc:.2f}%  time={total_min:.1f}min")
    return best_acc


# ═══════════════════════════════════════════════════════════════════════
# ONNX Export
# ═══════════════════════════════════════════════════════════════════════
class _ModelWithSoftmax(nn.Module):
    """Wrapper เพิ่ม Softmax ท้าย → output เป็น probabilities เหมือน Brand Classifier.onnx เดิม."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base    = base
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.base(x))


def export_onnx(model: nn.Module, num_classes: int, best_ckpt: Path,
                output_path: Path, class_names: list[str]):
    # โหลด best weights
    ckpt = torch.load(str(best_ckpt), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    export_model = _ModelWithSoftmax(model.cpu())
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["probs"],
        opset_version=12,
        dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}},
    )

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path),
                                providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    test_out = sess.run(None, {"image": dummy.numpy()})[0]
    assert test_out.shape == (1, num_classes), f"Shape mismatch: {test_out.shape}"
    assert abs(test_out[0].sum() - 1.0) < 1e-4, "Output is not valid probability"

    # บันทึก class_names.json ข้างๆ final.onnx
    cn_path = output_path.parent / "class_names.json"
    cn_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2))

    print(f"\n{SEP}")
    print(f"  ✅  Exported → {output_path}")
    print(f"  ✅  Class names → {cn_path}")
    print(f"  Input:  image  [1, 3, {IMG_SIZE}, {IMG_SIZE}]  float32  ImageNet normalized")
    print(f"  Output: probs  [1, {num_classes}]  float32  softmax probabilities")
    print(f"  Classes: {class_names[:4]} ... {class_names[-2:]}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main(args):
    device = get_device()

    # Directories
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    p1_ckpt   = ckpt_dir / "brand_phase1_best.pt"
    best_ckpt = ckpt_dir / "brand_best.pt"

    # Data
    train_loader, val_loader, class_names = build_dataloaders(
        args.brand_dir, args.batch, args.workers
    )
    num_classes = len(class_names)

    # Model
    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()  # noqa: F841

    t_total = time.time()

    # Phase 1
    best_p1 = phase1(model, train_loader, val_loader, device,
                     PHASE1_EPOCHS, PHASE1_LR, p1_ckpt)

    # Phase 2
    best_p2 = phase2(model, train_loader, val_loader, device,
                     PHASE1_EPOCHS, args.phase2_epochs, PHASE2_LR, PATIENCE,
                     p1_ckpt, best_ckpt)

    total_min = (time.time() - t_total) / 60
    print(f"\n{'═'*64}")
    print(f"  Training complete!  Phase1={best_p1:.2f}%  Phase2={best_p2:.2f}%")
    print(f"  Total time: {total_min:.1f} min")
    print("═" * 64)

    # Export
    export_onnx(model, num_classes, best_ckpt, Path(args.output), class_names)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train Brand Classifier locally (M4 MPS) → export final.onnx",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--brand_dir",      default="Car Brand",
                    help="Path to Car Brand dataset directory")
    ap.add_argument("--output",         default="final.onnx",
                    help="Output ONNX file path")
    ap.add_argument("--batch",          type=int, default=64,
                    help="Batch size (64 works well on M4 16GB)")
    ap.add_argument("--phase2_epochs",  type=int, default=PHASE2_EPOCHS,
                    help="Max Phase 2 epochs (early stopping ช่วยหยุดเร็วกว่า)")
    ap.add_argument("--workers",        type=int, default=2,
                    help="DataLoader num_workers")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
