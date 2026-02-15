"""
╔══════════════════════════════════════════════════════════════════════╗
║  DINOv3 ReID Finetuning — Kaggle Notebook                         ║
║  ─────────────────────────────────────────                         ║
║  Run this entire file as a single Kaggle notebook cell,            ║
║  or paste each section into separate cells.                        ║
║                                                                    ║
║  Prerequisites:                                                    ║
║    • Kaggle GPU runtime  (Settings → Accelerator → GPU T4 ×2)     ║
║    • Internet ON         (Settings → Internet → On)               ║
║    • Upload your 'personvit' folder as a Kaggle Dataset            ║
║      OR push to GitHub and clone it here.                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ═══════════════════════════════════════════════════════════════════════
import subprocess, sys

def pip_install(*pkgs):
    for p in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "-q", "--no-warn-script-location", p])

pip_install(
    "yacs",            # config system used by PersonViT
    "timm",            # vision model library
    "transformers",    # HuggingFace — hosts DINOv3 weights
)

print("✅ Dependencies installed")


# ═══════════════════════════════════════════════════════════════════════
# CELL 2 — Configuration  (EDIT THESE)
# ═══════════════════════════════════════════════════════════════════════

# ── How your code gets onto Kaggle ────────────────────────────────────
# Option A  — Uploaded as a Kaggle Dataset named "visoura-personvit"
#             The path will be /kaggle/input/visoura-personvit/...
# Option B  — Clone from GitHub
CODE_SOURCE = "dataset"  # "dataset" or "github"

# If using dataset upload:
KAGGLE_DATASET_SLUG = "visoura-personvit"
# If using github:
GITHUB_REPO = "https://github.com/YOUR_USER/YOUR_REPO.git"

# ── Dataset ───────────────────────────────────────────────────────────
# Market-1501 is available on Kaggle as:
#   https://www.kaggle.com/datasets/pengcw1/market-1501
# Add it as a dataset in your notebook.
MARKET1501_DATASET_SLUG = "pengcw1/market-1501"  # Kaggle dataset slug
DATASET_NAME = "market"  # market | duke | msmt17

# ── DINOv3 model ──────────────────────────────────────────────────────
DINO_MODEL = "dinov3_vitb14"   # dinov3_vits14 | dinov3_vitb14 | dinov3_vitl14

# ── Training parameters ──────────────────────────────────────────────
EPOCHS        = 120
BATCH_SIZE    = 64    # T4 (16 GB): 64 for vitb, 32 for vitl
LR            = 0.0004
FREEZE_EPOCHS = 10    # freeze backbone for first N epochs

# ═══════════════════════════════════════════════════════════════════════
# CELL 3 — Setup file structure
# ═══════════════════════════════════════════════════════════════════════
import os, shutil
from pathlib import Path

WORK_DIR  = Path("/kaggle/working")
PERSONVIT = WORK_DIR / "personvit"

if CODE_SOURCE == "dataset":
    SRC = Path(f"/kaggle/input/{KAGGLE_DATASET_SLUG}")
    # Kaggle input is read-only, so we copy into /kaggle/working/
    if not PERSONVIT.exists():
        # Find the personvit folder inside the dataset
        candidates = list(SRC.rglob("finetuning"))
        if candidates:
            src_personvit = candidates[0].parent
        else:
            src_personvit = SRC
        shutil.copytree(str(src_personvit), str(PERSONVIT))
        print(f"✅ Copied code to {PERSONVIT}")
    else:
        print(f"ℹ️  Code already at {PERSONVIT}")

elif CODE_SOURCE == "github":
    if not PERSONVIT.exists():
        os.system(f"git clone {GITHUB_REPO} {PERSONVIT}")
        print(f"✅ Cloned to {PERSONVIT}")
    else:
        print(f"ℹ️  Repo already at {PERSONVIT}")

# Verify structure
FINETUNING = PERSONVIT / "finetuning"
assert FINETUNING.exists(), (
    f"Expected finetuning dir at {FINETUNING}. "
    "Check your dataset upload structure."
)

print("\n📁 Finetuning directory contents:")
for p in sorted(FINETUNING.iterdir()):
    print(f"   {'📂' if p.is_dir() else '📄'} {p.name}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 4 — Locate dataset
# ═══════════════════════════════════════════════════════════════════════

# Market-1501 on Kaggle is typically at:
DATA_ROOT = Path(f"/kaggle/input/{MARKET1501_DATASET_SLUG.split('/')[-1]}")

# Some Kaggle datasets nest the folder differently:
if not DATA_ROOT.exists():
    DATA_ROOT = Path(f"/kaggle/input/{MARKET1501_DATASET_SLUG.replace('/', '-')}")
if not DATA_ROOT.exists():
    # Try to find it
    possible = list(Path("/kaggle/input").glob("*market*"))
    if possible:
        DATA_ROOT = possible[0]
    else:
        raise FileNotFoundError(
            "Market-1501 dataset not found. "
            "Add 'pengcw1/market-1501' as a Kaggle dataset input."
        )

print(f"✅ Dataset root: {DATA_ROOT}")
print("   Contents:", [p.name for p in DATA_ROOT.iterdir()])


# ═══════════════════════════════════════════════════════════════════════
# CELL 5 — Setup Python path & working directory
# ═══════════════════════════════════════════════════════════════════════
import sys

sys.path.insert(0, str(FINETUNING))
os.chdir(FINETUNING)

print(f"✅ Working dir: {os.getcwd()}")
print(f"✅ Python path includes: {FINETUNING}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 6 — Import pipeline & register DINOv3 config keys
# ═══════════════════════════════════════════════════════════════════════
import torch
import numpy as np
import random

from config import cfg
from utils.logger import setup_logger
from datasets import make_dataloader
from model.make_model_extended import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train

print(f"✅ PyTorch {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════
# CELL 7 — Build config
# ═══════════════════════════════════════════════════════════════════════

# Feature dim lookup
FEAT_DIMS = {
    'dinov3_vits14': 384,
    'dinov3_vitb14': 768,
    'dinov3_vitl14': 1024,
}
feat_dim = FEAT_DIMS.get(DINO_MODEL, 768)

# Find config file
config_path = FINETUNING / "configs" / DATASET_NAME / "dino_v3.yml"
if not config_path.exists():
    config_path = FINETUNING / "configs" / "market" / "dino_v3.yml"
    print(f"⚠️  No DINOv3 config for '{DATASET_NAME}', using market config")

# Register DINOv3 keys (not in defaults.py)
cfg.defrost()
if not hasattr(cfg.MODEL, 'DINO_MODEL_NAME'):
    cfg.MODEL.DINO_MODEL_NAME = 'dinov3_vitb14'
if not hasattr(cfg.MODEL, 'DINO_HUB_SOURCE'):
    cfg.MODEL.DINO_HUB_SOURCE = 'facebookresearch/dinov3'
if not hasattr(cfg.MODEL, 'FREEZE_BACKBONE'):
    cfg.MODEL.FREEZE_BACKBONE = False
if not hasattr(cfg.MODEL, 'FREEZE_BACKBONE_EPOCHS'):
    cfg.MODEL.FREEZE_BACKBONE_EPOCHS = 0
cfg.freeze()

# Merge YAML + overrides
OUTPUT_DIR = str(WORK_DIR / "output" / f"{DATASET_NAME}_{DINO_MODEL}")

cfg.defrost()
cfg.merge_from_file(str(config_path))
cfg.merge_from_list([
    'MODEL.NAME',                  'dino_v3',
    'MODEL.DINO_MODEL_NAME',       DINO_MODEL,
    'MODEL.FEAT_DIM',              str(feat_dim),
    'MODEL.PRETRAIN_PATH',         '',
    'MODEL.PRETRAIN_CHOICE',       'imagenet',
    'MODEL.FREEZE_BACKBONE',       'True',
    'MODEL.FREEZE_BACKBONE_EPOCHS', str(FREEZE_EPOCHS),
    'DATASETS.ROOT_DIR',           str(DATA_ROOT),
    'SOLVER.MAX_EPOCHS',           str(EPOCHS),
    'SOLVER.IMS_PER_BATCH',        str(BATCH_SIZE),
    'SOLVER.BASE_LR',              str(LR),
    'OUTPUT_DIR',                   OUTPUT_DIR,
])
cfg.freeze()

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n✅ Config built")
print(f"   Model     : {cfg.MODEL.NAME}")
print(f"   Backbone  : {cfg.MODEL.DINO_MODEL_NAME}")
print(f"   Feat dim  : {feat_dim}")
print(f"   Freeze    : {FREEZE_EPOCHS} epochs")
print(f"   Dataset   : {cfg.DATASETS.NAMES} @ {cfg.DATASETS.ROOT_DIR}")
print(f"   Epochs    : {cfg.SOLVER.MAX_EPOCHS}")
print(f"   Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
print(f"   LR        : {cfg.SOLVER.BASE_LR}")
print(f"   Output    : {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 8 — Set seed & logger
# ═══════════════════════════════════════════════════════════════════════
seed = cfg.SOLVER.SEED
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

logger = setup_logger("dinov3_kaggle", OUTPUT_DIR, if_train=True)
logger.info("=" * 60)
logger.info("DINOv3 ReID Finetuning on Kaggle")
logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# CELL 9 — Build dataloader, model, loss, optimizer, scheduler
# ═══════════════════════════════════════════════════════════════════════

# Dataloader
train_loader, train_loader_normal, val_loader, \
    num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
print(f"✅ Dataloader ready — {num_classes} classes, {num_query} query images")

# Model (DINOv3 weights download automatically from HuggingFace)
model = make_model(cfg, num_class=num_classes,
                   camera_num=camera_num, view_num=view_num)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model built — {total_params/1e6:.1f}M params total, "
      f"{train_params/1e6:.1f}M trainable")

# Loss
loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
print(f"✅ Loss: {cfg.MODEL.METRIC_LOSS_TYPE}")

# Optimizer
optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

# Scheduler
if cfg.SOLVER.WARMUP_METHOD == 'cosine':
    scheduler = create_scheduler(cfg, optimizer)
else:
    scheduler = WarmupMultiStepLR(
        optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
        cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
        cfg.SOLVER.WARMUP_METHOD,
    )
print(f"✅ Optimizer: {cfg.SOLVER.OPTIMIZER_NAME}, Scheduler: {cfg.SOLVER.WARMUP_METHOD}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 10 — Backbone-unfreeze hook
# ═══════════════════════════════════════════════════════════════════════

if cfg.MODEL.FREEZE_BACKBONE and cfg.MODEL.FREEZE_BACKBONE_EPOCHS > 0:
    _original_do_train = do_train

    def _do_train_with_unfreeze(cfg, model, center_criterion,
                                 train_loader, val_loader,
                                 optimizer, optimizer_center,
                                 scheduler, loss_fn, num_query,
                                 local_rank):
        _epoch_counter = {'epoch': 0}
        _real_train = model.train

        def _train_with_hook(mode=True):
            _real_train(mode)
            if mode:
                _epoch_counter['epoch'] += 1
                if (_epoch_counter['epoch'] ==
                        cfg.MODEL.FREEZE_BACKBONE_EPOCHS + 1):
                    if hasattr(model, 'unfreeze_backbone'):
                        model.unfreeze_backbone()
                    else:
                        model.module.unfreeze_backbone()
            return model

        model.train = _train_with_hook
        return _original_do_train(
            cfg, model, center_criterion,
            train_loader, val_loader,
            optimizer, optimizer_center,
            scheduler, loss_fn, num_query, local_rank,
        )

    do_train = _do_train_with_unfreeze
    print(f"✅ Backbone will unfreeze after epoch {FREEZE_EPOCHS}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 11 — TRAIN! 🚀
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🚀 Starting DINOv3 ReID Training")
print("=" * 60 + "\n")

do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_func,
    num_query,
    local_rank=0,
)

print(f"\n✅ Training complete!")
print(f"📁 Checkpoints saved to: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 12 — (Optional) Save best checkpoint as Kaggle output
# ═══════════════════════════════════════════════════════════════════════
output_files = list(Path(OUTPUT_DIR).glob("*.pth"))
if output_files:
    print("\n📦 Saved checkpoints:")
    for f in sorted(output_files):
        size_mb = f.stat().st_size / 1e6
        print(f"   {f.name}  ({size_mb:.1f} MB)")
    print("\n💡 Download from Kaggle Output tab, or use:")
    print('   from kaggle_secrets import UserSecretsClient')
    print('   # then upload to your Kaggle Dataset')
else:
    print("⚠️  No checkpoint files found")
