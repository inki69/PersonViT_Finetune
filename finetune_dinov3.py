"""
DINOv3 Finetuning Script for Person Re-Identification
======================================================
Convenience wrapper to finetune a DINOv3 backbone on ReID datasets.
The backbone is loaded via HuggingFace Transformers; no local checkpoint needed.

Usage:
    # Finetune DINOv3-ViT-B/14 on Market-1501
    python finetune_dinov3.py --dataset market --data_dir /path/to/Market-1501

    # Use a larger backbone
    python finetune_dinov3.py --dataset market --dino_model dinov3_vitl14

    # Freeze backbone for 20 epochs, then unfreeze
    python finetune_dinov3.py --dataset market --freeze_epochs 20

    # Custom options
    python finetune_dinov3.py --dataset duke --epochs 120 --batch_size 64 --lr 0.0004
"""

import os
import sys
import argparse
from pathlib import Path

# Add finetuning dir to path
SCRIPT_DIR = Path(__file__).parent
FINETUNING_DIR = SCRIPT_DIR / "finetuning"
sys.path.insert(0, str(FINETUNING_DIR))

# Change to finetuning dir for relative paths
os.chdir(FINETUNING_DIR)


# ---------------------------------------------------------------------------
# DINOv3 model → feature-dim lookup
# ---------------------------------------------------------------------------
DINO_FEAT_DIMS = {
    'dinov3_vits14': 384,
    'dinov3_vitb14': 768,
    'dinov3_vitl14': 1024,
    'dinov3_vitg14': 1536,
}


def get_config_file(dataset):
    """Get the DINOv3 config file path for the given dataset."""
    dataset_map = {
        'market': 'market',
        'market1501': 'market',
        'duke': 'dukemtmc',
        'dukemtmc': 'dukemtmc',
        'msmt17': 'msmt17',
        'msmt': 'msmt17',
        'occ_duke': 'occ_duke',
        'occluded_duke': 'occ_duke',
    }

    dataset_dir = dataset_map.get(dataset.lower())
    if dataset_dir is None:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_map.keys())}")

    config_path = FINETUNING_DIR / 'configs' / dataset_dir / 'dino_v3.yml'
    if not config_path.exists():
        # Fall back to the market config (most complete)
        config_path = FINETUNING_DIR / 'configs' / 'market' / 'dino_v3.yml'
        print(f"[DINOv3] No dataset-specific config found for '{dataset_dir}', "
              f"using Market-1501 config as base.")

    return str(config_path)


def main():
    parser = argparse.ArgumentParser(description="DINOv3 ReID Finetuning")
    parser.add_argument('--dataset', type=str, default='market',
                        choices=['market', 'market1501', 'duke', 'dukemtmc',
                                 'msmt17', 'msmt', 'occ_duke'],
                        help='Dataset to finetune on')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb14',
                        choices=list(DINO_FEAT_DIMS.keys()),
                        help='DINOv3 model variant (loaded via HuggingFace)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory of dataset')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--freeze_epochs', type=int, default=None,
                        help='Freeze backbone for N epochs (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID(s) to use')

    args = parser.parse_args()

    # Get config file
    config_file = get_config_file(args.dataset)
    print(f"[DINOv3] Using config: {config_file}")

    # Build command line options
    opts = []

    # DINOv3-specific overrides
    opts.extend(['MODEL.NAME', 'dino_v3'])
    opts.extend(['MODEL.DINO_MODEL_NAME', args.dino_model])

    # Auto-set feature dim from model name
    feat_dim = DINO_FEAT_DIMS.get(args.dino_model, 768)
    opts.extend(['MODEL.FEAT_DIM', str(feat_dim)])

    # No local pretrained checkpoint needed (weights come from hub)
    opts.extend(['MODEL.PRETRAIN_PATH', ''])
    opts.extend(['MODEL.PRETRAIN_CHOICE', 'imagenet'])

    # Output directory
    if args.output_dir:
        opts.extend(['OUTPUT_DIR', args.output_dir])
    else:
        output_dir = str(SCRIPT_DIR / 'outputs' / f'{args.dataset}_{args.dino_model}')
        opts.extend(['OUTPUT_DIR', output_dir])

    # Data directory
    if args.data_dir:
        opts.extend(['DATASETS.ROOT_DIR', args.data_dir])

    # Freeze epochs
    if args.freeze_epochs is not None:
        opts.extend(['MODEL.FREEZE_BACKBONE', 'True'])
        opts.extend(['MODEL.FREEZE_BACKBONE_EPOCHS', str(args.freeze_epochs)])

    # Training params
    if args.epochs:
        opts.extend(['SOLVER.MAX_EPOCHS', str(args.epochs)])
    if args.batch_size:
        opts.extend(['SOLVER.IMS_PER_BATCH', str(args.batch_size)])
    if args.lr:
        opts.extend(['SOLVER.BASE_LR', str(args.lr)])

    # GPU — set before importing CUDA stuff
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # ------------------------------------------------------------------
    # Import pipeline components (must be after sys.path / chdir setup)
    # ------------------------------------------------------------------
    from config import cfg
    from utils.logger import setup_logger
    from datasets import make_dataloader
    from model.make_model_extended import make_model
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from loss import make_loss
    from processor import do_train
    import torch
    import numpy as np
    import random

    # ------------------------------------------------------------------
    # Register DINOv3-specific config keys that are NOT in defaults.py
    # (yacs allows this with set_new_allowed)
    # ------------------------------------------------------------------
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

    # Load config
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()

    # Set seed
    seed = cfg.SOLVER.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create output dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Setup logger
    logger = setup_logger("dinov3", cfg.OUTPUT_DIR, if_train=True)
    logger.info(f"[DINOv3] Dataset      : {args.dataset}")
    logger.info(f"[DINOv3] Backbone     : {args.dino_model}")
    logger.info(f"[DINOv3] Feature dim  : {feat_dim}")
    logger.info(f"[DINOv3] Freeze epochs: {cfg.MODEL.FREEZE_BACKBONE_EPOCHS}")
    logger.info(f"[DINOv3] Output       : {cfg.OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    train_loader, train_loader_normal, val_loader, \
        num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes,
                       camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(
            optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
            cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
            cfg.SOLVER.WARMUP_METHOD,
        )

    # ------------------------------------------------------------------
    # Backbone-unfreeze callback
    # ------------------------------------------------------------------
    if cfg.MODEL.FREEZE_BACKBONE and cfg.MODEL.FREEZE_BACKBONE_EPOCHS > 0:
        _original_do_train = do_train

        def _do_train_with_unfreeze(cfg, model, center_criterion,
                                     train_loader, val_loader,
                                     optimizer, optimizer_center,
                                     scheduler, loss_fn, num_query,
                                     local_rank):
            """Wraps do_train to unfreeze backbone mid-training.

            Because do_train does not expose an epoch hook, we
            monkey-patch the model's train() method to detect the
            current epoch and unfreeze when appropriate.
            """
            _epoch_counter = {'epoch': 0}
            _real_train = model.train

            def _train_with_hook(mode=True):
                _real_train(mode)
                if mode:
                    _epoch_counter['epoch'] += 1
                    if (_epoch_counter['epoch'] ==
                            cfg.MODEL.FREEZE_BACKBONE_EPOCHS + 1):
                        # +1 because model.train() is called at the START
                        # of each epoch (hence epoch N happens on call N)
                        if hasattr(model, 'unfreeze_backbone'):
                            model.unfreeze_backbone()
                        else:
                            # Distributed wrapper
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

    # ------------------------------------------------------------------
    # Train!
    # ------------------------------------------------------------------
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

    print(f"\n[DINOv3] Training complete!")
    print(f"[DINOv3] Checkpoints saved to: {cfg.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
