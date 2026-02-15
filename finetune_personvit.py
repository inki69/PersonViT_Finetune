"""
PersonViT Finetuning Script
===========================
Convenience wrapper to finetune PersonViT models on ReID datasets.

Usage:
    # Finetune on Market-1501
    python finetune_personvit.py --dataset market --pretrained path/to/checkpoint0220.pth
    
    # Finetune on DukeMTMC
    python finetune_personvit.py --dataset duke --pretrained path/to/checkpoint0220.pth
    
    # Finetune on MSMT17
    python finetune_personvit.py --dataset msmt17 --pretrained path/to/checkpoint0220.pth
    
    # Custom options
    python finetune_personvit.py --dataset market --model vit_base --epochs 120 --batch_size 64
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


def get_config_file(dataset, model):
    """Get the config file path for dataset and model combo."""
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
    
    model_map = {
        'vit_small': 'vit_small.yml',
        'vit_base': 'vit_base.yml',
        'vit_small_ics': 'vit_small_ics.yml',  # with ICS
        'vit_base_ics': 'vit_base_ics_384.yml',  # with ICS
    }
    
    dataset_dir = dataset_map.get(dataset.lower())
    model_file = model_map.get(model.lower(), 'vit_small.yml')
    
    if dataset_dir is None:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_map.keys())}")
    
    config_path = FINETUNING_DIR / 'configs' / dataset_dir / model_file
    if not config_path.exists():
        # Fallback to vit_small
        config_path = FINETUNING_DIR / 'configs' / dataset_dir / 'vit_small.yml'
    
    return str(config_path)


def main():
    parser = argparse.ArgumentParser(description="PersonViT Finetuning")
    parser.add_argument('--dataset', type=str, default='market',
                        choices=['market', 'market1501', 'duke', 'dukemtmc', 'msmt17', 'msmt', 'occ_duke'],
                        help='Dataset to finetune on')
    parser.add_argument('--model', type=str, default='vit_small',
                        choices=['vit_small', 'vit_base', 'vit_small_ics', 'vit_base_ics'],
                        help='Model architecture')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained checkpoint (e.g., checkpoint0220.pth)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory of dataset')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID(s) to use')
    
    args = parser.parse_args()
    
    # Get config file
    config_file = get_config_file(args.dataset, args.model)
    print(f"[PersonViT] Using config: {config_file}")
    
    # Build command line options
    opts = []
    
    # Pretrained weights
    opts.extend(['MODEL.PRETRAIN_PATH', args.pretrained])
    opts.extend(['MODEL.PRETRAIN_CHOICE', 'imagenet'])
    
    # Output directory
    if args.output_dir:
        opts.extend(['OUTPUT_DIR', args.output_dir])
    else:
        output_dir = str(SCRIPT_DIR / 'outputs' / f'{args.dataset}_{args.model}')
        opts.extend(['OUTPUT_DIR', output_dir])
    
    # Data directory
    if args.data_dir:
        opts.extend(['DATASETS.ROOT_DIR', args.data_dir])
    
    # Training params
    if args.epochs:
        opts.extend(['SOLVER.MAX_EPOCHS', str(args.epochs)])
    if args.batch_size:
        opts.extend(['SOLVER.IMS_PER_BATCH', str(args.batch_size)])
    if args.lr:
        opts.extend(['SOLVER.BASE_LR', str(args.lr)])
    
    # GPU - set before importing CUDA stuff
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Import and run training
    from config import cfg
    from utils.logger import setup_logger
    from datasets import make_dataloader
    from model import make_model
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from loss import make_loss
    from processor import do_train
    import torch
    import numpy as np
    import random
    
    # Load config
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
    logger = setup_logger("personvit", cfg.OUTPUT_DIR, if_train=True)
    logger.info(f"[PersonViT] Dataset: {args.dataset}")
    logger.info(f"[PersonViT] Model: {args.model}")
    logger.info(f"[PersonViT] Pretrained: {args.pretrained}")
    logger.info(f"[PersonViT] Output: {cfg.OUTPUT_DIR}")
    
    # Build dataloader
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    # Build model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    # Build loss
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    
    # Build optimizer
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    
    # Build scheduler
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
    
    # Train
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
        local_rank=0
    )
    
    print(f"\n[PersonViT] Training complete!")
    print(f"[PersonViT] Checkpoints saved to: {cfg.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
