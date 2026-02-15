"""
PersonViT Finetuning Package
============================
TransReID-style finetuning for PersonViT models.

This package is a copy of the original PersonViT/transreid_pytorch training code,
organized for easy use within the training-pipeline.

Structure:
    config/         - Configuration system (YACS)
    configs/        - YAML config files for different datasets
    datasets/       - Dataset loaders (Market-1501, DukeMTMC, MSMT17, etc.)
    loss/           - Loss functions (softmax, triplet, arcface, etc.)
    model/          - Model definitions (ViT backbones)
    processor/      - Training/inference loops
    solver/         - Optimizer and LR schedulers
    utils/          - Utilities (logging, metrics, reranking)
    
    train.py        - Main training script
    test.py         - Evaluation script

Usage:
    # Train on Market-1501 with ViT-Small
    python train.py --config_file configs/market/vit_small.yml \\
        MODEL.PRETRAIN_PATH path/to/checkpoint0220.pth
    
    # Evaluate
    python test.py --config_file configs/market/vit_small.yml \\
        TEST.WEIGHT path/to/transformer_120.pth
"""

from .config import cfg
