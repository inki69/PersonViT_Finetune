"""
PersonViT Model Package
=======================
Wraps the finetuning pipeline's make_model with HuggingFace auto-download.

Usage:
    from Architecture.models.personvit import load_model
    model, cfg = load_model('market1501')
"""

import os
import sys
from pathlib import Path

# Paths
FINETUNING_DIR = Path(__file__).parent / "finetuning"
CACHE_DIR = Path.home() / ".cache" / "personvit"
sys.path.insert(0, str(FINETUNING_DIR))

# HuggingFace model paths: name -> (config_subdir, hf_repo, hf_file, cameras)
MODELS = {
    'market1501': ('market', 'lakeAGI/PersonViTReID', 'market.vits.lup.256x128.wopt.csk.4-8.ar.375.n8.e0220/transformer_120.pth', 6),
    'duke': ('dukemtmc', 'lakeAGI/PersonViTReID', 'duke.vits.lup.256x128.wopt.csk.4-8.ar.375.n8.e0220/transformer_120.pth', 8),
    'msmt17': ('msmt17', 'lakeAGI/PersonViTReID', 'msmt.vits.lup.256x128.wopt.csk.4-8.ar.375.n8.e0220/transformer_120.pth', 15),
    'pretrained': ('market', 'lakeAGI/PersonViT', 'vits.lup.256x128.wopt.csk.4-8.ar.375.n8/checkpoint0220.pth', 0),
}


def download_weights(name):
    """Download weights from HuggingFace."""
    _, repo, path, _ = MODELS[name]
    cache_path = CACHE_DIR / path
    if cache_path.exists():
        return str(cache_path)
    
    from huggingface_hub import hf_hub_download
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo_id=repo, filename=path, local_dir=str(CACHE_DIR))


def load_model(name='market1501', checkpoint=None, num_classes=751):
    """
    Load PersonViT model.
    
    Args:
        name: Model name ('market1501', 'duke', 'msmt17', 'pretrained')
        checkpoint: Custom checkpoint path (optional)
        num_classes: Number of classes
        
    Returns:
        (model, cfg) tuple
    """
    config_subdir, _, _, cameras = MODELS[name]
    ckpt = checkpoint or download_weights(name)
    
    # Switch to finetuning dir for imports
    orig_dir = os.getcwd()
    os.chdir(FINETUNING_DIR)
    
    try:
        from config import cfg
        from model import make_model
        
        cfg.merge_from_file(str(FINETUNING_DIR / "configs" / config_subdir / "vit_small.yml"))
        cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'self', 'MODEL.PRETRAIN_PATH', ckpt])
        cfg.freeze()
        
        model = make_model(cfg, num_class=num_classes, camera_num=cameras, view_num=1)
        model.load_param(ckpt)
        model.eval()
        
        return model, cfg
    finally:
        os.chdir(orig_dir)
