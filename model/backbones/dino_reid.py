"""
DINOv3 Backbone Wrapper for Person Re-Identification
=====================================================
Drop-in replacement for the PersonViT backbone that uses a DINOv3
Vision Transformer loaded via HuggingFace Transformers.

The forward interface is identical to ``build_transformer`` so the
existing training loop, loss functions, and dataloaders work untouched.

Supported model names (HuggingFace)
------------------------------------
    facebook/dinov3-vits14-pretrain-lvd1689m  →  384-dim
    facebook/dinov3-vitb14-pretrain-lvd1689m  →  768-dim
    facebook/dinov3-vitl14-pretrain-lvd1689m  → 1024-dim

Also works with HuggingFace model IDs passed via
``cfg.MODEL.DINO_MODEL_NAME``.
"""

import torch
import torch.nn as nn

try:
    from transformers import DINOv3VisionTransformerBackbone
except ImportError:
    DINOv3VisionTransformerBackbone = None


# ---------------------------------------------------------------------------
# HuggingFace model-name → feature dim lookup
# ---------------------------------------------------------------------------
DINOV3_FEAT_DIMS = {
    "facebook/dinov3-vits14-pretrain-lvd1689m": 384,
    "facebook/dinov3-vits16-pretrain-lvd1689m": 384,
    "facebook/dinov3-vitsplus-pretrain-lvd1689m": 384,
    "facebook/dinov3-vitb14-pretrain-lvd1689m": 768,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": 768,
    "facebook/dinov3-vitl14-pretrain-lvd1689m": 1024,
    "facebook/dinov3-vitl16-pretrain-lvd1689m": 1024,
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": 1536,
}

# Short aliases for convenience
DINOV3_ALIASES = {
    "dinov3_vits14": "facebook/dinov3-vits14-pretrain-lvd1689m",
    "dinov3_vitb14": "facebook/dinov3-vitb14-pretrain-lvd1689m",
    "dinov3_vitl14": "facebook/dinov3-vitl14-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


# ---------------------------------------------------------------------------
# Weight-init helpers (same as make_model.py to stay consistent)
# ---------------------------------------------------------------------------

def _weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def _weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def _weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# DINOv3 → ReID model
# ---------------------------------------------------------------------------

class DinoReID(nn.Module):
    """DINOv3 backbone + ReID head.

    Uses HuggingFace Transformers ``DINOv3VisionTransformerBackbone``
    under the hood, so it works on Kaggle (cached model download),
    Colab, and any environment with ``pip install transformers``.

    Config keys read (all under ``cfg.MODEL`` unless noted):

    Required
    ~~~~~~~~
    * ``DINO_MODEL_NAME`` – HF model ID or short alias
      e.g. ``'dinov3_vitb14'`` or ``'facebook/dinov3-vitb14-pretrain-lvd1689m'``

    Optional
    ~~~~~~~~
    * ``FREEZE_BACKBONE``        – freeze backbone at init  (default False)
    * ``FREEZE_BACKBONE_EPOCHS`` – unfreeze after N epochs  (default 0)
    * ``NECK``                   – 'bnneck' or 'no'
    * ``REDUCE_FEAT_DIM``        – add FC to reduce dim
    * ``FEAT_DIM``               – target dim if reducing
    * ``DROPOUT_RATE``           – dropout before classifier
    * ``TEST.NECK_FEAT``         – 'before' or 'after' BNNeck
    """

    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(DinoReID, self).__init__()

        if DINOv3VisionTransformerBackbone is None:
            raise ImportError(
                "DINOv3VisionTransformerBackbone not found. "
                "Install with: pip install transformers>=4.40"
            )

        # ---- config -------------------------------------------------------
        dino_model_name = cfg.MODEL.DINO_MODEL_NAME

        # Resolve short alias → full HF model ID
        if dino_model_name in DINOV3_ALIASES:
            dino_model_name = DINOV3_ALIASES[dino_model_name]

        self.neck       = cfg.MODEL.NECK
        self.neck_feat  = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim   = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.cos_layer  = cfg.MODEL.COS_LAYER
        self.num_classes = num_classes

        # ---- backbone (HuggingFace) ---------------------------------------
        print(f'[DinoReID] Loading backbone: {dino_model_name}')
        self.backbone = DINOv3VisionTransformerBackbone.from_pretrained(
            dino_model_name
        )

        # Infer feature dimension
        if dino_model_name in DINOV3_FEAT_DIMS:
            self.in_planes = DINOV3_FEAT_DIMS[dino_model_name]
        elif hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'hidden_sizes'):
            self.in_planes = self.backbone.config.hidden_sizes[-1]
        else:
            self.in_planes = self._infer_feat_dim(dino_model_name)

        print(f'[DinoReID] Backbone feature dim = {self.in_planes}')

        # ---- optional dim reduction (FC neck) -----------------------------
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(_weights_init_xavier)
            self.in_planes = self.feat_dim

        # ---- classifier ---------------------------------------------------
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(_weights_init_classifier)

        # ---- BNNeck -------------------------------------------------------
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_weights_init_kaiming)

        # ---- dropout ------------------------------------------------------
        self.dropout = nn.Dropout(self.dropout_rate)

        # ---- optional backbone freeze -------------------------------------
        self.freeze_backbone = cfg.MODEL.FREEZE_BACKBONE
        self.freeze_backbone_epochs = cfg.MODEL.FREEZE_BACKBONE_EPOCHS
        if self.freeze_backbone:
            self._set_backbone_grad(False)
            print(f'[DinoReID] Backbone FROZEN for first '
                  f'{self.freeze_backbone_epochs} epochs')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_feat_dim(model_name: str) -> int:
        """Fallback: infer embed dim from the model name string."""
        name = model_name.lower()
        if '7b' in name or 'vitg' in name:
            return 1536
        if 'vitl' in name:
            return 1024
        if 'vitb' in name:
            return 768
        if 'vits' in name:
            return 384
        raise ValueError(
            f'Cannot infer feature dim from "{model_name}". '
            'Set MODEL.FEAT_DIM explicitly.'
        )

    def _set_backbone_grad(self, requires_grad: bool):
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    def unfreeze_backbone(self):
        self._set_backbone_grad(True)
        self.freeze_backbone = False
        print('[DinoReID] Backbone UNFROZEN')

    # ------------------------------------------------------------------
    # Forward  – matches build_transformer interface exactly
    # ------------------------------------------------------------------

    def forward(self, x, label=None, cam_label=None, view_label=None):
        """
        Parameters
        ----------
        x : Tensor [B, 3, H, W]
        label : Tensor [B]  (unused here, required by training loop)
        cam_label, view_label : ignored (DINOv3 has no camera/view embed)

        Returns
        -------
        Training : (cls_score, global_feat)
        Eval     : feat
        """
        # HuggingFace backbone: pixel_values → pooler_output [B, dim]
        out = self.backbone(pixel_values=x)
        global_feat = out.pooler_output

        # Safety: if pooler_output is somehow 3-D, take CLS token
        if global_feat.dim() == 3:
            global_feat = global_feat[:, 0]

        # Optional dim reduction
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        # BNNeck
        if self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        else:
            feat = global_feat

        feat_cls = self.dropout(feat)

        if self.training:
            cls_score = self.classifier(feat_cls)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    # ------------------------------------------------------------------
    # Checkpoint loading (for test.py compatibility)
    # ------------------------------------------------------------------

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu',
                                weights_only=False)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for key in param_dict:
            try:
                self.state_dict()[key.replace('module.', '')].copy_(
                    param_dict[key]
                )
            except Exception:
                continue
        print(f'[DinoReID] Loaded checkpoint from {trained_path}')
