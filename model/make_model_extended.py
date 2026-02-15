"""
Extended make_model with DINOv3 support
=======================================
Drop-in replacement for ``from model import make_model``.

    from model.make_model_extended import make_model

Adds a ``dino_v3`` branch while delegating everything else to the
original ``make_model`` — no existing files are modified.
"""

from .make_model import make_model as _original_make_model


def make_model(cfg, num_class, camera_num, view_num):
    """Build a model based on config.

    If ``cfg.MODEL.NAME == 'dino_v3'``, builds a :class:`DinoReID` model.
    Otherwise falls through to the original PersonViT ``make_model``.
    """
    if cfg.MODEL.NAME == 'dino_v3':
        from .backbones.dino_reid import DinoReID

        model = DinoReID(
            num_classes=num_class,
            camera_num=camera_num,
            view_num=view_num,
            cfg=cfg,
        )
        print('===========building DINOv3 ReID model===========')
        return model

    # Fall through to original PersonViT / ResNet / Transformer logic
    return _original_make_model(cfg, num_class, camera_num, view_num)
