import torch
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer, checkpoint_filter_fn
from timm.models.deit import VisionTransformerDistilled, _create_deit, partial
from timm.models.helpers import build_model_with_cfg, checkpoint_seq, resolve_pretrained_cfg

from typing import List, Tuple, Iterator

###
### DeiT Models
###
class RegisteredVisionTransformerDistilled(VisionTransformerDistilled):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._reg_info = {
            'use_registers' : True,
            'num_registers' : 4,
        }
        ### Create registers
        self.register_tokens = torch.nn.Parameter(
            torch.zeros(1, self.num_register_tokens, self.embed_dim), requires_grad=True
        )
        self.register_toggle(True)

    def register_toggle(self, mode : bool):
        if mode == True:
            self._reg_info['use_registers'] = True
            self.num_prefix_tokens += self._reg_info['num_registers']
        else:
            self._reg_info['use_registers'] = False
            self.num_prefix_tokens -= self._reg_info['num_registers']

    def forward_features(self, x : torch.Tensor) -> torch.Tensor:
        ### Do patch embedding
        x = self.patch_embed(x)
        B, _, _ = x.size()
        ### Concatenate cls, dist
        x = torch.cat(
            (
                self.cls_token.expand(B, -1, -1),
                self.dist_token.expand(B, -1, -1),
                x,
            ),
            dim=1,
        )
        x = self.pos_drop(x + self.pos_embed)
        ### Concatenate registers
        if self.enable_registers:
            x = torch.cat(
                (
                    x[:, 0:(self.num_prefix_tokens - self.num_register_tokens), :],
                    self.register_tokens.expand(B, self.num_register_tokens, -1),
                    x[:, (self.num_prefix_tokens - self.num_register_tokens):, :],
                ),
                dim=1,
            )
        x = self.blocks(x)
        x = self.norm(x)
        return x
    
    def register_parameters(self) -> Iterator[torch.nn.Parameter]:
        for name, param in self.named_parameters():
            if name != 'register_tokens':
                pass
            else:
                yield param

###
### ViT Models / DeiT models without distillation
###
class RegisteredVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._reg_info = {
            'use_registers' : True,
            'num_registers' : 4,
        }
        ### Create registers
        self.register_tokens = torch.nn.Parameter(
            torch.zeros(1, self.num_register_tokens, self.embed_dim), requires_grad=True
        )
        self.register_toggle(True)
        
    def register_toggle(self, mode : bool):
        if mode == True:
            self._reg_info['use_registers'] = True
            self.num_prefix_tokens += self._reg_info['num_registers']
        else:
            self._reg_info['use_registers'] = False
            self.num_prefix_tokens -= self._reg_info['num_registers']

    def forward_features(self, x : torch.Tensor) -> torch.Tensor:
        ### Do patch embedding
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        ### Get batch size dim
        B, N, C = x.size()

        ### Concatenate registers
        if self.enable_registers:
            x = torch.cat(
                (
                    x[:, 0:(self.num_prefix_tokens - self.num_register_tokens), :],
                    self.register_tokens.expand(B, self.num_register_tokens, -1),
                    x[:, (self.num_prefix_tokens - self.num_register_tokens):, :],
                ),
                dim=1,
            )
        x = self.blocks(x)
        x = self.norm(x)
        return x
    
    def register_parameters(self) -> Iterator[torch.nn.Parameter]:
        for name, param in self.named_parameters():
            if name != 'register_tokens':
                pass
            else:
                yield param

###
### Model Configurations
###
def vit_small_register_patch16_224(pretrained=False, **kwargs) -> RegisteredVisionTransformer:
    """ViT-Small (ViT-S/16)"""
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model

def deit_small_register_patch16_224(pretrained=False, **kwargs) -> RegisteredVisionTransformerDistilled:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

def deit_small_distilled_register_patch16_224(pretrained=False, **kwargs) -> RegisteredVisionTransformerDistilled:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit('deit_small_distilled_patch16_224', distilled=True, pretrained=pretrained, **model_kwargs)
    return model


def vit_base_register_patch16_224(pretrained=False, **kwargs) -> RegisteredVisionTransformer:
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_registered_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model

def deit_base_register_patch16_224(pretrained=False, **kwargs) -> RegisteredVisionTransformerDistilled:
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_registered_deit('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

def deit_base_distilled_register_patch16_224(pretrained=False, **kwargs) -> RegisteredVisionTransformerDistilled:
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_registered_deit('deit_base_distilled_patch16_224', distilled=True, pretrained=pretrained, **model_kwargs)
    return model

###
### Builder Function
###
def _create_registered_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        RegisteredVisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=VisionTransformer.check,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        ### Important !
        pretrained_strict=False,
        **kwargs)
    return model

def _create_registered_deit(variant, pretrained=False, distilled=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model_cls = RegisteredVisionTransformerDistilled if distilled else RegisteredVisionTransformer
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        ### Important !
        pretrained_strict=False,
        **kwargs)
    return model