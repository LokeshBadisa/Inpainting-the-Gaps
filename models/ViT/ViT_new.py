""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from helpers import load_pretrained
from weight_init import trunc_normal_
from layer_helpers import to_2tuple


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    # https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth
    'vit_small_patch16_224': _cfg(
        url='vit_small_p16_224-15ec54c9.pth',
    ),
    # https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
    'vit_base_patch16_224': _cfg(
        # url='./weights/vit_funnybirds.pth.tar',
        url='./weights/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    # https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar
      'vit_base_patch16_224_moco': _cfg(
        url='./weights/linear-vit-b-300ep.pth.tar',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    # https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth
    # https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
    # https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth
      'vit_base_patch16_224_dino': _cfg(
        url_backbone='./weights/dino_vitbase16_pretrain.pth',
        url_linear='./weights/dino_vitbase16_linearweights.pth',
        url='./weights/dino_vitbase16_pretrain_full_checkpoint.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    # https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth
    'vit_mae_patch16_224': _cfg(
        url='./weights/mae_finetuned_vit_base.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    # https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth
    'vit_large_patch16_224': _cfg(
        url='./weights/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        self.attention_map = None
        
        self.vproj = None
        self.input = None
        self.v = None

        ##Addition By Lokesh
        self.q = None
        self.k = None
        self.out = None
        ##End of Addition By Lokesh
    
    def save_vproj(self, vproj):
        self.vproj = vproj
    
    def get_vproj(self):
        return self.vproj
    
    ##Addition By Lokesh

    def save_q(self, q):
        self.q = q

    def get_q(self):
        return self.q
    
    def save_k(self, k):
        self.k = k

    def get_k(self):
        return self.k
    
    def save_attwt(self, attwt):
        self.attwt = attwt

    def get_attwt(self):
        return self.attwt
    
    def save_out(self, out):
        self.out = out

    def get_out(self):
        return self.out

    ##End of Addition By Lokesh
    
    def save_v(self, v):
        self.v = v
    
    def get_v(self):
        return self.v
    
    def save_input(self, z):
        self.input = z
    
    def get_input(self):
        return self.input

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map
    

    def forward(self, x, register_hook=False):
        b, n, _, h = *x.shape, self.num_heads
        self.save_input(x)
        

        # self.save_output(x)
        # x.register_hook(self.save_output_grad)

        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        ##Addition By Lokesh
        # self.save_q(q)
        # self.save_k(k)
        ##End of Addition By Lokesh
        self.save_v(v)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        self.save_attention_map(attn)
        if register_hook:
            ## Your Notes:
            ## register_hook:
            ## self.save_attn_gradients(saves the gradient) is executed
            ## when a backward pass is called
            attn.register_hook(self.save_attn_gradients)

        out = rearrange(out, 'b h n d -> b n (h d)')
        
#         z = torch.matmul(out , self.proj.weight.t())
#         self.save_out(z)
     
        out = self.proj(out)
        out = self.proj_drop(out)
        
        ##Addition By Lokesh
        # self.save_out(out)
        ##End of Addition By Lokesh
           
        vproj = torch.matmul(rearrange(v, 'b h n d -> b n (h d)') , self.proj.weight.t())
        
        
        self.save_vproj(vproj)
        
        return out


class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.input = None
        self.output = None
        self.tilde = None
        
    def save_input(self, z):
        self.input = z
        
    def get_input(self):
        return self.input
    
    def save_tilde(self, z):
        self.tilde = z
        
    def get_tilde(self):
        return self.tilde
    
    def save_output(self, z):
        self.output = z
        
    def get_output(self):
        return self.output

    def forward(self, x, register_hook=False):
        
        self.save_input(x)
        out = x + self.attn(self.norm1(x), register_hook=register_hook)
        # self.save_tilde(out)
        out = out + self.mlp(self.norm2(out))
        # self.save_output(out)
        
        return out


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm, global_pooling=False, dino=False):
        super().__init__()
        self.global_pooling = global_pooling
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        if global_pooling:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        # Classifier head
        if dino:
            self.head = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.cls_gradients=None
        self.input_grad=None
        self.dino=dino
        self.hs = []
        self.hs_grads = []
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.device = torch.device('cuda:2')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def save_cls_gradients(self, cls_gradients):
        self.cls_gradients = cls_gradients
    
    def get_cls_gradients(self):
        return self.cls_gradients
    
    def save_input_gradients(self, input_grad):
        self.input_grad = input_grad
    
    def get_input_gradients(self):
        return self.input_grad
    
    def save_hs_gradients(self, hs_grads):
        self.hs_grads.append(hs_grads)
        
    def forward(self, x, register_hook=False,save_h=False):
        B = x.shape[0]
#         if register_hook:
#             x.register_hook(self.save_input_gradients)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks[:-1]):
            x = blk(x, register_hook=register_hook)
            if save_h:
                self.hs.append(x)
                x.register_hook(self.save_hs_gradients)
        
        # x.register_hook(self.save_cls_gradients)
        
        x = self.blocks[-1](x, register_hook=register_hook)
        if save_h:
            self.hs.append(x)
            x.register_hook(self.save_hs_gradients)

        if self.global_pooling:
            x = self.fc_norm(x)
            x = x[:, 1:, :].mean(axis=1) 
        elif self.dino:
            x = self.norm(x)
            x = torch.cat((x[:, 0].unsqueeze(-1), x[:, 1:, :].mean(axis=1).unsqueeze(-1)), dim=-1)
            x = x.reshape(x.shape[0], -1)
        else:
            x = self.norm(x)
            x = x[:, 0]
                
        x = self.head(x)
        return x
    
    def continuation(self,x):
        with torch.no_grad():
            x = torch.einsum('bhij,bhjd->bhid', x, self.blocks[-1].attn.get_v())
            x = rearrange(x, 'b h n d -> b n (h d)')     
            x = self.blocks[-1].attn.proj(x)
            x = self.blocks[-1].get_input() + x
            x = x + self.blocks[-1].mlp(self.blocks[-1].norm2(x))
            if self.global_pooling:
                x = x[:, 1:, :].mean(axis=1) 
                x = self.fc_norm(x)
            elif self.dino:
                x = self.norm(x)
                x = torch.cat((x[:, 0].unsqueeze(-1), x[:, 1:, :].mean(axis=1).unsqueeze(-1)), dim=-1)
                x = x.reshape(x.shape[0], -1)
            else:
                x = self.norm(x)
                x = x[:, 0]
                    
            x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=8, num_heads=6, mlp_ratio=3, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

def vit_base_patch16_224_dino(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), dino=True, **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224_dino']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, dino=True)
    return model

def vit_base_patch16_224_moco(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224_moco']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, moco=True)
    return model

def vit_mae_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pooling=True, **kwargs)
    model.default_cfg = default_cfgs['vit_mae_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, mae=True)
    return model

def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
