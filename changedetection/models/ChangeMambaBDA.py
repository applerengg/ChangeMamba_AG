import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from changedetection.models.Mamba_backbone import Backbone_VSSM
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from changedetection.models.ChangeDecoder import ChangeDecoder
from changedetection.models.SemanticDecoder import SemanticDecoder

from changedetection.models.alignment_module import AlignmentHead, AlignmentArgs
from changedetection.models.attn_gate import AttentionGateArgs

logger = logging.getLogger(__name__)
logger.info("ChangeMambaBDA.py")

class ChangeMambaBDA(nn.Module):
    def __init__(self, output_building, output_damage, pretrained, alignment_args: AlignmentArgs, attn_gate_args: AttentionGateArgs, **kwargs):
        logger.info("ChangeMambaBDA class")
        
        super(ChangeMambaBDA, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


       
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            enable_attention_gate=attn_gate_args.enable_damage_ag,
            **clean_kwargs
        )

        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            enable_attention_gate=attn_gate_args.enable_building_ag,
            **clean_kwargs
        )
      
        self.main_clf = nn.Conv2d(in_channels=128, out_channels=output_damage, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_building, kernel_size=1)

        # -------- ALIGNMENT ----------
        
        self.align_enable: bool = alignment_args.enabled
        self.align_stages: tuple[int, ...] = alignment_args.stages
        if self.align_enable:
            # self.encoder.dims: e.g., [C0, C1, C2, C3]
            self.align_heads = nn.ModuleDict()
            for s in self.align_stages:
                in_ch = int(self.encoder.dims[s])
                self.align_heads[f"s{s}"] = AlignmentHead(in_ch=in_ch, mid_ch=alignment_args.mid_ch)
        # a place to inspect flows without changing API
        self._last_alignment_flows: dict[int, torch.Tensor] = {}
        # ----------------------------------------

    @torch.no_grad()
    def get_last_alignment_flows(self) -> dict[int, torch.Tensor]:
        """helper to fetch flows for visualization without changing return signature"""
        return self._last_alignment_flows


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # ----- ALIGNMENT only for damage path -----
        if self.align_enable:
            # Copy list so building decoder sees *unaligned* pre_features
            pre_features_aligned = list(pre_features)
            self._last_alignment_flows = {}
            for s in self.align_stages:
                head: AlignmentHead = self.align_heads[f"s{s}"]
                fpre = pre_features_aligned[s]
                fpost = post_features[s]
                fpre_warp, flow = head(fpre, fpost) # shapes preserved
                pre_features_aligned[s] = fpre_warp
                self._last_alignment_flows[s] = flow # debug only
        else:
            pre_features_aligned = pre_features

        # Decoder processing - passing encoder outputs to the decoder
        output_building = self.decoder_building(pre_features)
        output_building = self.aux_clf(output_building)
        output_building = F.interpolate(output_building, size=pre_data.size()[-2:], mode='bilinear')
        
        output_damage = self.decoder_damage(pre_features_aligned, post_features)
        output_damage = self.main_clf(output_damage)
        output_damage = F.interpolate(output_damage, size=post_data.size()[-2:], mode='bilinear')

        return output_building, output_damage
