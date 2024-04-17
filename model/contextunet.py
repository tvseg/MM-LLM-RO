
# Modified by Yujin Oh, Mar-18-2024

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep
from transformers import LlamaTokenizer

from .modules import ContextUnetrUpBlock, UnetOutUpBlock
from .sam import TwoWayTransformer
from .text_encoder import tokenize, TextContextEncoder
from .llama2.llama_custom import LlamaForCausalLM


# LLMSeg
class ContextUNETR(nn.Module):
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        context=False,
        args=None,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")
        
        self.context = context
        self.normalize = normalize

        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = in_channels,
            out_channels = feature_size,
            kernel_size =3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = feature_size,
            out_channels = feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = feature_size,
            out_channels = 2 * feature_size ,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels= 2 * feature_size,
            out_channels = 4 * feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder10 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = 4 * feature_size,
            out_channels = 8 * feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        # decoder
        self.decoder4 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
            in_channels = feature_size * 8 ,
            out_channels = feature_size * 4,
            kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name, res_block=True,
            add_channels = (args.n_prompts if args.align_score else 0 if self.context else 0))

        self.decoder3 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
            in_channels = feature_size * 4 - (args.n_prompts if args.align_score else 0 if self.context else 0),
            out_channels = feature_size * 2,
            kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name,
            add_channels = (args.n_prompts if args.align_score else 0 if self.context else 0))
        
        self.decoder2 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
            in_channels = feature_size * 2 - (args.n_prompts if args.align_score else 0 if self.context else 0),
            out_channels = feature_size,
            kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name,
            add_channels = (args.n_prompts if args.align_score else 0 if self.context else 0))

        self.decoder1 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
            in_channels = feature_size - (args.n_prompts if args.align_score else 0 if self.context else 0),
            out_channels = feature_size, 
            kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name,
            add_channels =(args.n_prompts if args.align_score else 0 if self.context else 0))

        # out
        self.out = UnetOutUpBlock(spatial_dims=spatial_dims, 
            in_channels=feature_size, 
            out_channels=out_channels, 
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
            
        feature_size_list = [self.encoder1.layer.conv3.out_channels, self.encoder2.layer.conv3.out_channels, self.encoder3.layer.conv3.out_channels, self.encoder4.layer.conv3.out_channels, self.encoder10.layer.conv3.out_channels]

        # multiomdal text encoder
        if self.context:

            self.align_score = args.align_score
           
            if args.textencoder == 'llama2':
                self.txt_embed_dim = 4096 
            elif args.textencoder == 'llama2_13b':
                self.txt_embed_dim = 5120 
            else:
                self.txt_embed_dim = 512

            # interactive align module 
            txt2vis, attntrans = [], []
            for i in range(len(depths)+1):
                txt2vis.append(nn.Linear(self.txt_embed_dim, feature_size_list[i]))
                attntrans.append(TwoWayTransformer(depth=2,
                                                embedding_dim=feature_size_list[i],
                                                mlp_dim=feature_size*(2**i),
                                                num_heads=8,
                                                ))
            self.txt2vis = nn.Sequential(*txt2vis)
            self.attn_transformer = nn.Sequential(*attntrans)
            
            # clip text encoder
            self.text_encoder = TextContextEncoder(embed_dim=self.txt_embed_dim)
            self.context_length = args.context_length
            self.token_embed_dim = self.text_encoder.text_projection.shape[-1]
            self.contexts = nn.Parameter(torch.randn(args.n_prompts, self.context_length, self.token_embed_dim))
            self.max_length = 77
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad_(False)
            
            # llama2
            if args.textencoder.find('llama') >= 0:
                self.text_encoder.llm = True
                if args.textencoder == 'llama2':
                    rep_llama = args.llama_rep
                self.tokenizer = LlamaTokenizer.from_pretrained(rep_llama)
                self.max_length = 128 

                self.text_encoder.transformer  = LlamaForCausalLM.from_pretrained(
                        rep_llama,
                        # load_in_8bit=True, # Add this for using int8
                        torch_dtype=torch.float16,
                        device_map="cpu", #args.gpu "cpu"
                    ).model
                    
                self.tokenizer._add_tokens(["<SEG>"], special_tokens=True)
                self.text_encoder.transformer.resize_token_embeddings(len(self.tokenizer) + 1)
                self.text_encoder.token_embedding = self.text_encoder.transformer.embed_tokens
                
                for name, param in self.text_encoder.transformer.named_parameters():
                    param.requires_grad_(False)
                    
        
    def load_from(self, weights):
        pass

    def interactive_alignment(self, hidden_states_out, report_in, x_in):
        
        tok_txt = []
        emb_txt = []
        emb_txt_t = []
            
        # prepare text tokens
        tok_txt = report_in
        emb_txt = self.text_encoder(tok_txt.to(x_in.device), self.contexts) 

        # projection
        report_l = []
        for i in self.txt2vis._modules.keys():
            report_l.append(self.txt2vis._modules[i](emb_txt))        

        # interactive alignment
        h_offset = 0
        for j, text_vis in enumerate(zip(report_l[h_offset:], hidden_states_out[h_offset:])):
            
            txt, vis = text_vis

            if len(report_in) != len(x_in):
                txt = torch.repeat_interleave(txt, vis.shape[0], dim=0)
            
            _, hidden_states_out[j+h_offset] = self.attn_transformer[j+h_offset](vis, None, txt)

        return hidden_states_out, emb_txt, emb_txt_t
    
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
    def forward(self, x_in, report_in=None, target=None):
        
        hidden_states_out = []

        # visual encoder
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(enc0)
        enc2 = self.encoder3(enc1)
        enc3 = self.encoder4(enc2)
        dec4 = self.encoder10(enc3)
    
        hidden_states_out.append(enc0)
        hidden_states_out.append(enc1)
        hidden_states_out.append(enc2)
        hidden_states_out.append(enc3)
        hidden_states_out.append(dec4)

        # multimodal alignment
        hidden_states_out, _, _ = self.interactive_alignment(hidden_states_out, report_in, x_in)

        # visual decoder
        dec2 = self.decoder4(hidden_states_out[4], hidden_states_out[3])
        dec1 = self.decoder3(dec2, hidden_states_out[2])
        dec0 = self.decoder2(dec1, hidden_states_out[1])
        out = self.decoder1(dec0, hidden_states_out[0])

        logits = self.out(out)
 
        return logits
      