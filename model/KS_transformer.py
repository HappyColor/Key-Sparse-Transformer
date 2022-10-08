"""
Created on Tue Sep 28 16:53:37 CST 2021
@author: lab-chen.weidong
"""
import torch
import torch.nn as nn
from model.transformer import build_transformer
from model.KS_transformer_block import build_ks_transformer
from model.CCAB_block import build_CCAB

class KS_transformer(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, num_classes):
        super().__init__()

        self.input_dim = input_dim

        self.audio_self_Trans = build_transformer(self_attn=True, num_layers=num_layers[0], embed_dim=input_dim[0], ffn_embed_dim=ffn_embed_dim[0], num_heads=num_heads)
        self.text_self_Trans = build_transformer(self_attn=True, num_layers=num_layers[0], embed_dim=input_dim[1], ffn_embed_dim=ffn_embed_dim[1], num_heads=num_heads)
        
        self.at_cross_Trans = build_CCAB(num_layers=num_layers[1], embed_dim=input_dim[0], kdim=input_dim[1], ffn_embed_dim=ffn_embed_dim[0], num_heads=num_heads)
        self.ta_cross_Trans = build_CCAB(num_layers=num_layers[1], embed_dim=input_dim[1], kdim=input_dim[0], ffn_embed_dim=ffn_embed_dim[1], num_heads=num_heads)
        
        self.last_audio_self_Trans = build_ks_transformer(self_attn=True, num_layers=num_layers[2], embed_dim=input_dim[0], ffn_embed_dim=ffn_embed_dim[0], num_heads=num_heads)
        self.last_text_self_Trans = build_ks_transformer(self_attn=True, num_layers=num_layers[2], embed_dim=input_dim[1], ffn_embed_dim=ffn_embed_dim[1], num_heads=num_heads)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        fc_dim = self.input_dim[0] + self.input_dim[1]
        self.classifier = nn.Sequential(
            nn.Linear(fc_dim, fc_dim//2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(fc_dim//2, fc_dim//4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(fc_dim//4, num_classes),
        )
    
    def forward(self, x_a: torch.Tensor, x_t: torch.Tensor, x_a_padding_mask, x_t_padding_mask):
        x_a = self.audio_self_Trans(x_a, key_padding_mask=x_a_padding_mask)
        x_t = self.text_self_Trans(x_t, key_padding_mask=x_t_padding_mask)

        x_at = x_a
        x_ta = x_t
        x_at = self.at_cross_Trans(x=x_at, k=x_t, key_padding_mask=x_t_padding_mask)
        x_ta = self.ta_cross_Trans(x=x_ta, k=x_a, key_padding_mask=x_a_padding_mask)
        
        x_a = x_a + x_at
        x_t = x_t + x_ta
        
        x_a = self.last_audio_self_Trans(x_a, key_padding_mask=None).transpose(1,2)
        x_t = self.last_text_self_Trans(x_t, key_padding_mask=None).transpose(1,2)

        x_a = self.avgpool(x_a).view(x_a.shape[0], -1)
        x_t = self.avgpool(x_t).view(x_t.shape[0], -1)

        x = torch.cat((x_a, x_t), dim=-1)
        x = self.classifier(x)
     
        return x

def build_ks_transformer(**kwargs):
    return KS_transformer(**kwargs)


