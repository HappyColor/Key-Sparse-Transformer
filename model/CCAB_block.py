"""
Created on Thu Dec 16 14:39:52 CST 2021
@author: lab-chen.weidong
"""
import torch.nn as nn

from model.KS_transformer_block import build_ks_transformer_block

class single_CCAB(nn.Module):
    def __init__(self, embed_dim, kdim, ffn_embed_dim, num_heads):
        super().__init__()
        self.ks_trans_1 = build_ks_transformer_block(self_attn=False, num_layers=1, embed_dim=embed_dim, kdim=kdim, ffn_embed_dim=ffn_embed_dim, num_heads=num_heads)
        self.ks_trais_2 = build_ks_transformer_block(self_attn=True, num_layers=1, embed_dim=embed_dim, ffn_embed_dim=ffn_embed_dim, num_heads=num_heads)
    
    def forward(self, x, k, key_padding_mask):
        x = self.ks_trans_1(query=x, key=k, key_padding_mask=key_padding_mask)
        x = self.ks_trais_2(query=x, key_padding_mask=None)
        return x
    
class CCABs(nn.Module):
    def __init__(self, embed_dim, kdim, ffn_embed_dim, num_layers, num_heads):
        super().__init__()
        self.CCABs_model = nn.ModuleList([single_CCAB(embed_dim, kdim, ffn_embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, k, key_padding_mask):
        residual = x
        for layer in self.CCABs_model:
            x = layer(x, k, key_padding_mask)
        x = x + residual
        return x

def build_CCAB(embed_dim, kdim, ffn_embed_dim, num_layers, num_heads):
    '''
    forward: x, k, key_padding_mask
    '''
    return CCABs(embed_dim, kdim, ffn_embed_dim, num_layers, num_heads)

