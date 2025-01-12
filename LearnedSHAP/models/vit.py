# vit.py

import torch
import timm
from timm.models.vision_transformer import Attention

def get_vit_model(model_name='vit_base_patch16_224', pretrained=True, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = model.to(device)
    model.eval()
    
    return model

def get_vit_model_with_hooks(model_name='vit_base_patch16_224', pretrained=True, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    attention_maps = []

    def modified_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn_probs = attn.softmax(dim=-1)
        
        attention_maps.append(attn_probs.detach().cpu().numpy())
        
        attn = self.attn_drop(attn_probs)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.forward = modified_forward.__get__(module)

    model = model.to(device)
    model.eval()
    
    return model, attention_maps
