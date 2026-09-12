"""ID-only two-branch target-attention baseline; explicit local adaptation."""
import torch
from torch import nn
from feature_store import FIELDS


class TargetAttention(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.query=nn.Linear(width,width)
        self.key=nn.Linear(width,width)

    def forward(self,target,history,valid):
        scores=(self.query(target)[:,None,:]*self.key(history)).sum(-1)/history.shape[-1]**.5
        weights=scores.masked_fill(~valid,torch.finfo(scores.dtype).min).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        return (history*weights[:,:,None]).sum(1)


class C0(nn.Module):
    def __init__(self,sizes,width=16,short_length=50):
        super().__init__()
        self.short_length=short_length
        self.embeddings=nn.ModuleDict({f:nn.Embedding(sizes[f],width,padding_idx=0,sparse=True) for f in FIELDS})
        for layer in self.embeddings.values():
            nn.init.normal_(layer.weight,std=.01)
            with torch.no_grad():layer.weight[:2].zero_()
        self.short_attention=TargetAttention(width*2)
        self.long_attention=TargetAttention(width*2)
        self.tower=nn.Sequential(nn.Linear(width*(len(FIELDS)+4),128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))

    def representation(self,features,item,category,valid):
        scalar=[self.embeddings[f](features[:,i]) for i,f in enumerate(FIELDS)]
        target=torch.cat([scalar[6],scalar[7]],-1)
        history=torch.cat([self.embeddings['205'](item),self.embeddings['206'](category)],-1)
        short=self.short_attention(target,history[:,-self.short_length:],valid[:,-self.short_length:])
        long=self.long_attention(target,history,valid)
        return torch.cat(scalar+[short,long],-1)

    def forward(self,features,item,category,valid):
        return self.tower(self.representation(features,item,category,valid)).squeeze(-1)
