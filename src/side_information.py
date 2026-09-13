"""Item-level Vector and behavior-level SimScore with shared C0 backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from c0 import C0
from feature_store import FIELDS


class SideInformation(C0):
    def __init__(self,sizes,width=16,short_length=50,mode='vector'):
        if mode not in ['vector','simscore']:raise ValueError('Unknown side-information mode')
        super().__init__(sizes,width,short_length)
        self.mode=mode
        # [ID, side] -> ID + W(side) is a concatenation projection whose ID
        # block is identity. Zero initialization preserves the C0 prediction.
        self.side_projection=nn.Linear(128 if mode=='vector' else 2,width*2,bias=False)
        nn.init.zeros_(self.side_projection.weight)

    def forward(self,features,item,category,valid,target_content,history_content,
                target_available,history_available):
        scalar=[self.embeddings[f](features[:,i]) for i,f in enumerate(FIELDS)]
        target=torch.cat([scalar[6],scalar[7]],-1)
        history=torch.cat([self.embeddings['205'](item),self.embeddings['206'](category)],-1)
        q=F.normalize(target_content.detach(),dim=-1)*target_available[:,None]
        h=F.normalize(history_content.detach(),dim=-1)*(history_available & valid)[...,None]
        if self.mode=='vector':
            target=target+self.side_projection(q)
            history=history+self.side_projection(h)
        else:
            available=valid & history_available & target_available[:,None]
            available=available & (target_content.norm(dim=-1)>0)[:,None] & (history_content.norm(dim=-1)>0)
            similarity=(q[:,None,:]*h).sum(-1).clamp(-1,1)*available
            side=torch.stack([similarity,available.to(similarity.dtype)],-1)
            history=history+self.side_projection(side)
        short=self.short_attention(target,history[:,-self.short_length:],valid[:,-self.short_length:])
        long=self.long_attention(target,history,valid)
        return self.tower(torch.cat(scalar+[short,long],-1)).squeeze(-1)
