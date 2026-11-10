"""Additional controlled SimScore + frozen MAKE experiment."""
import torch
from torch import nn
from torch.nn import functional as F
from side_information import SideInformation
from feature_store import FIELDS
from make import ContentMAKE

class SimScoreMAKE(SideInformation):
    def __init__(self,sizes,width=16,short_length=50,freeze_content=True):
        super().__init__(sizes,width,short_length,mode='simscore')
        self.content_branch=ContentMAKE()
        self.freeze_content=freeze_content
        self.content_branch.output.requires_grad_(False)
        if freeze_content:self.content_branch.requires_grad_(False)
        self.content_projection=nn.Linear(160,128,bias=False)
        nn.init.zeros_(self.content_projection.weight)

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
        first=self.tower[0](torch.cat(scalar+[short,long],-1))
        args=(target_content,history_content,valid,target_available,history_available)
        if self.freeze_content:
            with torch.no_grad():interest,hidden=self.content_branch.extract(*args)
        else:interest,hidden=self.content_branch.extract(*args)
        first=first+self.content_projection(torch.cat([interest,hidden],-1))
        return self.tower[1:](first).squeeze(-1)
