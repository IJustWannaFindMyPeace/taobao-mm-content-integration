"""Count-histogram SimTier adaptation, separate from MUSE learned log-count features."""
import torch
from torch import nn
from c0 import C0


def histogram(target,history,valid,target_available,history_available,bins=20):
    if bins<1:raise ValueError('Positive bin count required')
    target_norm=torch.linalg.vector_norm(target,dim=-1)
    history_norm=torch.linalg.vector_norm(history,dim=-1)
    mask=valid & history_available & target_available[:,None] & (history_norm>0) & (target_norm[:,None]>0)
    target=target/target_norm.clamp_min(1e-12)[:,None]
    history=history/history_norm.clamp_min(1e-12)[:,:,None]
    scores=(history*target[:,None,:]).sum(-1).clamp(-1,1)
    index=torch.floor((scores+1)*bins/2).long().clamp(0,bins-1)
    counts=torch.zeros((len(target),bins),device=target.device,dtype=torch.float32)
    counts.scatter_add_(1,index,mask.float())
    return counts


class C3(C0):
    def __init__(self,sizes,width=16,short_length=50,bins=20):
        super().__init__(sizes,width,short_length)
        self.histogram_projection=nn.Linear(bins,128,bias=False)
        nn.init.zeros_(self.histogram_projection.weight)

    def forward(self,features,item,category,valid,counts):
        combined=self.representation(features,item,category,valid)
        first=self.tower[0](combined)+self.histogram_projection(counts)
        return self.tower[1:](first).squeeze(-1)
