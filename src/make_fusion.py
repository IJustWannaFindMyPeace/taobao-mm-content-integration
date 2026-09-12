"""Factorial MAKE integration: initialization and freezing are independent."""
import torch
from torch import nn
from c0 import C0
from make import ContentMAKE


class MAKEFusion(C0):
    def __init__(self, sizes, width=16, short_length=50, freeze_content=False):
        super().__init__(sizes, width, short_length)
        self.content_branch = ContentMAKE()
        # The pretraining logit head is not part of the transferred representation.
        self.content_branch.output.requires_grad_(False)
        self.freeze_content = freeze_content
        if freeze_content:
            self.content_branch.requires_grad_(False)
        self.content_projection = nn.Linear(128+32, 128, bias=False)
        nn.init.zeros_(self.content_projection.weight)

    def forward(self, features, item, category, valid, target_content,
                history_content, target_available, history_available):
        arguments=(target_content, history_content, valid, target_available, history_available)
        if self.freeze_content:
            with torch.no_grad(): interest, hidden = self.content_branch.extract(*arguments)
        else:
            interest, hidden = self.content_branch.extract(*arguments)
        first=self.tower[0](self.representation(features,item,category,valid))
        first=first+self.content_projection(torch.cat([interest,hidden],-1))
        return self.tower[1:](first).squeeze(-1)
