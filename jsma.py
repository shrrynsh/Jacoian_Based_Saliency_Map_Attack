from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Optional ,Tuple


def compute_jacobian(
    model : nn.Module,
    x: torch.Tensor,
    use_logits : bool=True,
) -> torch.Tensor:

    x=x.clone().detach().requires_grad_(True)

    if use_logits:
        outputs=model.logits(x)
    else:
        outputs=model(x)

    
    num_classes=outputs.shape[1]
    num_features=x.numel()//x.shape[0]

    jacobian=torch.zeros(num_classes,num_features)

    for j in range(num_classes):
        if x.grad is not None:
            x.grad.zero_()
        outputs[0,j].backward(retain_graph=(j<num_classes-1))
        jacobian[j]=x.grad.view(-1).clone
    
    return jacobian





