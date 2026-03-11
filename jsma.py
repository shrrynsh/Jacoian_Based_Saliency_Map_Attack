from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def compute_jacobian(model: nn.Module, x: torch.Tensor, use_logits: bool = True) -> torch.Tensor:
    """Compute Jacobian dF_j/dx for a single input sample."""
    x = x.clone().detach().requires_grad_(True)

    outputs = model.logits(x) if use_logits else model(x)
    num_classes = outputs.shape[1]
    num_features = x[0].numel()

    jacobian = torch.zeros(num_classes, num_features, device=x.device)
    for class_idx in range(num_classes):
        grads = torch.autograd.grad(
            outputs=outputs[0, class_idx],
            inputs=x,
            retain_graph=(class_idx < num_classes - 1),
            create_graph=False,
            allow_unused=False,
        )[0]
        jacobian[class_idx] = grads.view(-1).detach()

    return jacobian


def _saliency_pair(
    jacobian: torch.Tensor,
    target: int,
    search_mask: torch.Tensor,
    increase: bool,
) -> Tuple[int, int]:
    """Pick the best pixel pair according to JSMA saliency criterion."""
    target_grad = jacobian[target]
    other_grad = jacobian.sum(dim=0) - target_grad

    domain = torch.nonzero(search_mask, as_tuple=False).view(-1)
    n = int(domain.numel())
    if n < 2:
        return -1, -1

    t = target_grad.index_select(0, domain)
    o = other_grad.index_select(0, domain)

    alpha = t[:, None] + t[None, :]
    beta = o[:, None] + o[None, :]

    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=jacobian.device), diagonal=1)
    if increase:
        valid = (alpha > 0) & (beta < 0) & upper
        scores = torch.where(valid, alpha * (-beta), torch.full_like(alpha, -1.0))
    else:
        valid = (alpha < 0) & (beta > 0) & upper
        scores = torch.where(valid, (-alpha) * beta, torch.full_like(alpha, -1.0))

    flat_scores = scores.view(-1)
    best_flat = int(torch.argmax(flat_scores).item())
    if float(flat_scores[best_flat].item()) < 0.0:
        return -1, -1

    i = best_flat // n
    j = best_flat % n
    return int(domain[i].item()), int(domain[j].item())


def jsma_attack(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    theta: float,
    max_distortion: float = 0.145,
    increase: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float | int | bool]]:
    """Craft one adversarial example with JSMA for a given target class."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = x.to(device)
    x_adv = x.clone()

    num_features = x[0].numel()
    max_iter = int(num_features * max_distortion / 2.0)

    with torch.no_grad():
        source_class = int(model.logits(x).argmax(dim=1).item())

    if source_class == target_class:
        return x_adv, {
            "success": False,
            "n_iter": 0,
            "distortion": 0.0,
            "source_class": source_class,
            "final_pred": source_class,
        }

    search_mask = torch.ones(num_features, dtype=torch.bool, device=device)
    n_iter = 0
    current_pred = source_class

    while current_pred != target_class and n_iter < max_iter and int(search_mask.sum().item()) >= 2:
        jacobian = compute_jacobian(model, x_adv, use_logits=True)
        p1, p2 = _saliency_pair(jacobian, target_class, search_mask, increase)

        if p1 == -1:
            if verbose:
                print(f"  [iter {n_iter:3d}] no valid pixel pair found")
            break

        x_adv_flat = x_adv.view(-1)
        delta = theta if increase else -theta
        x_adv_flat[p1] = torch.clamp(x_adv_flat[p1] + delta, clip_min, clip_max)
        x_adv_flat[p2] = torch.clamp(x_adv_flat[p2] + delta, clip_min, clip_max)
        x_adv = x_adv_flat.view_as(x_adv)

        if x_adv_flat[p1].item() <= clip_min or x_adv_flat[p1].item() >= clip_max:
            search_mask[p1] = False
        if x_adv_flat[p2].item() <= clip_min or x_adv_flat[p2].item() >= clip_max:
            search_mask[p2] = False

        n_iter += 1
        with torch.no_grad():
            current_pred = int(model.logits(x_adv).argmax(dim=1).item())

        if verbose:
            print(f"  [iter {n_iter:3d}] pred={current_pred}, target={target_class}")

    delta_pixels = (x_adv - x).view(-1)
    n_modified = int((delta_pixels.abs() > 1e-6).sum().item())
    distortion = float(n_modified / num_features)

    return x_adv, {
        "success": current_pred == target_class,
        "n_iter": n_iter,
        "distortion": distortion,
        "source_class": source_class,
        "final_pred": current_pred,
    }


class JSMAAttack:
    def __init__(
        self,
        model: nn.Module,
        theta: float = 1.0,
        max_distortion: float = 0.145,
        increase: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.theta = theta
        self.max_distortion = max_distortion
        self.increase = increase
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device

    def craft(
        self,
        x: torch.Tensor,
        target_class: int,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float | int | bool]]:
        return jsma_attack(
            model=self.model,
            x=x,
            target_class=target_class,
            theta=self.theta,
            max_distortion=self.max_distortion,
            increase=self.increase,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            device=self.device,
            verbose=verbose,
        )





        








