from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import quadprog  # type: ignore
except Exception:  # pragma: no cover
    quadprog = None


def _forward_logits_and_features(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model(x)
    if isinstance(out, (tuple, list)):
        logits = out[0]
        features = out[1] if len(out) > 1 else logits
        return logits, features
    return out, out


def _cat_grads(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.contiguous().view(-1) for g in grads], dim=0)


def project2cone2(
    gradient: torch.Tensor,
    memories: torch.Tensor,
    margin: float = 0.5,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Equivalent to upstream project2cone2 for a single-memory constraint."""
    grad_flat = gradient.detach().contiguous().view(-1)
    mem_flat = memories.detach().contiguous().view(-1)

    if quadprog is not None:
        memories_np = mem_flat.unsqueeze(0).cpu().double().numpy()
        gradient_np = grad_flat.cpu().double().numpy()
        t = memories_np.shape[0]
        p_mat = np.dot(memories_np, memories_np.transpose())
        p_mat = 0.5 * (p_mat + p_mat.transpose()) + np.eye(t) * eps
        q_vec = np.dot(memories_np, gradient_np) * -1
        g_mat = np.eye(t)
        h_vec = np.zeros(t) + margin
        v_sol = quadprog.solve_qp(p_mat, q_vec, g_mat, h_vec)[0]
        projected = np.dot(v_sol, memories_np) + gradient_np
        return torch.tensor(projected, dtype=grad_flat.dtype, device=grad_flat.device).view(-1)

    # Fallback when quadprog is unavailable (single constraint closed-form).
    dot = torch.dot(mem_flat, grad_flat)
    denom = torch.dot(mem_flat, mem_flat) + eps
    v_star = torch.clamp(-dot / denom, min=margin)
    return (v_star * mem_flat + grad_flat).view(-1)


class DCS2Defender:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        lambda_g: float = 0.7,
        lambda_x: float = 0.01,
        lambda_z: float = 0.01,
        epsilon: float = 0.01,
        dcs_iter: int = 1000,
        dcs_lr: float = 0.1,
        num_sen: int = 64,
        per_adv: int = 1,
        lambda_y: float | None = None,
        xsim_thr: float = 150.0,
        project: bool = True,
        mixup: bool = True,
        startpoint: str = "none",
        early_stop: bool = True,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.lambda_g = float(lambda_g)
        self.lambda_x = float(lambda_x)
        self.lambda_z = float(lambda_z)
        self.epsilon = float(epsilon)
        self.dcs_iter = int(dcs_iter)
        self.dcs_lr = float(dcs_lr)
        self.num_sen = int(num_sen)
        self.per_adv = int(per_adv)
        self.lambda_y = float(self.lambda_g if lambda_y is None else lambda_y)
        self.xsim_thr = float(xsim_thr)
        self.project = bool(project)
        self.mixup = bool(mixup)
        self.startpoint = str(startpoint)
        self.early_stop = bool(early_stop)

    def _defense_closure(
        self,
        params: List[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        sen_img: torch.Tensor,
        sen_out: torch.Tensor,
        sen_g: List[torch.Tensor],
        proxy_imgs: torch.Tensor,
        proxy_labels: torch.Tensor,
    ):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()

            proxy_logits, _ = _forward_logits_and_features(self.model, proxy_imgs)
            proxy_losses = self.criterion(proxy_logits, proxy_labels)
            proxy_g = torch.autograd.grad(proxy_losses, params, create_graph=True)

            rec_loss = torch.zeros((), device=proxy_imgs.device)
            pnorm0 = torch.zeros((), device=proxy_imgs.device)
            pnorm1 = torch.zeros((), device=proxy_imgs.device)
            for gx, gy in zip(sen_g, proxy_g):
                rec_loss += (gx * gy).sum()
                pnorm0 += gx.pow(2).sum()
                pnorm1 += gy.pow(2).sum()
            g_sim = 1.0 - rec_loss / (torch.sqrt(pnorm0) * torch.sqrt(pnorm1) + 1e-12)

            x_sim = torch.norm(
                proxy_imgs.reshape(proxy_imgs.size(0), -1) - sen_img.reshape(sen_img.size(0), -1),
                dim=1,
            ).mean()
            fx_sim = (torch.norm(proxy_logits - sen_out, dim=1) / (torch.norm(sen_out, dim=1) + 1e-12)).mean()

            total_loss = g_sim + torch.exp(-x_sim * self.lambda_x) + self.lambda_z * (fx_sim - self.epsilon)
            total_loss.backward(retain_graph=True)
            return total_loss, g_sim.detach(), x_sim.detach(), fx_sim.detach()

        return closure

    def defense_optim(
        self,
        gt_imgs: torch.Tensor,
        gt_labels: torch.Tensor,
        proxy_imgs: torch.Tensor,
        proxy_labels: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        self.model.eval()
        params = [p for p in self.model.parameters() if p.requires_grad]

        use_num_sen = min(self.num_sen, gt_imgs.size(0))
        total_proxy_needed = use_num_sen * self.per_adv
        if proxy_imgs.size(0) < total_proxy_needed:
            repeat_factor = (total_proxy_needed + proxy_imgs.size(0) - 1) // max(1, proxy_imgs.size(0))
            proxy_imgs = proxy_imgs.repeat(repeat_factor, 1, 1, 1)
            proxy_labels = proxy_labels.repeat(repeat_factor)
        proxy_imgs = proxy_imgs[:total_proxy_needed]
        proxy_labels = proxy_labels[:total_proxy_needed]

        gt_logits, _ = _forward_logits_and_features(self.model, gt_imgs)
        gt_losses = self.criterion(gt_logits, gt_labels)
        gt_gradients = torch.autograd.grad(gt_losses, params, retain_graph=True)
        ori_g = _cat_grads([grad.detach() for grad in gt_gradients])

        if gt_imgs.size(0) == use_num_sen:
            sen_g = [g for g in gt_gradients]
            sen_out = gt_logits
        else:
            sen_logits, _ = _forward_logits_and_features(self.model, gt_imgs[-use_num_sen:])
            sen_loss = self.criterion(sen_logits, gt_labels[-use_num_sen:])
            sen_g = torch.autograd.grad(sen_loss, params, retain_graph=True)
            sen_out = sen_logits

        sen_img = gt_imgs[-use_num_sen:]
        sen_label = gt_labels[-use_num_sen:]

        if self.mixup:
            for sk in range(use_num_sen):
                beg = sk * self.per_adv
                end = beg + self.per_adv
                proxy_imgs[beg:end] = self.lambda_y * proxy_imgs[beg:end] + (1.0 - self.lambda_y) * sen_img[sk].repeat(
                    self.per_adv, 1, 1, 1
                )

        my_sen_img = torch.repeat_interleave(sen_img, repeats=self.per_adv, dim=0)
        my_senout = torch.repeat_interleave(sen_out, repeats=self.per_adv, dim=0)
        proxy_imgs = proxy_imgs.requires_grad_(True)

        optimizer = torch.optim.Adam([proxy_imgs], lr=self.dcs_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[self.dcs_iter // 2.667, self.dcs_iter // 1.6, self.dcs_iter // 1.142],
            gamma=0.1,
        )

        last_total = torch.zeros((), device=gt_imgs.device)
        last_gsim = torch.zeros((), device=gt_imgs.device)
        last_xsim = torch.zeros((), device=gt_imgs.device)
        for _ in range(self.dcs_iter):
            closure = self._defense_closure(
                params=params,
                optimizer=optimizer,
                sen_img=my_sen_img,
                sen_out=my_senout,
                sen_g=sen_g,
                proxy_imgs=proxy_imgs,
                proxy_labels=proxy_labels,
            )
            total_loss, g_sim, x_sim, _ = optimizer.step(closure)
            last_total = total_loss.detach()
            last_gsim = g_sim.detach()
            last_xsim = x_sim.detach()
            if self.early_stop and float(last_xsim.item()) > self.xsim_thr:
                break
            scheduler.step()
            optimizer.zero_grad()

        adv_img = torch.cat([proxy_imgs, gt_imgs], dim=0)
        adv_logits, _ = _forward_logits_and_features(self.model, adv_img)

        proxy_logits = adv_logits[:total_proxy_needed]
        ori_logits = adv_logits[total_proxy_needed:]
        divd = 2.0 if (self.lambda_y == 0.0 or self.lambda_y == 1.0) else 3.0
        loss = (
            self.lambda_y * self.criterion(proxy_logits, proxy_labels)
            + (1.0 - self.lambda_y) * self.criterion(proxy_logits, sen_label.repeat_interleave(self.per_adv))
            + self.criterion(ori_logits, gt_labels)
        ) / divd

        adv_dydw = torch.autograd.grad(loss, params)
        adv_g = _cat_grads([grad.detach() for grad in adv_dydw])

        dotg = torch.dot(adv_g, ori_g)
        projected = False
        if self.project and dotg.item() < 0:
            new_grad = project2cone2(adv_g.unsqueeze(0), ori_g.unsqueeze(1))
            pointer = 0
            protected_grads: List[torch.Tensor] = []
            for p in params:
                num_param = p.numel()
                protected_grads.append(new_grad[pointer : pointer + num_param].view_as(p).to(gt_imgs.device))
                pointer += num_param
            projected = True
        else:
            protected_grads = [grad.detach().clone() for grad in adv_dydw]

        stats = {
            "conceal_obj": float(last_total.item()),
            "grad_cosine": float(1.0 - last_gsim.item()),
            "proj_applied_ratio": 1.0 if projected else 0.0,
        }
        return protected_grads, stats
