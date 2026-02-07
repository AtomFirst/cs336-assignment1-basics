import math
from typing import Callable, Iterable

import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[Tensor, 'batch_size vocab_size'],
    targets: Int[Tensor, 'batch_size']
) -> Float[Tensor, '']:
    inputs = inputs.reshape(-1, inputs.size(-1))
    targets = targets.reshape(-1)

    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs), dim=-1))

    siz = targets.size(0)
    return torch.mean(-inputs[torch.arange(siz), targets] + log_sum_exp)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                t = state.get('t', 1)

                m = state.get('m', torch.zeros_like(grad))
                v = state.get('v', torch.zeros_like(grad))

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        theta = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
        return min_learning_rate + 0.5 * (1 + math.cos(theta)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad.data for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    
    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2
    )

    clip_coeff = max_l2_norm / (total_norm + eps)
    
    scale = torch.clamp(clip_coeff, max=1.0)

    for g in grads:
        g.mul_(scale)
