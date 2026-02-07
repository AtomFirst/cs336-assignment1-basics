import math
from typing import Callable

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
                beta1_t = state.get('beta1_t', beta1)
                beta2_t = state.get('beta2_t', beta2)

                m = state.get('m', torch.zeros_like(grad))
                v = state.get('v', torch.zeros_like(grad))

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                lr_t = lr * math.sqrt(1 - beta2_t) / (1 - beta1_t)

                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state['t'] = t + 1
                state['beta1_t'] = beta1_t * beta1
                state['beta2_t'] = beta2_t * beta2
                state['m'] = m
                state['v'] = v

        return loss
