import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int


def softmax(
    x: Float[Tensor, " ..."],
    dim: int,
    temperature: float
) -> Float[Tensor, " ..."]:
    y = x - torch.max(x, dim=dim, keepdim=True).values
    y = y / temperature
    z = torch.exp(y)
    return z / z.sum(dim=dim, keepdim=True)


def top_p_sampling(
    logits: Float[Tensor, " ..."],
    dim: int,
    temperature: float,
    p: float
) -> Float[Tensor, " ..."]:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim, temperature), dim=dim)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=dim, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')

    probs = softmax(logits, dim, temperature)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def generate(
    model: nn.Module,
    prompt: Int[Tensor, " batch_size sequence_length"],
    end_token: int,
    max_generated_length: int,
    temperature: float,
    p: float
) -> Int[Tensor, " batch_size new_sequence_length"]:
    generated = prompt.clone()
    
    for _ in range(max_generated_length):
        logits = model(generated)[:, -1, :]
        next_token = top_p_sampling(logits, -1, temperature, p)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == end_token:
            break

    return generated
