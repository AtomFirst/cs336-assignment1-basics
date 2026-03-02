import os
import math
import yaml

import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int, Bool
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(
            [out_features, in_features],
            **factory_kwargs
        ))

        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, 0, sigma ** 2, -3 * sigma, 3 * sigma)

    def extra_repr(self) -> str:
        return f'in_features = {self.in_features}, out_features = {self.out_features}'

    def forward(
        self,
        x: Float[Tensor, '... d_in']
    ) -> Float[Tensor, '... d_out']:
        return einsum(self.weight, x, 'd_out d_in, ... d_in -> ... d_out')
    

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(
            [num_embeddings, embedding_dim],
            **factory_kwargs
        ))

        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(
        self,
        x: Int[Tensor, '...']
    ) -> Float[Tensor, '... d_model']:
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(
            [d_model], **factory_kwargs
        ))

    def forward(
        self,
        x: Float[Tensor, '... d_model']
    ) -> Float[Tensor, '... d_model']:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rRMS = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        res = x * rRMS * self.weight
        return res.to(in_dtype)


def silu(x: Float[Tensor, '...']) -> Float[Tensor, '...']:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(
            d_model, d_ff, **factory_kwargs
        )

        self.w2 = Linear(
            d_ff, d_model, **factory_kwargs
        )

        self.w3 = Linear(
            d_model, d_ff, **factory_kwargs
        )

    def forward(
        self,
        x: Float[Tensor, '... d_model']
    ) -> Float[Tensor, '... d_model']:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        assert d_k % 2 == 0, 'd_k must be even'

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        i = torch.arange(max_seq_len, **factory_kwargs)
        k = torch.arange(0, d_k, 2, **factory_kwargs) / d_k
        theta_k = theta ** k
        angle = torch.outer(i, 1 / theta_k)
        self.register_buffer('cos_buffer', torch.cos(angle), persistent=False)
        self.register_buffer('sin_buffer', torch.sin(angle), persistent=False)

    def forward(
        self,
        in_query_or_key: Float[Tensor, '... sequence_length d_k'],
        token_positions: Int[Tensor, '... sequence_length'] | None = None,
    ) -> Float[Tensor, '... sequence_length d_k']:
        in_0 = in_query_or_key[..., 0::2]
        in_1 = in_query_or_key[..., 1::2]

        if token_positions is None:
            token_positions = torch.arange(in_query_or_key.size(-2))

        rotated = torch.stack([
            in_0 * self.cos_buffer[token_positions] - in_1 * self.sin_buffer[token_positions],
            in_0 * self.sin_buffer[token_positions] + in_1 * self.cos_buffer[token_positions]
        ], dim=-1).flatten(-2)

        return rotated


def softmax(
    x: Float[Tensor, " ..."],
    dim: int
) -> Float[Tensor, " ..."]:
    y = x - torch.max(x, dim=dim, keepdim=True).values
    z = torch.exp(y)
    return z / z.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    QK = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys') / math.sqrt(Q.size(-1))
    if mask is not None:
        QK = QK + torch.where(mask, 0.0, -torch.inf)
    
    return einsum(softmax(QK, -1), V, '... queries keys, ... keys d_v -> ... queries d_v')


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        assert d_model % num_heads == 0, 'd_model must be multiples of num_heads'

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

    def forward(
        self,
        x: Float[Tensor, '... sequence_length d_in'],
    ) -> Float[Tensor, '... sequence_length d_out']:
        q = rearrange(self.q_proj(x), '... seq (head dim) -> ... head seq dim', head=self.num_heads)
        k = rearrange(self.k_proj(x), '... seq (head dim) -> ... head seq dim', head=self.num_heads)
        v = rearrange(self.v_proj(x), '... seq (head dim) -> ... head seq dim', head=self.num_heads)

        seq = x.size(-2)
        heads = scaled_dot_product_attention(q, k, v, torch.tril(torch.ones((seq, seq), dtype=torch.bool)))
        o = rearrange(heads, '... head seq dim -> ... seq (head dim)')

        return self.output_proj(o)


class MultiheadSelfAttentionWithRope(MultiheadSelfAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        assert d_model % num_heads == 0, 'd_model must be multiples of num_heads'
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(d_model, num_heads, **factory_kwargs)

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, **factory_kwargs)
        self.mask = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device))

    def forward(
        self,
        x: Float[Tensor, '... sequence_length d_in'],
        token_positions: Int[Tensor, '... sequence_length'] | None = None
    ) -> Float[Tensor, '... sequence_length d_out']:
        q = rearrange(self.q_proj(x), '... seq (head dim) -> ... head seq dim', head=self.num_heads)
        k = rearrange(self.k_proj(x), '... seq (head dim) -> ... head seq dim', head=self.num_heads)
        v = rearrange(self.v_proj(x), '... seq (head dim) -> ... head seq dim', head=self.num_heads)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        seq = x.size(-2)
        heads = scaled_dot_product_attention(q, k, v, self.mask[:seq, :seq])
        o = rearrange(heads, '... head seq dim -> ... seq (head dim)')

        return self.output_proj(o)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        assert d_model % num_heads == 0, 'd_model must be multiples of num_heads'
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.attn = MultiheadSelfAttentionWithRope(d_model, num_heads, max_seq_len, theta, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)
        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, **factory_kwargs)

    def forward(
        self,
        x: Float[Tensor, 'batch sequence_length d_model']
    ) -> Float[Tensor, 'batch sequence_length d_model']:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 1e4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        assert d_model % num_heads == 0, 'd_model must be multiples of num_heads'
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, **factory_kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    @classmethod
    def from_files(
        cls,
        config_filepath: str | os.PathLike
    ) -> 'TransformerLM':
        with open(config_filepath, 'r') as f:
            config: dict = yaml.safe_load(f)

        model = cls(**config['model'], device=config['device'])

        checkpoint_path = config['checkpoint']['path']
        if checkpoint_path is not None:
            status = torch.load(checkpoint_path, map_location=config['device'])
            model.load_state_dict(status['model'])

        return model

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.ln_final(x))
