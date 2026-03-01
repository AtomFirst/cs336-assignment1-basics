import yaml
import types
import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import wandb

from implements.transformer import RMSNorm, TransformerBlock, RotaryPositionalEmbedding, SwiGLU, Linear, Float, Tensor, silu
from trainer import Trainer, WANDB_CONFIG


@dataclass
class AblationConfig:
    title: str
    target: type[nn.Module]
    action_fn: Callable


def search_and_modify(model: nn.Module, target_module: type[nn.Module], action_fn) -> None:
    for name, module in model.named_modules():
        if isinstance(module, target_module):
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent, module_name = model, name
            
            action_fn(module, parent, module_name)


def ablation(config, ablation_config: AblationConfig):
    run = wandb.init(
        **WANDB_CONFIG,
        config=config,
        name=ablation_config.title
    )

    trainer = Trainer(config)
    search_and_modify(
        trainer.model,
        ablation_config.target,
        ablation_config.action_fn
    )
    
    print(ablation_config.title,':')
    print(trainer.model)
    print()

    trainer.train(run)
    run.finish()


def post_norm_transformer_block_forward(self, x):
    x = self.ln1(x + self.attn(x))
    return self.ln2(x + self.ffn(x))


def nope_forward(self, in_query_or_key, token_positions):
    return in_query_or_key


class SiLU(nn.Module):
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

    def forward(
        self,
        x: Float[Tensor, '... d_model']
    ) -> Float[Tensor, '... d_model']:
        return self.w2(silu(self.w1(x)))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    base_config['checkpoint']['path'] = None

    _ablation_configs = [
        # (
        #     'layer_norm_ablation',
        #     RMSNorm,
        #     lambda module, parent, module_name: setattr(parent, module_name, nn.Identity())
        # ),
        (
            'pre_norm_ablation',
            TransformerBlock,
            lambda module, parent, module_name: setattr(module, 'forward', types.MethodType(post_norm_transformer_block_forward, module))
        ),
        (
            'no_pos_emb',
            RotaryPositionalEmbedding,
            lambda module, parent, module_name: setattr(module, 'forward', types.MethodType(nope_forward, module))
        ),
        (
            'swiglu_ablation',
            SwiGLU,
            lambda module, parent, module_name: setattr(parent, module_name, SiLU(module.d_model, module.d_model * 4, next(module.parameters()).device))
        )
    ]

    ablation_configs = [AblationConfig(title, target, action_fn) for title, target, action_fn in _ablation_configs]

    for ablation_config in ablation_configs:
        ablation(base_config, ablation_config)


if __name__ == '__main__':
    main()
