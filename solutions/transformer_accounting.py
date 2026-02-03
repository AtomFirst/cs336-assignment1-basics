import math
from pprint import pprint

import torch
import torch.nn as nn
from torch import Tensor

from implements import transformer


def build_transformer(args: dict[str, int]):
    with torch.device('meta'):
        return transformer.TransformerLM(**args)


def build_input(args: dict[str, int]):
    return torch.empty((1, args['context_length']), dtype=torch.int, device='meta')


def calc_paras(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_FLOPS(
    model: nn.Module,
    input: Tensor
) -> tuple[dict[tuple[str], tuple[int]], int]:
    flops = {}
    total_flops = 0
    hooks = []

    def make_hook_fn(layers: str, module: str):
        def hook_fn_linear(model: transformer.Linear, input: Tensor, output):
            nonlocal flops, total_flops
            n = model.in_features
            m = input[0].numel() // n
            p = model.out_features

            flops[layers] = m, n, p
            total_flops += 2 * m * n * p

        def hook_fn_attn(model: transformer.MultiheadSelfAttentionWithRope, input: tuple[Tensor, Tensor], output):
            nonlocal flops, total_flops
            x = input[0]
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            
            h = model.num_heads
            d_model = model.d_model
            d_head = d_model // h
            
            m = seq_len
            k = d_head
            n = seq_len
            
            current_qk_flops = 2 * m * n * k * h * batch_size
            
            flops[layers] = (m, n, k, h)
            total_flops += current_qk_flops

        return hook_fn_linear if module == 'linear' else hook_fn_attn

    def dfs_module_tree(module: nn.Module, layers: tuple[str]):
        if isinstance(module, transformer.Linear):
            hooks.append(module.register_forward_hook(make_hook_fn(layers, 'linear')))

        if isinstance(module, transformer.MultiheadSelfAttentionWithRope):
            hooks.append(module.register_forward_hook(make_hook_fn(layers, 'attn')))

        for child_name, child in module.named_children():
            dfs_module_tree(child, (*layers, child_name))
    
    dfs_module_tree(model, ())

    model(input)

    for hook in hooks:
        hook.remove()

    return flops, total_flops


def main():
    args = {
        'GPT-2 XL': {
            'vocab_size': 50257,
            'context_length': 1024,
            'num_layers': 48,
            'd_model': 1600,
            'num_heads': 25,
            'd_ff': 6400
        }
    }

    # (a)
    model = build_transformer(args['GPT-2 XL'])
    paras = calc_paras(model)
    print(f'{paras=:,}')

    # (b) (c)
    input = build_input(args['GPT-2 XL'])
    flops, total_flops = calc_FLOPS(model, input)
    pprint(flops, indent=4, sort_dicts=False)
    print(f'{total_flops=:,}')

    # (d) (e)
    args['GPT-2 small'] = {
        **args['GPT-2 XL'],
        'num_layers': 12,
        'd_model': 768,
        'd_ff': 768 * 4,
        'num_heads': 12
    }

    args['GPT-2 medium'] = {
        **args['GPT-2 XL'],
        'num_layers': 24,
        'd_model': 1024,
        'd_ff': 1024 * 4,
        'num_heads': 16
    }

    args['GPT-2 large'] = {
        **args['GPT-2 XL'],
        'num_layers': 36,
        'd_model': 1280,
        'd_ff': 1280 * 4,
        'num_heads': 20
    }

    args['GPT-2 XL +'] = {
        **args['GPT-2 XL'],
        'context_length': 16384
    }

    def analysis_FLOPS(model_name: str):
        model = build_transformer(args[model_name])
        input = build_input(args[model_name])

        flops, total_flops = calc_FLOPS(model, input)

        def f(conds):
            res = 0
            for layers, mm in flops.items():
                if all(cond in layers for cond in conds):
                    res += 2 * math.prod(mm)
            return res
        
        partial_flops = {
            cond: (f(cond), f(cond) / total_flops) 
            for cond in [
                ('lm_head',),
                ('attn',),
                ('ffn',),
            ]
        }

        return partial_flops
    
    for arg in args:
        print(arg, ':')

        for conds, (flops, rate) in analysis_FLOPS(arg).items():
            print(f'{".".join(conds) :<10}: {flops :>20,} | {rate * 100 :>5.2f} %')

        print()


if __name__ == '__main__':
    main()
