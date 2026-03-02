import argparse
from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from implements.transformer import TransformerLM
from implements.tokenizer import Tokenizer
from implements.generating_tokens import generate


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_generated_length: int = 128,
    temperature: float = 1.0,
    p: float = 0.9,
    device = 'cuda'
):
    inputs = torch.tensor(tokenizer.encode(prompt), dtype=torch.int32, device=device)
    # print(inputs)
    end_token_ids = [id for id, token in tokenizer.vocab.items() if token == '<|endoftext|>'.encode()]
    assert len(end_token_ids) == 1, 'end_token should be unique'
    end_token_id = end_token_ids[0]

    generated = generate(model, inputs, end_token_id, max_generated_length, temperature, p)

    for token_id in generated:
        yield tokenizer.decode([token_id])

    # return tokenizer.decode(generated.detach().cpu().reshape(-1).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', type=str, help='模型权重路径')
    parser.add_argument('-t', '--tokenizer-data-path', type=str, help='分词器数据路径')
    parser.add_argument('prompt', type=str, help='提示词')
    args = parser.parse_args()

    model = TransformerLM.from_files(args.checkpoint_path)
    tokenizer = Tokenizer.from_files(args.tokenizer_data_path)

    generated = generate_text(model, tokenizer, args.prompt)

    for token in generated:
        print(token, end='')
    print()
    # print(generated)


if __name__ == '__main__':
    main()
