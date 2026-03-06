import yaml
import argparse

import torch

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
    parser.add_argument('-c', '--config-path', type=str, help='模型配置路径')
    parser.add_argument('-t', '--tokenizer-data-path', type=str, help='分词器数据路径')
    parser.add_argument('-m', '--max-generated-length', type=int, help='最大生成token数', default=1024)
    parser.add_argument('--temperature', type=float, help='生成温度', default=1.0)
    parser.add_argument('--p', type=float, help='累积概率阈值', default=0.9)
    parser.add_argument('prompt', type=str, help='提示词')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config: dict = yaml.safe_load(f)

    model = TransformerLM.from_files(args.config_path)
    tokenizer = Tokenizer.from_files(args.tokenizer_data_path, ['<|endoftext|>'])

    generated = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_generated_length,
        args.temperature,
        args.p,
        config['device']
    )

    for token in generated:
        print(token, end='')
    print()
    # print(generated)


if __name__ == '__main__':
    main()
