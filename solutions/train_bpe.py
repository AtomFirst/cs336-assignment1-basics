import argparse
import pickle
from pathlib import Path

from implements import bpe


def get_longest_tokens(vocab: dict[int, bytes]) -> tuple[int, list[bytes]]:
    max_len = 0
    longest_tokens = []
    
    for token in vocab.values():
        if len(token) > max_len:
            longest_tokens = [token]
            max_len = len(token)
        elif len(token) == max_len:
            longest_tokens.append(token)

    return max_len, longest_tokens


def work(dataset_name: str, vocab_size: int, num_process: int):
    vocab, merges = bpe.train_bpe(
        Path('../data') / f'{dataset_name}train.txt',
        vocab_size,
        ['<|endoftext|>'],
        num_process
    )

    data = {
        'dataset': dataset_name,
        'vocab': vocab,
        'merges': merges
    }

    with open(f'bpe-{dataset_name}train.pkl', 'wb') as f:
        pickle.dump(data, f)

    max_len, longest_tokens = get_longest_tokens(vocab)
    
    print(dataset_name, max_len, longest_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    parser.add_argument('--num-process', type=int, required=True)
    
    args = parser.parse_args()
    work(args.dataset_name, args.vocab_size, args.num_process)


if __name__ == '__main__':
    main()
