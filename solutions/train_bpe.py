import pickle

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


def main():
    # dataset_name = 'TinyStoriesV2-GPT4-train'
    dataset_name = 'owt_train'

    vocab, merges = bpe.train_bpe(
        f'../data/{dataset_name}.txt',
        # 10_000,
        32_000,
        ['<|endoftext|>'],
        # 128,
        512,
    )

    data = {
        'dataset': dataset_name,
        'vocab': vocab,
        'merges': merges
    }

    with open(f'bpe-{dataset_name}.pkl', 'wb') as f:
        pickle.dump(data, f)

    max_len, longest_tokens = get_longest_tokens(vocab)
    
    print(max_len, longest_tokens)


if __name__ == '__main__':
    main()

'''     
Problem (train_bpe_*) (a)

$ /usr/bin/time -v uv run train_bpe.py

tinystories output:
pre_tokenize: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:11<00:00, 10.94chunk/s]
merge tokens: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 9743/9743 [00:22<00:00, 441.18merge/s]
15 [b' accomplishment', b' disappointment', b' responsibility']
        Command being timed: "uv run train_bpe.py"
        User time (seconds): 392.77
        System time (seconds): 5.50
        Percent of CPU this job got: 1004%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:39.66
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 341696
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 1954047
        Voluntary context switches: 2239
        Involuntary context switches: 46184
        Swaps: 0
        File system inputs: 0
        File system outputs: 440
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0

owt output:


---

Problem (train_bpe_tinystories) (b)

$ python -m cProfile -o train_bpe.prof train_bpe.py
$ snakeviz train_bpe.prof

'''