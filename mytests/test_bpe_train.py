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
        512
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

$ /usr/bin/time -v uv run test_bpe_train.py

tinystories output:
15 [b' accomplishment', b' disappointment', b' responsibility']
        Command being timed: "uv run mytests/test_bpe_train.py"
        User time (seconds): 308.32
        System time (seconds): 5.42
        Percent of CPU this job got: 99%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 5:13.87
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 268048
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 1744557
        Voluntary context switches: 18
        Involuntary context switches: 846
        Swaps: 0
        File system inputs: 0
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0

owt output:
19 [b' disproportionately', b' telecommunications']
        Command being timed: "uv run test_bpe_train.py"
        User time (seconds): 41608.06
        System time (seconds): 31.89
        Percent of CPU this job got: 99%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 11:34:14
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 16965908
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 1
        Minor (reclaiming a frame) page faults: 16686853
        Voluntary context switches: 253213
        Involuntary context switches: 1677032
        Swaps: 0
        File system inputs: 256
        File system outputs: 36176
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0

---

Problem (train_bpe_tinystories) (b)

$ python -m cProfile -o bpe_train.prof test_bpe_train.py
$ snakeviz bpe_train.prof

'''