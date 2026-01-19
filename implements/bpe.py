import os
import regex as re
from collections import defaultdict

from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries
from ._bi_linked_list import BiNode, BiLinkedList
from ._faster_counter import FasterCounter


def pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 64
) -> dict[str, int]:
    
    pattern_special_tokens = '|'.join(map(re.escape, special_tokens))
    pattern_pre_tokens = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        re.UNICODE
    )

    pre_tokens_count: dict[str, int] = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), desc='pre_tokenize', unit='chunk'):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # strip out all special tokens from chunk
            for doc in re.splititer(pattern_special_tokens, chunk):
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                for match in re.finditer(pattern_pre_tokens, doc):
                    pre_token = match.group()
                    pre_tokens_count[pre_token] += 1
            
    return pre_tokens_count


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 64,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # initialize vocab
    assert len(special_tokens) + 256 <= vocab_size, 'Total number of special tokens and base characters exceeds vocabulary size.'

    vocab_total = len(special_tokens)
    vocab: dict[int, bytes] = {
        i: pre_token.encode() for i, pre_token
        in enumerate(special_tokens)
    }

    for i in range(256):
        vocab[vocab_total] = bytes([i])
        vocab_total += 1

    # initilize token pairs    
    pre_tokens_and_count: list[tuple[BiLinkedList, int]] = [(
            BiLinkedList(map(
                lambda x: bytes([x]),
                pre_token.encode()
            )),
            count
        )
        for pre_token, count in pre_tokenize(input_path, special_tokens, num_processes).items()
    ]

    # (token1, token2) -> {(node of token1, index of their pre tokens in pre_tokens_and_count)}
    token_pairs_pointer: dict[tuple[bytes, bytes], set[tuple[BiNode,int]]] = defaultdict(set)
    token_pairs_count = FasterCounter()

    for k, (pre_token, count) in enumerate(pre_tokens_and_count):
        i = next(iter(pre_token))
        j = next(i)

        while j != pre_token.tail:
            token_pair: tuple[bytes, bytes] = i.value, j.value
            token_pairs_pointer[token_pair].add((i, k))
            token_pairs_count[token_pair] += count

            i = j
            j = next(j)

    merges: list[tuple[bytes, bytes]] = []

    pbar = tqdm(total=vocab_size - vocab_total, desc='merge tokens', unit='merge')
    while vocab_total < vocab_size:
        if token_pairs_count.empty():
            break

        token_pair, _count = token_pairs_count.most_common_1()
        token1, token2 = token_pair

        new_id = vocab_total
        new_token = token1 + token2
        vocab_total += 1
        vocab[new_id] = new_token
        merges.append(token_pair)

        for node1, i in set(token_pairs_pointer[token_pair]):
            if (node1, i) not in token_pairs_pointer[token_pair]:
                continue

            node2 = next(node1)
            bilist, count = pre_tokens_and_count[i]
            node0, node3 = node1.prev, next(node2)

            if node0 != bilist.head:
                token_pairs_pointer[(node0.value, node1.value)].remove((node0, i))
                token_pairs_count[(node0.value, node1.value)] -= count
                token_pairs_pointer[(node0.value, new_token)].add((node0, i))
                token_pairs_count[(node0.value, new_token)] += count
            
            if node3 != bilist.tail:
                token_pairs_pointer[(node2.value, node3.value)].remove((node2, i))
                token_pairs_count[(node2.value, node3.value)] -= count
                token_pairs_pointer[(new_token, node3.value)].add((node1, i))
                token_pairs_count[(new_token, node3.value)] += count
            
            node1.value = new_token
            bilist.erase(node2)
        
        del token_pairs_pointer[token_pair]
        del token_pairs_count[token_pair]

        pbar.update()
    
    pbar.close()

    return vocab, merges