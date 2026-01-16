import os
import regex as re
from collections import defaultdict, Counter

from cs336_basics.pretokenization_example import find_chunk_boundaries
from ._bi_linked_list import BiNode, BiLinkedList


def pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str]
) -> dict[str, int]:
    
    pattern_special_tokens = '|'.join(map(re.escape, special_tokens))
    pattern_pre_tokens = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        re.UNICODE
    )

    pre_tokens_count: dict[str, int] = defaultdict(int)

    with open(input_path, "rb") as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
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
                lambda x: int(x) + len(special_tokens),
                pre_token.encode()
            )),
            count
        )
        for pre_token, count in pre_tokenize(input_path, special_tokens).items()
    ]

    # (token1, token2) -> {(node of token1, index of their pre tokens in pre_tokens_and_count)}
    token_pairs_pointer: dict[tuple[int, int], set[tuple[BiNode,int]]] = defaultdict(set)
    token_pairs_count: Counter[tuple[int,int]] = Counter()

    for k, (pre_token, count) in enumerate(pre_tokens_and_count):
        i = next(iter(pre_token))
        j = next(i)

        while j != pre_token.tail:
            token_id_pair: tuple[int, int] = i.value, j.value
            token_pairs_pointer[token_id_pair].add((i, k))
            token_pairs_count[token_id_pair] += count

            i = j
            j = next(j)

    # merge tokens
    def get_most_common_with_tiebreaker(counter: Counter[tuple[int, int]]):
        try:
            max_count = counter.most_common(1)[0][1]
            candidates = [key for key, count in counter.items() if count == max_count]
            best_key = max(candidates, key=lambda x: (vocab[x[0]], vocab[x[1]]))
        except:
            best_key = None
            max_count = 0
        return best_key, max_count

    merges: list[tuple[bytes, bytes]] = []

    while vocab_total < vocab_size:
        token_id_pair, _count = get_most_common_with_tiebreaker(token_pairs_count)
        if token_id_pair is None:
            break
        token1, token2 = map(lambda x: vocab[x], token_id_pair)

        # debug
        # tpc = [((vocab[id1], vocab[id2]), count) for (id1, id2), count in token_pairs_count.most_common()]
        # print(f'{token_id_pair=}\n{tpc=}')

        new_id = vocab_total
        vocab_total += 1
        vocab[new_id] = token1 + token2
        merges.append((token1, token2))

        # debug
        # if len(merges) == 197:
        #     max_count = token_pairs_count.most_common(1)[0][1]
        #     candidates = [key for key, count in token_pairs_count.items() if count == max_count]
        #     for a, b in candidates:
        #         print(f'{vocab[a]}, {vocab[b]} ')

        for node1, i in set(token_pairs_pointer[token_id_pair]):
            if (node1, i) not in token_pairs_pointer[token_id_pair]:
                continue

            node2 = next(node1)
            bilist, count = pre_tokens_and_count[i]
            node0, node3 = node1.prev, next(node2)

            # debug
            # vocab[None] = None
            # print(f'{vocab[node0.value]} {vocab[node1.value]} {vocab[node2.value]} {vocab[node3.value]}')

            if node0 != bilist.head:
                token_pairs_pointer[(node0.value, node1.value)].remove((node0, i))
                token_pairs_count[(node0.value, node1.value)] -= count
                token_pairs_pointer[(node0.value, new_id)].add((node0, i))
                token_pairs_count[(node0.value, new_id)] += count
            
            if node3 != bilist.tail:
                token_pairs_pointer[(node2.value, node3.value)].remove((node2, i))
                token_pairs_count[(node2.value, node3.value)] -= count
                token_pairs_pointer[(new_id, node3.value)].add((node1, i))
                token_pairs_count[(new_id, node3.value)] += count
            
            node1.value = new_id
            bilist.erase(node2)
        
        del token_pairs_count[token_id_pair]
        del token_pairs_pointer[token_id_pair]

    return vocab, merges