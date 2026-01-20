import os
import regex as re
from typing import Iterable, Iterator
import pickle

from ._bi_linked_list import BiLinkedList
from ._heap_dict import HeapDict

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        if special_tokens is not None:
            base = max(vocab.keys())
            new_special_tokens = set(map(lambda s: s.encode(), special_tokens)) - set(vocab.values())
            for i, token in enumerate(new_special_tokens):
                self.vocab[i + base + 1] = token
        self.token2id = {token: id for id, token in self.vocab.items()}

        self.merges = merges
        self.merges2order = {merge: order for order, merge in enumerate(merges)}
        self.total_merges = len(self.merges)

        self.special_tokens = special_tokens

        # regex pattern build
        if special_tokens is not None:
            self.pattern_special_tokens = '(' + '|'.join(map(
                re.escape,
                sorted(special_tokens, key=len, reverse=True)
            )) + ')'

        self.pattern_pre_tokens = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE
        )

    @classmethod
    def from_files(
        cls,
        bpe_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None
    ) -> 'Tokenizer':
        with open(bpe_filepath, 'rb') as f:
            data = pickle.load(f)

        vocab, merges = data['vocab'], data['merges']

        return cls(vocab, merges, special_tokens)

    def _merge_tokens(
        self,
        pre_token: bytes
    ) -> BiLinkedList:
        tokens = BiLinkedList(map(lambda x: bytes([x]), pre_token))

        token_pair_order = HeapDict()

        p = next(iter(tokens))
        q = next(p)
        tot = 0

        while q != tokens.tail:
            token_pair: tuple[bytes, bytes] = p.value, q.value
            order = self.merges2order.get(token_pair)
            if order is not None:
                token_pair_order[p] = order, tot
                tot += 1

            p = q
            q = next(q)

        while not token_pair_order.empty():
            node1, (order, index) = token_pair_order.top()
            node2 = next(node1)
            token_pair: bytes = node1.value + node2.value
            node0, node3 = node1.prev, next(node2)

            node1.value = token_pair
            tokens.erase(node2)
            del token_pair_order[node1]

            if node0 != tokens.head:
                del token_pair_order[node0]
                order = self.merges2order.get((node0.value, token_pair))
                if order is not None:
                    token_pair_order[node0] = order, index - 1

            if node3 != tokens.tail:
                del token_pair_order[node2]
                order = self.merges2order.get((token_pair, node3.value))
                if order is not None:
                    token_pair_order[node1] = order, index

        return tokens

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for str in iterable:
            for doc in (re.splititer(self.pattern_special_tokens, str) if self.special_tokens is not None else [str]):
                if (self.special_tokens is not None) and doc in self.special_tokens:
                    yield self.token2id[doc.encode()]
                else:
                    for match in re.finditer(self.pattern_pre_tokens, doc):
                        pre_token = match.group().encode()
                        tokens = self._merge_tokens(pre_token)
                        
                        for token in tokens:
                            yield self.token2id[token.value]

    def encode(
        self,
        text: str
    ) -> list[int]:
        return list(self.encode_iterable([text]))

    def decode(
        self,
        ids: list[int]
    ) -> str:
        tokens = [self.vocab[id] for id in ids]
        return b''.join(tokens).decode(errors='replace')