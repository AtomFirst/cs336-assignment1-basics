import regex as re
from typing import Iterable, Iterator

from ._bi_linked_list import BiLinkedList

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        if special_tokens is not None:
            base = max(vocab.keys())
            new_special_tokens = set(map(lambda s: s.encode(), special_tokens)) - set(vocab.values())
            for i, token in enumerate(new_special_tokens):
                self.vocab[i + base + 1] = token
        self.token2id = {token: id for id, token in self.vocab.items()}

        self.merges = merges
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
        bpe_filepath: str,
        special_tokens: list[str] | None = None
    ):
        raise NotImplementedError

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
                        tokens = BiLinkedList(map(lambda x: bytes([x]), pre_token))

                        # merge token
                        for fir, sec in self.merges:
                            merge = fir + sec
                            p = next(iter(tokens))
                            q = next(p)
                            
                            if q == tokens.tail:
                                break

                            while q != tokens.tail:
                                if p.value == fir and q.value == sec:
                                    p.value = merge
                                    tokens.erase(q)
                                    q = next(p)
                                
                                if q == tokens.tail:
                                    break

                                p = q
                                q = next(q)

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