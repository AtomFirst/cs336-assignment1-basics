from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
from tqdm import tqdm

from implements.tokenizer import Tokenizer


def docs_iter(
    file: BinaryIO,
    split_special_token: bytes = b'<|endoftext|>',
    chunk_size: int = 4096
) -> Iterator[str]:
    
    file.seek(0)
    token_len = len(split_special_token)
    buffer = b''
    stream_pos = 0 

    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        buffer += chunk
        
        while True:
            nxt = buffer.find(split_special_token)
            if nxt == -1:
                break
            
            end_idx = nxt + token_len
            doc_bytes = buffer[:end_idx]
            yield doc_bytes.decode('utf-8', errors='ignore')
            
            buffer = buffer[end_idx:]
            stream_pos += end_idx

    if buffer:
        yield buffer.decode('utf-8', errors='ignore')


def main():
    dataset_names = [
        'TinyStoriesV2-GPT4-',
        # 'owt_'
    ]

    cls = [
        # 'train',
        'valid'
    ]

    for dataset_name in dataset_names:
        tokenizer = Tokenizer.from_files(Path('../data') / f'bpe-{dataset_name}train.pkl') 

        for cl in cls:
            print(f'{dataset_name}{cl} tokenizing ...')

            with open(Path('../data') / f'{dataset_name}{cl}.txt', 'rb') as f:
                ids = tokenizer.encode_iterable(docs_iter(f))
                num_tokens = 0
                for _ in tqdm(ids, desc='calc num_tokens', unit='token'):
                    num_tokens += 1

                ids = tokenizer.encode_iterable(docs_iter(f))
                fp = np.memmap(Path('../data') / f'{dataset_name}{cl}.bin', dtype=np.uint16, mode='w+', shape=(num_tokens,))
                cur = 0

                for id in tqdm(ids, total=num_tokens, desc='write in file', unit='token'):
                    fp[cur] = id
                    cur += 1

                fp.flush()    


if __name__ == '__main__':
    main()
