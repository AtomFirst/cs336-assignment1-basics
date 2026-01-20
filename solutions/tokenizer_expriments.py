import random
from pathlib import Path
from typing import BinaryIO, List
import numpy as np
import time

from implements.tokenizer import Tokenizer


def sample_documents(
    file: BinaryIO,
    desired_num_docs: int,
    split_special_token: bytes = b'<|endoftext|>',
    chunk_size: int = 4096
) -> List[str]:
    if desired_num_docs <= 0:
        return []
    
    file.seek(0)
    token_len = len(split_special_token)

    buffer = b''
    chunk_boundaries: list[tuple[int, int]] = []
    offset = 0
    
    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        
        buffer += chunk
        pos: int = 0
        while True:
            doc_start = offset + pos

            nxt = buffer.find(split_special_token, pos)
            if nxt == -1:
                break
            pos = nxt

            doc_end = offset + pos
            chunk_boundaries.append((doc_start, doc_end))

            pos += token_len
        
        offset += pos
        buffer = buffer[pos:]

    if len(chunk_boundaries) == 0:
        return []
    
    if len(chunk_boundaries) <= desired_num_docs:
        selected_indices = range(len(chunk_boundaries))
    else:
        selected_indices = random.sample(range(len(chunk_boundaries)), desired_num_docs)
    
    documents = []

    for idx in selected_indices:
        start_pos, end_pos = chunk_boundaries[idx]
        file.seek(start_pos)
        doc_bytes = file.read(end_pos - start_pos)
        
        doc_text = doc_bytes.decode('utf-8', errors='ignore')
        documents.append(doc_text)
    
    return documents


def main():
    dataset_names = [
        'TinyStoriesV2-GPT4-',
        'owt_'
    ]

    num_dataset, num_docs = len(dataset_names), 10

    docs: list[list[str]] = []

    for dataset_name in dataset_names:
        with open(Path('../data') / f'{dataset_name}valid.txt', 'rb') as f:
            docs.append(sample_documents(f, num_docs))

    tokenizers = [
        Tokenizer.from_files(Path('.') / f'bpe-{dataset_name}train.pkl') 
        for dataset_name in dataset_names
    ]

    docs_bytes_len = np.array([
        [len(s.encode()) for s in c]
        for c in docs
    ])

    tokens_len = np.zeros([num_dataset,num_dataset,num_docs], dtype=np.int16)
    durings = np.zeros([num_dataset,num_dataset,num_docs])

    for i in range(num_dataset):
        for j in range(num_dataset):
            for k in range(num_docs):
                begin = time.perf_counter()
                encoded = tokenizers[i].encode(docs[j][k])
                end = time.perf_counter()

                tokens_len[i][j][k] = len(encoded)
                durings[i][j][k] = end - begin
    
    compression_ratio = docs_bytes_len / tokens_len
    throughput = docs_bytes_len / durings

    average_compression_ratio = np.mean(compression_ratio, -1)
    average_throughput = np.mean(throughput, -1)

    print(average_compression_ratio, average_throughput, sep='\n')

if __name__ == '__main__':
    main()

'''
output:
[[4.12733437 3.3520567 ]
 [4.02881855 4.5974067 ]]
[[240586.46870312 247093.3079057 ]
 [228965.86957351 217904.38830636]]
'''