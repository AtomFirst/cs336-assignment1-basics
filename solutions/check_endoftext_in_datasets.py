import os

import numpy as np


def memmap(file_path, dtype):
    file_size_bytes = os.path.getsize(file_path)
    num_elements = file_size_bytes // np.dtype(dtype).itemsize
    return np.memmap(file_path, dtype=dtype, mode='r', shape=(num_elements,))


token_id = memmap('./data/TinyStoriesV2-GPT4-valid.bin', np.uint16)
print(f'{0 in token_id = }')
print(token_id[ :100])