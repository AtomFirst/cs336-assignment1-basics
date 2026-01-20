# writeup

## 2 BPE 
### 2.1
- (a) '\x00'
- (b) It's stringify of `__str__()` here.
- (c) '\x00' was hidden.

### 2.2
- (a) The range of utf-8 is much smaller.
- (b) it decodes bytes by bytes independently, but some code is longer than a byte. '牛'.encode('utf-8')
- (c) b'\xc0\x80'

### Problem (train_bpe_tinystories): BPE Training on TinyStories
- script in mytests/test_bpe_train.py
- (a) (with multiprocess in pre_tokenizer)
  - time: 39.66 s (wall clock), memory: 341696 kbytes (by usr/bin/time)
  - longest tokens: [b' accomplishment', b' disappointment', b' responsibility']
- (b) (by cProfile)
  - train_bpe: 68.0
    - pre_tokenize: 19.8
    - FasterCounter: 41.17

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText
- my os break down few times
- (a)
  - use FasterCounter.most_common_1() with O(logn), so the whole token merging is O(nlogn)
  - it seems that merging is slower at the beginning, since more count of token pairs need to change.
  - time: , memory:  (by usr/bin/time)
  - longest tokens: [b' disproportionately', b' telecommunications']
- (b) 