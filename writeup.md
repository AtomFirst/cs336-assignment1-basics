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
- (a)
  - time: 308.32 s, memory: 268048 kb (=0.25GB) (by usr/bin/time)
  - longest tokens: [b' accomplishment', b' disappointment', b' responsibility']
- (b) (by cProfile)
  - train_bpe: 550.6
    - pre_tokenize: 468.6 (maybe loops is too slow in py)
    - get_most_common_with_tiebreaker: 75.98
      - most_common: 38.01

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText
- my os break down few times
- (a)
  - it seems that Counter.most_common(1) is O(nlogn), so the whole token merging is O(n2logn) 
  - time: 11h34m, memory: 16.1 GB (by usr/bin/time)
  - longest tokens: [b' disproportionately', b' telecommunications']
- (b) 