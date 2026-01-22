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

### 2.5 Problem (train_bpe_tinystories): BPE Training on TinyStories
- script in mytests/test_bpe_train.py
- (a) (with multiprocess in pre_tokenizer)
  - time: 38.12 s (wall clock), memory: 324692 kbytes (by usr/bin/time)
  - longest tokens: [b' accomplishment', b' disappointment', b' responsibility']
- (b) (by cProfile)
  - train_bpe: 27.9 s
    - pre_tokenize: 19.6 s
    - FasterCounter: 20+ s (?)

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText
- my os break down few times
- (a)
  - use FasterCounter.most_common_1() with O(logn), so the whole token merging is O(nlogn)
  - it seems that merging is slower at the beginning, since more count of token pairs need to change.
  - it can be proved that the count of token pairs no increase with merging.
  - time: 2h22m, memory: 42319768kbytes (by usr/bin/time)
  - longest tokens: [b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' ('ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ' in utf-8), b'----------------------------------------------------------------']
- (b) 

### Time complexity analysis
- O(nlogn), n is the number of pre-token.
- | dataset | pre-token | time | t / (p * log(p))
  | - | - | - | - |
  | tinystories | 59933 | 38.12s | 4.01e-5 |
  | owt | 6601892 | 2h22m | 5.70e-5 |
  | owt(10000 vocab end) | 6601892 | 1h58m | 4.72e-5 |
- With the impact of python gc and cpu hopping, the time complexity fit data very well.

### 2.7 Problem (tokenizer_experiments)
- (a)(b) compression ratio
  | tokenizer \ docs | ts | owt |
  | - | - | - |
  | ts | 4.03 | 3.38 |
  | owt | 3.96 | 4.39 |

- (c) throughput
  | tokenizer \ docs | ts | owt |
  | - | - | - |
  | ts | 2.4e5 | 2.5e5 |
  | owt | 2.3e5 | 2.2e5 |

  - m = 825GB = 825 * (1024 ** 3) bytes = 885837004800 bytes
  - m / v = 3.9e6 s = 1069 h

- (d) 2 ** 16 = 65536, which greater than most vocab_size. Also, it is 2 bytes and easy to store.