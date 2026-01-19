from ._deletable_heap import MaxHeap, DeletableHeap


class FasterCounter:
    def __init__(self):
        self._key_to_count = {}
        self._max_heap = DeletableHeap(MaxHeap)
    
    def _update(self, key, old_count, new_count):
        if old_count > 0:
            self._max_heap.erase((old_count, key))
        
        if new_count > 0:
            self._max_heap.push((new_count, key))
            self._key_to_count[key] = new_count
        elif key in self._key_to_count:
            del self._key_to_count[key]
    
    def __getitem__(self, key):
        return self._key_to_count.get(key, 0)

    def __setitem__(self, key, count):
        old_count = self._key_to_count.get(key, 0)
        self._update(key, old_count, count)
    
    def __delitem__(self, key):
        old_count = self._key_to_count.get(key, 0)
        self._update(key, old_count, 0)
    
    def empty(self):
        return self._max_heap.empty()

    def most_common_1(self):
        count, key = self._max_heap.top()
        return key, count