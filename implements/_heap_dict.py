from ._deletable_heap import MinHeap, MaxHeap, DeletableHeap


class HeapDict:
    def __init__(self, Heap=MinHeap):
        self._key_to_value = {}
        self._heap = DeletableHeap(Heap)
    
    def __getitem__(self, key):
        return self._key_to_value.get(key)

    def __setitem__(self, key, value):
        old_value = self._key_to_value.get(key)
        if old_value is not None:
            self._heap.erase((old_value, key))

        self._key_to_value[key] = value
        self._heap.push((value, key))

    def __delitem__(self, key):
        old_value = self._key_to_value.get(key)

        if old_value is not None:
            del self._key_to_value[key]
            self._heap.erase((old_value, key))
    
    def empty(self):
        return self._heap.empty()

    def top(self):
        value, key = self._heap.top()
        return key, value
