import heapq
from functools import total_ordering


@total_ordering
class ReverseOrder:
    def __init__(self, obj):
        self.obj = obj
    
    def __lt__(self, other):
        return self.obj > other.obj
    
    def __eq__(self, other):
        return self.obj == other.obj
    
    def __repr__(self):
        return f'ReverseOrder({self.obj!r})'
    
    def unwrap(self):
        return self.obj
    

class MinHeap:
    def __init__(self):
        self.siz = 0
        self.heap = []

    def push(self, x):
        self.siz += 1
        heapq.heappush(self.heap, x)

    def pop(self):
        self.siz -= 1
        return heapq.heappop(self.heap)
    
    def size(self):
        return self.siz

    def top(self):
        return self.heap[0]


class MaxHeap(MinHeap):
    def push(self, x):
        super().push(ReverseOrder(x))
    
    def pop(self):
        return super().pop().unwrap()
    
    def top(self):
        return super().top().unwrap()
    

class DeletableHeap:
    def __init__(self, Heap=MinHeap):
        self.heap = Heap()
        self.deleted = Heap()

    def _update(self):
        while self.heap.size() and self.deleted.size() and self.heap.top() == self.deleted.top():
            self.heap.pop()
            self.deleted.pop()

    def push(self, x):
        self.heap.push(x)

    def pop(self):
        self._update()
        return self.heap.pop()
    
    def erase(self, x):
        self.deleted.push(x)
        self._update()
    
    def empty(self):
        self._update()
        return self.heap.size() == 0

    def top(self):
        self._update()
        return self.heap.top()
