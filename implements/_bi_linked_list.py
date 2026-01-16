from typing import Iterator


class BiNode:
    def __init__(self, value=None):
        self.value = value
        self.prev: BiNode = self
        self.next: BiNode = self

    def __next__(self):
        return self.next


class BiLinkedList:
    def __init__(self, iterable=None):
        self.head = BiNode()
        self.tail = BiNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        
        if iterable is not None:
            self._extend(iterable)
    
    def _extend(self, iterable):
        for value in iterable:
            self.append(value)

    def append(self, value):
        new_node = BiNode(value)
        
        last = self.tail.prev
        last.next = new_node
        new_node.prev = last
        new_node.next = self.tail
        self.tail.prev = new_node
        
        return new_node
    
    def erase(self, node):
        if node is None or node.prev is None or node.next is None:
            raise ValueError("Invaild BiNode")
        
        node.prev.next = node.next
        node.next.prev = node.prev
        
        value = node.value
        del node

        return value

    def __iter__(self) -> Iterator[BiNode]:
        current = self.head.next
        while current != self.tail:
            yield current
            current = current.next
    
    def __repr__(self):
        values = [node.value for node in self]
        return f"BiLinkedList({values})"
