class Queue:
    def __init__(self, maxsize=None):
        self.items = []
        self.max_size = maxsize

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        if self.max_size is None or len(self.items) < self.max_size:
            self.items.append(item)
        elif len(self.items) == self.max_size:
            self.dequeue()
            self.items.append(item)
        else:
            print('the item size in queue is more than max_size')

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            print("Queue is empty.")

    def size(self):
        return len(self.items)

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        else:
            print("Queue is empty.")

    def clear(self):
        self.items.clear()