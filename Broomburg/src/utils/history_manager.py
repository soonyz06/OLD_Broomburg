from collections import deque

class HistoryManager:
    def __init__(self, max_depth=50):
        self.undo_stack = deque()
        self.redo_stack = deque()
        self.max_depth = max_depth
    def do(self, command):
        self.undo_stack.append(command)
        self.redo_stack.clear()
        self._trim()
    def undo(self):
        if not self.undo_stack:
            return None
        cmd = self.undo_stack.pop()
        self.redo_stack.append(cmd)
        return cmd
    def redo(self):
        if not self.redo_stack:
            return None
        cmd = self.redo_stack.pop()
        self.undo_stack.append(cmd)
        return cmd
    def _trim(self):
        while len(self.undo_stack) > self.max_depth:
            self.undo_stack.popleft()
    def peek(self):
        if not self.undo_stack:
            return None
        return self.undo_stack[-1]
            
