# Define memory for Experience Replay
from collections import deque
import random
import numpy as np
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.epsilon = 1e-5

    def append(self, experience):
        max_priority = self.priorities.max() if len(self.memory) > 0 else 1.0
        if max_priority <= 0:
            max_priority = 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.pos] = experience
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from empty memory")
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]
        probs = (priorities + self.epsilon) ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors) + 1e-5
        for i, error in zip(indices, td_errors):
            self.priorities[int(i)] = float(error)

    def __len__(self):
        return len(self.memory)