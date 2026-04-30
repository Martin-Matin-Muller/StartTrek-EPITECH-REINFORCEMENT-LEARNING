from collections import deque
import random
import numpy as np

class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.maxlen = maxlen
        self.memory = [None] * maxlen
        self.position = 0
        self.size = 0

        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.maxlen

        if self.size < self.maxlen:
            self.size += 1

    def sample(self, sample_size):
        if sample_size > self.size:
            raise ValueError("Sample size cannot exceed current memory size.")
        return random.sample(self.memory[:self.size], sample_size)

    def __len__(self):
        return self.size

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
        td_errors = np.abs(td_errors) + self.epsilon
        for i, error in zip(indices, td_errors):
            self.priorities[int(i)] = float(error)

    def __len__(self):
        return len(self.memory)