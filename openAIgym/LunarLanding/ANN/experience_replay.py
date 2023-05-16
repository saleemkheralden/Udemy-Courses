import random
import torch

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.size = 0

    def sample(self, batch_size=100):
        """
        :param batch_size: the size of the desired batch
        :return: tuple of batches (state batch, next state batch, reward batch, action batch)
        """
        # sample would look like
        # ((state1, state2, ..., state_batchsize),
        #  (nextstate1, ..., nextstate_batchsize),
        #  (reward1, ..., reward_batchsize),
        #  (action1, ..., action_batchsize))

        # the map function would return each tuple as a batch in torch
        # batch_size = min(batch_size, len(self.memory))
        sample = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), sample)

    def add(self, event):
        """
        :param event: event consists of (state, next state, reward, action)
        :return: None
        """
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        self.size = len(self.memory)














