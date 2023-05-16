# Experience Replay

# Importing the libraries
import numpy as np
from collections import namedtuple, deque
import torch
from ImageProcess import gray_scale

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])


# Making the AI progress on several (n_step) steps

class NStepProgress:

    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step

    def __iter__(self):
        """
        here we would play in the current env until either we finish the game (reach terminate state)
        or we finish the n steps

        since this class is a generator then the while True loop will always execute
        and we have to make sure that history we have is always of size n

        note that when we reach a terminate state we enter the if is_done block
        and the generator will stay there until the history is empty

        so in short the history is a window of size n at most on exactly one run
        once empty we start new run

        :return:
        """

        state = gray_scale(self.env.render())
        history = deque()
        reward = 0.0
        while True:
            action = self.ai(torch.from_numpy(state).unsqueeze(0)).item()
            observation, r, terminated, truncated, info = self.env.step(action)
            is_done = terminated or truncated
            next_state = gray_scale(self.env.render())

            reward += r
            history.append(Step(state=state, action=action, reward=r, done=is_done))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:  # if we finish the n steps we return the history
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                observation, info = self.env.reset()
                state = gray_scale(self.env.render())
                history.clear()

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


# Implementing Experience Replay

class ReplayMemory:
    """
    Similar to MCTS or Minimax algorithm where an agent runs through the states playing until either
    it reaches a terminate state or until it finishes n steps
    """

    def __init__(self, n_steps, capacity=10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size):  # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs + 1) * batch_size]
            ofs += 1

    def run_steps(self, samples):
        """
        this would generate <samples> samples with each sample it would contain n steps in the env
        similar to Monte-Carlo method where it generates few runs and
        it would get better understanding of the rewards it gets

        :param samples: the number of samples we want
        :return:
        """
        while samples > 0:
            entry = next(self.n_steps_iter)  # 10 consecutive steps
            self.buffer.append(entry)  # we put <samples> (200) for the current episode
            samples -= 1
        while len(self.buffer) > self.capacity:  # we accumulate no more than the capacity (10000)
            self.buffer.popleft()
