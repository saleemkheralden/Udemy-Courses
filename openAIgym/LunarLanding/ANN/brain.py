import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class DQN:
    def __init__(self,
                 input_size,
                 output_size,
                 memory,
                 brain=None,
                 T=0.1,
                 gamma=0.9,
                 lr=0.001):
        if brain is None:
            self.brain = CNN(input_size=input_size, output_size=output_size)
        elif isinstance(brain, nn.Module):
            self.brain: nn.Module = brain
        else:
            self.brain: nn.Module = brain(input_size=input_size,
                                          output_size=output_size)

        self.lr = lr
        self.optimizer = optim.Adam(self.brain.parameters(), lr=lr)

        self.gamma = gamma
        self.T = T
        self.memory = memory
        self.softmax = nn.Softmax(dim=1)

        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0



    def train(self,
              batch_state: torch.tensor,
              batch_next_state: torch.tensor,
              batch_reward: torch.tensor,
              batch_action: torch.tensor):
        """
        :param batch_state: the current state (s)
        :param batch_next_state: the next state (s')
        :param batch_reward: the reward of the state (R(s))
        :param batch_action: the action taken from the current state (a)

        form the bellman equation, Q(s, a) = R(s) + gamma * max_a'(Q(s', a'))

        the outputs is the estimate of the left side of the equation, Q(s, a), line 54
        the target is the right side of the euqation, R(s) + gamma * max_a'(Q(s', a')), line 55
        """

        outputs = self.brain(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)  # Q(s, a)
        next_outputs = self.brain(batch_next_state).detach().max(1)[0]  # max_a'(Q(s', a'))
        target = batch_reward + self.gamma * next_outputs
        loss = f.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, new_state, reward, threshold=100):
        self.memory.add((self.last_state,
                         new_state,
                         torch.Tensor([self.last_reward]),
                         torch.LongTensor([self.last_action])))
        action = self.get_action(new_state)
        if self.memory.size > threshold:
            state_batch, next_state_batch, reward_batch, action_batch = self.memory.sample()
            self.train(batch_state=state_batch,
                       batch_next_state=next_state_batch,
                       batch_action=action_batch,
                       batch_reward=reward_batch)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        return action

    def get_action(self, input):
        Q_values = self.brain(input)
        probs = self.softmax(Q_values / self.T)
        action = probs.multinomial(num_samples=1).data[0, 0]
        self.last_action = action
        return action


class CNN(nn.Module):
    def __init__(self, output_size, input_size=(1, 84, 84)):
        super(CNN, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )  # 1 x 84 x 84 -> 16 x 84 x 84

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )  # 16 x 84 x 84 -> 32 x 42 x 42

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )  # 32 x 42 x 42 -> 64 x 21 x 21


        self.ann = ANN(input_size=self.count_neurons(),
                             output_size=output_size)

    def count_neurons(self):
        x = torch.rand(1, *self.input_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.data.view(1, -1).size(1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.ann(x)
        return x



# ANN brain
class ANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ANN, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_size,
                      out_features=64),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=next(self.fc1.children()).out_features,
                      out_features=128),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=next(self.fc2.children()).out_features,
                      out_features=256),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(in_features=next(self.fc2.children()).out_features,
                      out_features=256),
            nn.ReLU()
        )

        self.last_layer = nn.Sequential(
            nn.Linear(in_features=next(self.fc4.children()).out_features,
                      out_features=output_size)
        )

    def forward(self, input) -> torch.tensor:
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.last_layer(x)
        return x












