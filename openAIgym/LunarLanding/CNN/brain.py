import torch
import torch.nn as nn
import torch.optim as optim

class DQN:
    def __init__(self,
                 input_size,
                 output_size,
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
        self.softmax = nn.Softmax(dim=1)

        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def __call__(self, inputs):
        # inputs = torch.from_numpy(np.array(inputs, dtype=np.float32))
        Q_values = self.brain(inputs)
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












