import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from LunarLanding.CNN.brain import DQN, CNN
import colorsys
from experience_replay import ReplayMemory, NStepProgress
import numpy as np

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset(seed=42)

brain = DQN(input_size=(1, 84, 84),
			output_size=env.action_space.n,
			T=0.001,
			brain=CNN)

# def imresize(img, bounding):
#     start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
#     end = tuple(map(lambda x, y: x + y, start, bounding))
#     slices = tuple(map(slice, start, end))
#     return img[slices]
#
# grayscale = True
#
# img = env.render()
# img_size = (64, 64)
#
# img = imresize(img, img_size)
# if grayscale:
# 	img = img.mean(-1, keepdims=True)
# img = np.transpose(img, (2, 0, 1))
# img = img.astype('float32') / 255.
# print(img.shape)
#
# exit()

n_steps = NStepProgress(env=env, ai=brain, n_step=10)
memory = ReplayMemory(n_steps=n_steps, capacity=10000)

def eligibility_trace(batch):
	"""
	in analogy to the bellman equation:
		Q(s, a) = R(s) + gamma * max_a'(Q(s', a'))

	a more general equation of the right side of the equation would be this functions targets

	:param batch: batch of series
	:return:
	"""
	gamma = 0.99
	inputs = []
	targets = []
	for series in batch:
		# the input here is a mini-batch for the CNN network with the first and last state in this run (series)
		input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))
		output = brain.brain(input)
		cumul_reward = 0.0 if series[-1].done else output[1].data.max()
		for step in reversed(series[:-1]):
			cumul_reward = step.reward + gamma * cumul_reward
		state = series[0].state
		target = output[0].data
		target[series[0].action] = cumul_reward
		inputs.append(state)
		targets.append(target)
	return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class MA:
	def __init__(self, size):
		self.list_of_rewards = []
		self.size = size
	def add(self, rewards):
		if isinstance(rewards, list):
			self.list_of_rewards += rewards
		else:
			self.list_of_rewards.append(rewards)
		while len(self.list_of_rewards) > self.size:
			del self.list_of_rewards[0]
	def average(self):
		return np.mean(self.list_of_rewards)
ma = MA(100)

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(brain.brain.parameters(), lr=0.001)
nb_epochs = 100
rewards = []

for epoch in range(1, nb_epochs + 1):
	memory.run_steps(200)
	for batch in memory.sample_batch(128):
		inputs, targets = eligibility_trace(batch)
		inputs, targets = torch.Tensor(inputs), torch.Tensor(targets)  # analogy to the right side of bellman equation
		predictions = brain.brain(inputs)  # left side of the bellman equation
		loss_error = loss(predictions, targets)
		optimizer.zero_grad()
		loss_error.backward()
		optimizer.step()
	rewards_steps = n_steps.rewards_steps()
	ma.add(rewards_steps)
	avg_reward = ma.average()
	rewards.append(avg_reward)
	print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))

env.close()


def HSVToRGB(h, s, v):
	(r, g, b) = colorsys.hsv_to_rgb(h, s, v)
	return int(255 * r) / 255, int(255 * g) / 255, int(255 * b) / 255

def getDistinctColors(n):
	huePartition = 1.0 / (n + 1)
	return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]

colors = getDistinctColors(nb_epochs)

plt.plot(range(len(rewards)), rewards)
plt.title("Average Score as function of epochs")
plt.xlabel("Epoch")
plt.ylabel("Average score")
plt.show()

torch.save(brain.brain.state_dict(), "model_weights_cnn.pkl")

# brain.brain.load_state_dict(torch.load("model_weights.pkl"))
