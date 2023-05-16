import gym
import torch
import matplotlib.pyplot as plt
from LunarLanding.ANN.brain import DQN, ANN
import colorsys
from experience_replay import ReplayMemory

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
brain = DQN(input_size=observation.shape[0],
			output_size=env.action_space.n,
			T=0.001,
			brain=ANN,
			memory=ReplayMemory())


rewards = {}
avg_reward = []
epochs = 5000

for i in range(epochs):
	terminated, truncated = False, False
	rewards[f'epoch-{i}'] = []
	reward = 0

	while not (terminated or truncated):
		observation = torch.from_numpy(observation).float().unsqueeze(0)
		action = brain.update(observation, reward).item()
		observation, reward, terminated, truncated, info = env.step(action)


		rewards[f'epoch-{i}'].append(reward)

		if terminated or truncated:
			observation, info = env.reset()
			avg_reward.append(sum(rewards[f'epoch-{i}']) / len(rewards[f'epoch-{i}']))
			print(f'epoch-{i}\taverage reward:', avg_reward[-1])

env.close()


def HSVToRGB(h, s, v):
	(r, g, b) = colorsys.hsv_to_rgb(h, s, v)
	return int(255 * r) / 255, int(255 * g) / 255, int(255 * b) / 255

def getDistinctColors(n):
	huePartition = 1.0 / (n + 1)
	return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]

colors = getDistinctColors(epochs)

for i, (epoch, graph) in enumerate(rewards.items()):
	r, g, b = colors[i]
	plt.plot(range(len(graph)), graph, color=(r, g, b, 1), label=epoch)
# plt.legend()
plt.title("Score as function of batches")
plt.xlabel("Batch number")
plt.ylabel("Score")
plt.show()

plt.plot(range(len(avg_reward)), avg_reward)
plt.title("Average Score as function of epochs")
plt.xlabel("Epoch")
plt.ylabel("Average score")
plt.show()

torch.save(brain.brain.state_dict(), "model_weights_ann.pkl")

# brain.brain.load_state_dict(torch.load("model_weights.pkl"))
