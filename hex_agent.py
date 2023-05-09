import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Categorical
import torch.optim as optim

#other imports
import math
import random
import datetime
import matplotlib.pyplot as plt
from collections import namedtuple, deque, OrderedDict

import numpy as np
from joblib import dump, load

# hex env
from fhtw_hex.hex_engine import hexPosition

torch.set_printoptions(threshold=10000)


class Actor(nn.Module):
    def __init__(self, n_actions, device, in_channels=1, kernel_size=3):
        super().__init__()

        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
        )

        self.lin = nn.Sequential(
            nn.Linear(64, n_actions),
            nn.Softmax()
        )

    def forward(self, X):
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        #print(len(tensor))
        tensor.unsqueeze_(-1)
        tensor = tensor.expand(len(tensor), len(tensor), 1)
        tensor = tensor.permute(2, 0, 1)
        #print(tensor)

        x = self.conv(tensor)
        x = torch.transpose(x, 0, 1)
        x = self.lin(x)

        return x.flatten()


class Critic(nn.Module):
    def __init__(self, device, in_channels=1, kernel_size=3):
        super().__init__()
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
        )

        self.lin = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softmax()
        )

    def forward(self, X):
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        #print(len(tensor))
        tensor.unsqueeze_(-1)
        tensor = tensor.expand(len(tensor), len(tensor), 1)
        tensor = tensor.permute(2, 0, 1)
        #print(tensor)

        x = self.conv(tensor)
        x = torch.transpose(x, 0, 1)
        x = self.lin(x)

        return x.flatten()


class Memory:
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)



class A2CAgent:
    """
        A2C Learning wrapper.

        Attributes
        ----------
        env : fhtw.hexPosition
            HEX Environment
        device : cuda.device
            The hardware used by torch in computation
        memory : replayMemory
            The transition memory of the A2C learner.
        n_actions : int
            Number of actions in the environment = board size squared.
        episode_reward : list[int]
            A list of episode rewards. In principle if agent won (1), lost (-1) or tied (0)
    """

    def __init__(self, board_size=7, env=None, memory_length=1000, kernel_size=3, opponent=None):
        self.env = hexPosition(size=board_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.n_actions = board_size**2
        
        self.state, self.player, self.winner = self.env.reset()
        
        self.episode_rewards = []
        self.episode_durations = []
        self.memory = Memory()

        self.actor = Actor(self.n_actions, self.device, in_channels=1, kernel_size=kernel_size)
        # set up critic
        # TODO:
        self.critic = Critic(self.device, in_channels=1, kernel_size=kernel_size)        
        self.opponent = opponent

    def get_action_space(self):
        return self.env.get_action_space()

    def plot_durations(self, averaging_window=50, title=""):
        """
        Visually represent the learning history to standard output.
        """
        averages = []
        for i in range(1, len(self.episode_durations) + 1):
            lower = max(0, i - averaging_window)
            averages.append(sum(self.episode_durations[lower:i]) / (i - lower))
        plt.xlabel("Episode")
        plt.ylabel("Episode length with " + str(averaging_window) + "-running average")
        plt.title(title)
        plt.plot(averages, color="black")
        plt.scatter(range(len(self.episode_durations)), self.episode_durations, s=2)
        plt.show()

    def play(self, num_episodes=500, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3):
        pass

    def _get_action(self, actor, state):
        probs = actor(state)
        #print(f"actual probs: {probs}")
        # set played probabilities impossibly low that they are not picked
        action_space = self.get_action_space()
        #print(action_space)
        played_logic = probs.detach().clone()

        for (x, y) in action_space:
            played_logic[x*self.env.size + y] = 0

        #print(played_logic)

        new_probs = torch.sub(probs, played_logic)

        #print(new_probs)

        dist = torch.distributions.Categorical(probs=new_probs)
        action = dist.sample()
        #print(action)
        log_prob = dist.log_prob(action)
        x = int(action.detach().numpy() / self.env.size)
        y = int(action.detach().numpy() % self.env.size)
        action = (x, y)

        return action, log_prob

    def learn(self, num_episodes=500, gamma=0.99, lr_actor=1e-4, lr_critic=1e-4, agent_player=1):
        adam_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        adam_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        for i_episode in range(num_episodes):
            print(f"episode: {i_episode}")
            done = False
            total_reward = 0
            state, player, winner = self.env.reset()
            steps = 0

            game_len = 0

            while not done:
                game_len += 1
                #print(f"game len: {game_len}")

                if player == agent_player:
                    # print('player')
                    action, log_prob = self._get_action(self.actor, state)
                    #print(f"action: {action}")
                    #print(f"log prob: {log_prob}")

                    next_state, reward, done, next_player = self.env.move(action)
                    #print(next_state)
                    #print(reward)
                    #print(done)

                    self.memory.add(log_prob, self.critic(state), reward, done)

                else:
                    # print('random')
                    if self.opponent is not None:
                        action, _ = self._get_action(self.opponent, state)
                        next_state, reward, done, next_player = self.env.move(action)
                        #print('opponent')
                    
                    else:
                        next_state, reward, done, next_player = self.env._random_move()

                if done:
                    #self.env.print(invert_colors=False)
                    self.memory.rewards = [reward for i in self.memory.rewards]
                    
                    # print("done")
                    last_q_val = self.critic(next_state).detach().data.numpy()

                    values = torch.stack(self.memory.values)
                    q_vals = np.zeros((len(self.memory), 1))

                    # target values are calculated backward
                    # it's super important to handle correctly done states,
                    # for those cases we want our to target to be equal to the reward only
                    q_val = last_q_val

                    for i, (_, _, r, d) in enumerate(self.memory.reversed()):
                        q_val = r + gamma * q_val * (1.0 - d)
                        q_vals[len(self.memory) - 1 - i] = q_val  # store values from the end to the beginning

                    advantage = torch.Tensor(q_vals) - values

                    critic_loss = advantage.pow(2).mean()
                    adam_critic.zero_grad()
                    critic_loss.backward()
                    adam_critic.step()
                    torch.nn.utils.clip_grad_norm(parameters=self.critic.parameters(), max_norm=10, norm_type=2.0)

                    actor_loss = (-torch.stack(self.memory.log_probs) * advantage.detach()).mean()
                    adam_actor.zero_grad()
                    actor_loss.backward()
                    adam_actor.step()
                    torch.nn.utils.clip_grad_norm(parameters=self.actor.parameters(), max_norm=10, norm_type=2.0)

                    self.memory.clear()

                player = next_player
                state = next_state

            #print(self.env.board)
            #print(self.env.winner)
            self.episode_rewards.append(self.env.winner * agent_player)
            print(f'reward: {self.env.winner}')

    def plot_rewards(self, averaging_window=50, title="Total reward per episode (online)", version=1):
        """
        Visually represent the learning history to standard output.
        """
        averages = []
        for i in range(1,len(self.episode_rewards)+1):
            lower = max(0, i-averaging_window)
            averages.append(sum(self.episode_rewards[lower:i])/(i-lower))
        plt.xlabel("Episode")
        plt.ylabel("Episode length with "+str(averaging_window)+"-running average")
        plt.title(title)
        plt.plot(averages, color="black")
        plt.scatter(range(len(self.episode_rewards)), self.episode_rewards, s=2)
        plt.show()
        plt.savefig(f"v{version}_reward_hex.png")


dt = datetime.datetime.now()
        
version = 2
        
opponent = load(f'v{version - 1}_hex_actor.a2c')
episodes = 200000
agent = A2CAgent(board_size=7, opponent=opponent)
agent.learn(num_episodes=episodes)
agent.plot_rewards(version=version)

dump(agent.actor, f"v{version}_hex_actor.a2c")
dump(agent.episode_rewards, f"v{version}_hex_episode_rewards.a2c")
print(agent.episode_rewards)

for ep in range(10):
    e = episodes / 10
    print(f'mean {ep*e} - {(ep+1) * e }: {np.mean(agent.episode_rewards[int(ep*e) : int((ep+1) * e)])}')
    
print(datetime.datetime.now() - dt)