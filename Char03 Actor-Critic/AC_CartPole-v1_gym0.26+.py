'''
gym 0.26+에서 reset()은 (obs, info) 반환
obs, info = env.reset() 해줘야 함 
기존에는 obs = env.reset()이었음

env.step()은 (obs, reward, done, truncated, info) 반환
기존에는 obs, reward, done, truncated = env.step(action)이었음
done = done or truncated  # 하나라도 True면 종료 추가

'''

import gym, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

plt.ion()

#Parameters
env = gym.make('CartPole-v1')
env = env.unwrapped  # 꼭 필요한 경우에만 unwrapped 사용

# reset 시 (obs, info) 튜플을 반환하므로, 언패킹
obs, info = env.reset(seed=1)
torch.manual_seed(1)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

#Hyperparameters
learning_rate = 0.01
gamma = 0.99
episodes = 20000
render = False
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, 32)
        self.action_head = nn.Linear(32, action_space)  # actor
        self.value_head = nn.Linear(32, 1)              # critic

        self.save_actions = []
        self.rewards = []
        os.makedirs('./AC_CartPole-v0', exist_ok=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        return F.softmax(action_score, dim=-1), state_value

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)
    path = './AC_CartPole-v0/' + 'RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 200 == 0:
        plt.savefig(path)
    plt.pause(0.1)

def select_action(state):
    # state: np.ndarray
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample() 
    model.save_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode():
    R = 0
    saved_actions = model.save_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)  # actor 손실
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))  # critic 손실

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]

def main():
    os.makedirs('./AC_CartPole_Model', exist_ok=True)
    running_reward = 10
    live_time = []

    for i_episode in range(episodes):
        # Gym 0.26+에서 reset()은 (obs, info) 반환
        obs, info = env.reset()
        
        for t in count():
            action = select_action(obs)
            # step()은 (obs, reward, done, truncated, info) 반환
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated  # 하나라도 True면 종료

            if render:
                env.render()
            model.rewards.append(reward)

            if done or t >= 1000:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        live_time.append(t)
        plot(live_time)
        # print(live_time)
        if i_episode % 100 == 0:
            modelPath = f'./AC_CartPole_Model/ModelTraing{i_episode}Times.pkl'
            torch.save(model, modelPath)

        finish_episode()

if __name__ == '__main__':
    main()
