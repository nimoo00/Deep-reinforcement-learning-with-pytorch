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

#Parameters
env = gym.make('CartPole-v1')
env = env.unwrapped

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

        self.action_head = nn.Linear(32, action_space) # actor 역할
        self.value_head = nn.Linear(32, 1) # Scalar Value # critic 역할

        self.save_actions = [] # log_prob, value 저장
        self.rewards = [] # reward 저장
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
    plt.pause(0.0000001)

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample() # categorical 분포를 사용해 확률적으로 행동 선택
    model.save_actions.append(SavedAction(m.log_prob(action), state_value)) # log_prob, value 저장

    return action.item()


def finish_episode():
    R = 0
    save_actions = model.save_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in model.rewards[::-1]: # 저장했던 reward를 역순으로 불러오면서 누적 할인 보상 계산산
        R = r + gamma * R
        rewards.insert(0, R) 

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps) # 텐서로 변환 후 정규화

    for (log_prob , value), r in zip(save_actions, rewards): # 실제 받은 보상인 rewards와 예측한 보상인 value의 차이를 이용해 loss 계산
        reward = r - value.item()
        policy_loss.append(-log_prob * reward) # actor(정책) 손실 계산
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r]))) # critic(가치) 손실 계산

    optimizer.zero_grad() # optimizer에 저장되어 있는 이전 배치의 그래디언트를 초기화
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum() # actor와 critic의 손실을 더해 최종 손실 계산
    loss.backward() # 최종 손실에 대해 역전파하여 각 파라미터에 대한 그래디언트 계산
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]

def main():
    running_reward = 10
    live_time = []
    for i_episode in count(episodes):
        state = env.reset()
        for t in count():
            action = select_action(state)
            state, reward, done, truncated, info = env.step(action)
            if render: env.render()
            model.rewards.append(reward)

            if done or t >= 1000:
                break
        running_reward = running_reward * 0.99 + t * 0.01
        live_time.append(t)
        plot(live_time)
        if i_episode % 100 == 0:
            modelPath = './AC_CartPole_Model/ModelTraing'+str(i_episode)+'Times.pkl'
            torch.save(model, modelPath)
        finish_episode()

if __name__ == '__main__':
    main()
