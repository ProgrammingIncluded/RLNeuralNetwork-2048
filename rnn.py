
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import TFE as tfet
from MCT import MCT


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]



class Simulator:
    def __init__(self, board_width):
        self.board_width = board_width
        self.tfe = None
        self.mct = None

        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-2)
        self.reset()

    def new_board(self, board_width):
        tfe = tfet.TFE(board_width)
        # generate a new
        tfe.putNew()
        tfe.putNew()
        return tfe

    def reset(self):
        self.tfe = self.new_board(self.board_width)
        self.mct = MCT(self.board_width,NN=self.model)

    def select_action(self):

        state = torch.from_numpy(state).float()
        probs, state_value = model(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.data[0]



def main():
    # create a new 4x4 board and two numbers
    sim = Simulator(4)

    print("STARTING BOARD: ")

    torch.manual_seed(0)

    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

    running_reward = 10
    for i_episode in count(1):
        state = sim.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            model.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
