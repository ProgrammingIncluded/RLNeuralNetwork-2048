from collections import namedtuple
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import TFE as tfet
from mct_config import DIR_VAL

gamma = 0.99
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
MODEL_PATH = 'model/model_saved'
LOAD_MODEL = False #prev epochs 5000
OUTPUT_PATH = 'output.txt'


def one_hot(board):
    grid = np.zeros((4, 4, 12))
    for i, row in enumerate(board):
        for j, tile in enumerate(row):
            grid[i, j, tile] = 1
    return grid


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(4*4*12, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, 4)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.linear1(x.view(-1)))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


class Simulator:

    def __init__(self, board_width):
        self.board_width = board_width
        self.tfe = None
        self.mct = None
        self.n_move = 0

        self.model = Policy()
        if LOAD_MODEL:
            self.model.load_state_dict(torch.load(MODEL_PATH))
        self.optimizer = optim.Adamax(self.model.parameters(), lr=3e-2)
        self.reset()

    def new_board(self, board_width):
        tfe = tfet.TFE(board_width)
        # generate a new
        tfe.putNew()
        tfe.putNew()
        return tfe

    def reset(self):
        # print('reset')
        self.tfe = self.new_board(self.board_width)
        self.n_move = 0
        self.model.rewards = []
        self.model.saved_actions = []
        # self.mct = MCT(self.board_width, self.value_function, None)

    def step(self):
        self.n_move += 1
        # state = torch.from_numpy(state).float()
        x = np.log2(self.tfe.grid)
        x[x == -np.inf] = 0
        x = one_hot(x.astype(int))
        # print(x)
        state = torch.Tensor(x)
        probs, state_value = self.model(Variable(state))
        # avail = [DIR_VAL[dir] for dir in self.tfe.availDir()]

        # print(m.probs.data.numpy())

        # choose next move randomly, as opposed to deterministically (i.e. argmax)
        # basically a MCT without an exploration term. Need to add exploration ?

        while True:
            # print(probs.data)

            m = Categorical(
                # F.softmax(
                probs
                # + Variable(torch.Tensor([0.01, 0.01, 0.01, 0.01])))
            )
            action = m.sample()

            old_grid = np.array(self.tfe.grid)

            self.tfe.moveGrid(DIR_VAL[action.data[0]])
            # print(f'{DIR_VAL[action.data[0]]}')
            new_grid = np.array(self.tfe.grid)


            if np.array_equal(old_grid, new_grid):
                probs.data[action.data[0]] = 0
            else:
                self.model.rewards.append(0)
                self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
                break
            """
            if np.array_equal(old_grid, new_grid):  # invalid move: negative feedback
                self.model.rewards.append(-10)
                # self.model.rewards.append(0)
    
                self.finish_episode()
            else:  # valid move: no negative feedback
                # self.model.rewards.append(self.tfe.grid.max())
                self.model.rewards.append(0)
            """
        # generate a new tile
        self.tfe.putNew()

    def finish_episode(self):
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []
        value_losses = []
        raw_rewards = self.model.rewards
        if self.tfe.isWin():
            raw_rewards[-1] = 1
            # rewards = []  # reward with decay
            # for r in reversed(raw_rewards):
            #     R = r + gamma * R
            #     rewards.insert(0, R)
            # rewards = torch.Tensor(rewards)
        elif self.tfe.isLose():
            # raw_rewards[-1] = len(saved_actions)
            raw_rewards[-1] = -1
            # rewards = []  # reward with decay
            # for r in reversed(raw_rewards):
            #     R = r + gamma * R
            #     rewards.insert(0, R)
            # rewards = torch.Tensor(rewards)
        else:
            pass
            # rewards = torch.Tensor(raw_rewards)
            # raise Exception('Game not finished yet')

        rewards = []  # reward with decay
        for r in reversed(raw_rewards):
            R = r + gamma * R
            rewards.insert(0, R)

        rewards = torch.Tensor(rewards)
        # rewards = np.array(raw_rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # print(raw_rewards)
        # print(rewards.numpy())
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.data[0]
            # print(log_prob)
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        self.reset()

    def done(self):
        return self.tfe.isLose() or self.tfe.isWin()


def main():

    #statistics
    f = open(OUTPUT_PATH,'w')
    sum_tile = []
    max_tile = []
    num_moves = []
    EPOCH_TIL_LOG = 100

    # create a new 4x4 board and two numbers
    sim = Simulator(4)

    torch.manual_seed(0)

    for i_episode in count(1):
        # sim.reset()

        while not sim.done():
            sim.step()
        print(i_episode, sim.tfe.isWin(), sim.tfe.grid.max(), len(sim.model.saved_actions), np.sum(sim.tfe.grid))
        sum_tile.append(np.sum(sim.tfe.grid))
        max_tile.append(sim.tfe.grid.max())
        num_moves.append(len(sim.model.saved_actions))

        sim.finish_episode()

        if i_episode % EPOCH_TIL_LOG == 0:
            torch.save(sim.model.state_dict(), MODEL_PATH)
            avg_sum = sum(sum_tile[-1*EPOCH_TIL_LOG:])/EPOCH_TIL_LOG
            avg_max = sum(max_tile[-1*EPOCH_TIL_LOG:])/EPOCH_TIL_LOG
            avg_moves = sum(num_moves[-1*EPOCH_TIL_LOG:])/EPOCH_TIL_LOG
            print('avg',avg_sum)
            print(i_episode,avg_max, avg_moves, avg_sum,file=f)
            f.flush()

    f.close()

if __name__ == '__main__':
    main()