############################################
# Project: MCT-TFE
# File: main.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################
from typing import List, Tuple

import torch
from numpy.core.multiarray import ndarray
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from MCTNode import Node
import TFE as tfet
from MCT import *


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear = nn.Linear(16, 128)
        self.action_head = nn.Linear(128, 4)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.linear(x.view(-1)))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


# TODO: Make your NN global or something?

# Function to generate tuples of size two:
# (
#    UCB to next state taking direction i, 
#    current node value 
# )
# Input is specified NxN play field.
# UCB should be in dictionary form where
# keys are shown in DIR_KEY within mct_config
model = Policy()
optimizer = Adam(model.parameters(), lr=3e-2)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def genValueFunction(grid):
    # encoding with log2
    x = np.log2(grid)
    x[x == -np.inf] = 0

    probs, state_value = model(Variable(torch.Tensor(grid)))
    s = sum(probs.data)

    prob_dict = {dir: probs.data[DIR_KEY[dir]] / s for dir in 'dlru'}
    return prob_dict, state_value.data[0]


# Function called for backprop. Arguments are archived list of actions
# queuedActions is an array of (grid, action-letter, action-state prob, state value, node)
def policyUpdate(actions: List[Tuple[ndarray, str, float, int, Node]]):
    # Insert backrpop logic
    # print(actions)

    policy_losses = []
    value_losses = []

    for grid, letter, _, _, node in actions:
        output_v = node.guess_val
        target_v = node.val  # node.val or node.UCB or write our own

        # print(node.parent.chldren)
        print()

        print(letter)
        print(grid)
        # print(node.parent)
        # break
        e = np.array([child.UCB for child in node.parent.children])

        max_e = np.max(e)
        # e = e - max_e
        denominator = e.sum()
        # print(e)
        # print(max_e)
        prob = np.exp(node.parent.children_action[letter] - max_e) / denominator

        # for (log_prob, value), r in zip(saved_actions, rewards):
        #     diff_reward = target_v - output_v
        # reward = r - value.data[0]
        # print(prob)
        policy_losses.append(- np.log(prob) * (target_v - output_v))
        value_losses.append(F.smooth_l1_loss(Variable(Tensor([output_v])), Variable(Tensor([target_v]))))
    # optimizer.zero_grad()
    # loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # loss.backward()
    pass


def main():
    board_width = 4
    tfe = tfet.TFE(board_width)
    # generate a new
    tfe.putNew()
    tfe.putNew()
    print("STARTING BOARD: ")
    # tfe.grid = np.array([0, 0, 2, 16, 0 ,0 , 64, 4, 0, 8, 16, 256, 2,4, 2, 16]).reshape((4,4))
    print(tfe.grid)
    print("")

    mct = MCT(board_width, genValueFunction, policyUpdate)
    while (not tfe.isWin()) and (not tfe.isLose()):
        start = time.clock()

        print("*********************")

        act = mct.run(tfe, MONTE_CARLO_RUN, True)

        print("AI SELECT ACTION: " + str(act))
        print("*********************")
        print("BEFORE: ")
        print(tfe.grid)
        print("")

        print("*********************")
        print("AFTER: ")

        # move grid
        tfe.moveGrid(act)

        # generate a new
        tfe.putNew()

        print(tfe.grid)
        print("")

        print("TIME TAKEN FOR STEP: " + str(time.clock() - start))
        print("")
        # Flush it
        sys.stdout.flush()

    print("FINISHED: ")
    print(tfe.grid)
    print("")

    print("IS WIN?: ")
    print(tfe.isWin())
    print("")

    print("IS LOSE?: ")
    print(tfe.isLose())
    print("")


if __name__ == '__main__':
    main()
