############################################
# Project: MCT-TFE
# File: main.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

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


def genValueFunction(grid):
    # encoding with log2
    x = np.log2(grid)
    x[x == -np.inf] = 0

    probs, state_value = model(Variable(torch.Tensor(grid)))
    s = sum(probs.data)
    prob_dict = {a: p / s for a, p in zip('dlru', probs.data)}
    return prob_dict, state_value.data[0]


# Function called for backprop. Arguments are archived list of actions
# queuedActions is an array of (grid, action-letter, action-state prob, state value, node)
def policyUpdate(actions):
    # Insert backrpop logic
    print(actions)
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
