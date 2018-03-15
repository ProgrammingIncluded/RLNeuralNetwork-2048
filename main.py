############################################
# Project: MCT-TFE
# File: main.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

import TFE as tfet
import mct_config as mctconfig
from MCT import *
import time
import random as rnd
import sys
import random
import CNN as cnn
import torch.nn as nn
from torch.autograd import Variable
import torch

# All the lovely neural network stuff goes here.
BOARD_WIDTH = 4
NN = cnn.CNN(BOARD_WIDTH * BOARD_WIDTH)
criterion = nn.MSELoss()

# Function to generate tuples of size two:
# (
#    UCB to next state taking direction i, 
#    current node value 
# )
# Input is specified NxN play field.
# UCB should be in dictionary form where
# keys are shown in DIR_KEY within mct_config


def genValueFunction(grid):
    # Convert numpy grid to input for NN.
    inVect = torch.from_numpy(grid.flatten())

    # Get the batch shape
    inVect = inVect.unsqueeze(0)
    results = NN.foward(Variable(inVect).float())

    # Get results into numpy
    resultsVect = results.data.numpy()

    # Reduce dimensions
    resultsVect = resultsVect[0, :]
    # Prepare for output
    resultDict = {'d': resultsVect[0], 'l': resultsVect[1]}
    # too lazy, just put it here
    resultDict['r'] = resultsVect[2]
    resultDict['u'] = resultsVect[3]

    resultTuple = (resultDict, resultsVect[4])
    
    NN.zero_grad()
    return resultTuple

# Function called for backprop. Arguments are archived list of actions
# queuedActions is an array of (grid, action-letter, action-state prob, state value, node)
def policyUpdate(actions):
    # Insert backrpop logi
    # Get normalized values
    pass

def main():
    board_width = BOARD_WIDTH
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