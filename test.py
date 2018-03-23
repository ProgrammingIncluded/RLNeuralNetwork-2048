############################################
# Project: MCT-TFE
# File: test.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

import copy
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
from torch.utils.data import DataLoader,TensorDataset
import torch

model = torch.load("model2.torch")
def getDecision(grid):
    grid = grid.copy()
    
    # Convert numpy grid to input for NN.
    logGrid = copy.deepcopy(grid.flatten())
    logGrid[logGrid!=0] = np.log2(logGrid[logGrid!=0])
    encoded_logGrid = np.zeros((logGrid.shape[0],12))
    encoded_logGrid[np.arange(logGrid.shape[0]),logGrid] = 1
    input = torch.from_numpy(encoded_logGrid.flatten())

    # Get the batch shape
    input = input.unsqueeze(0)
    output = model(Variable(input).float())
    output = output.squeeze(0)

    numoutput = output.data.numpy()
    return DIR_VAL[np.argmax(numoutput[0:4])]



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

    # mct = MCTZero(tfe, MONTE_CARLO_RUN, genValueFunction, policyUpdate)
    while (not tfe.isWin()) and (not tfe.isLose()):

        start = time.clock()

        print("*********************")

        # For the MCTZero
        # act = mct.playerDecision()

        # For MCTEZ
        act = getDecision(tfe.grid)


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
        # advDecision = tfe.putNew()
        # mct.adversaryDecision(advDecision)

        # Try MCTEZ
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