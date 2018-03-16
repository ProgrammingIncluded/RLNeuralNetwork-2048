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
optimizer = torch.optim.SGD(NN.parameters(), lr=0.01)

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
    resultDict = {0: resultsVect[0], 1: resultsVect[1]}
    # too lazy, just put it here
    resultDict[2] = resultsVect[2]
    resultDict[3] = resultsVect[3]

    resultTuple = (resultDict, resultsVect[4])
    
    NN.zero_grad()
    optimizer.zero_grad()
    return resultTuple

# Function called for backprop. Arguments are archived list of actions
# queuedActions is an array of (action-state probs, state value, node)
def policyUpdate(actions):
    # Again, run the neural network, this is lazy coding
    # We back prop this time.
    for v in actions:
        inVect = torch.from_numpy(v[2].game.grid.flatten())

        # Get the batch shape
        inVect = inVect.unsqueeze(0)
        results = NN.foward(Variable(inVect).float())
        
        # Prepare training.
        resultsCopy = torch.FloatTensor([0,0,0,0,0])
        # If probs does not have value, should be zero

        # Normalize the action-state probs
        probs = v[0]
        for key,val in probs.items():
            resultsCopy[key] = val
            
        # copy down whatever we have for the state value
        resultsCopy[4] = v[1]

        loss = criterion(results, Variable(resultsCopy))
        loss.backward()
        optimizer.step()




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

    mct = MCTZero(tfe, MONTE_CARLO_RUN, genValueFunction, policyUpdate)
    while (not tfe.isWin()) and (not tfe.isLose()):

        start = time.clock() 

        print("*********************")

        act = mct.playerDecision()


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
        advDecision = tfe.putNew()
        mct.adversaryDecision(advDecision)

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