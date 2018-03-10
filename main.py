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

# TODO: Make your NN global or something?

# Function to generate tuples of size two:
# (
#    probability to next state taking direction i, 
#    current node value 
# )
# Input is specified NxN play field.
# Probabilities should be in dictionary form where
# keys are shown in DIR_KEY within mct_config

def genValueFunction(grid):
    # Put training code here.
    # Insert forward prop code here
    return ({'d': 1, 'l': 1, 'r': 1, 'u': 1}, 0)

# Function called for backprop. Arguments are archived list of actions
# queuedActions is an array of (grid, action-letter, action-state prob, state value)
def policyUpdate(actions):
    # Insert backrpop logic
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