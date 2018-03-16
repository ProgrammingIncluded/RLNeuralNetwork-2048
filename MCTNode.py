############################################
# Project: MCT-TFE
# File: MCTNode.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

import numpy as np
import random
from mct_config import *

class MCTNode:
    def __init__(self, parent, gameState, option, isPlayerDecision):
        self.wins = 0
        # Option to keep track of what the node's parent chose as decision
        self.opt = option
        self.totalGames = 0
        self.parent = parent
        self.game = gameState.copy()
        self.isPlayerDecision = isPlayerDecision
        self.stateActProbs = np.zeros(4)

        # Variable to hold all possible children but not yet generated.
        self.childrenOptions = []

        # List to contain actual children generated
        self.children = []

        # Sets up possible options available for children
        # Also sets up their probabilities in parent node (this current node)
        self.setUpChildrenOptions()

    # Check if node is a leaf node
    # Has to be player decision by definition
    def isLeaf(self):
        return (self.allChildrenGenerated() and len(self.children) == 0)

    # Randomly generate a child
    # Returns false if no child can be generated
    def randGenChild(self):
        if self.allChildrenGenerated():
            return False
        
        # Randomly pick an option
        randOptionIndex = random.randint(0, len(self.childrenOptions)-1)
        randOption = self.childrenOptions[randOptionIndex]
        del self.childrenOptions[randOptionIndex]

        if self.isPlayerDecision:
            copyGame = self.game.copy()
            copyGame.moveGrid(DIR_VAL[randOption])
            res = MCTNode(self, copyGame, randOption, not self.isPlayerDecision)
        else:
            copyGame = self.game.copy()
            boardSize = self.game.board_width**2
            # We need to decode our options
            valOpt = randOption // (boardSize)
            loc = randOption - valOpt * boardSize
            
            # Get value to place at position
            val = 2
            if valOpt == 1:
                val = 4
            
            # Place the actual value
            bw = self.game.board_width
            copyGame.putNewAt(loc // bw, int(loc % bw), val)
            res = MCTNode(self, copyGame, randOption, not self.isPlayerDecision)
            
        self.children.append(res)
        return res

    # check if node has all the children
    def allChildrenGenerated(self):
        return (len(self.childrenOptions) == 0)

    def setUpChildrenOptions(self):
        res = []
        # If we are the player decision, we only have the
        # number of directions to move
        if self.isPlayerDecision:
            availGameStates = self.game.availDir()
            self.childrenOptions = [DIR_KEY[k] for k, v in availGameStates.items()]

            # Setup current probabilities. Can change based off plays
            self.stateActProbs = {DIR_KEY[k]: 1/4 for k,v in availGameStates.items()}
        else:
            v_f = self.game.grid.flatten("K")
            boardSize = self.game.board_width ** 2
            # Get all positions with 0 values
            zeroPos = np.argwhere(v_f == 0).tolist()
            self.childrenOptions = [0 + y[0] for y in zeroPos]
            self.childrenOptions += [1 * boardSize + y[0] for y in zeroPos]

            # Setup current probabilities. Always fixed for these actions in adversary.
            # 9/10 for 90% probability for generating a 2
            posSize = len(zeroPos)
            # get the probability of getting a Two
            probTwo = 9/(10 * posSize)
            # Probability of getting a Four
            probFour = 1/(10 * posSize)
            
            probsTwoDict = {(y[0]): probTwo for y in zeroPos}
            probsFourDict = {(1 * boardSize + y[0]): probFour for y in zeroPos}
            self.stateActProbs = {**probsTwoDict, **probsFourDict}