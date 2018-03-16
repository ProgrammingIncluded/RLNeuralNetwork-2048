############################################
# Project: MCT-TFE
# File: MCT.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

from MCTNode import *
import time
import math
import operator
import numpy as np

# This monte carlo implementation assumes MCT is retained every move.
class MCTZero:
    def __init__(self, game, secondsPerMove):
        # Create a new root
        self.root = MCTNode(None, game, -1, True)
        self.game = game
        self.secondsPerMove = secondsPerMove
    
    def adversaryDecision(self, decision):
        # Decode option number
        toggle =  0 if decision[1] == 2 else 1
        boardSize = self.game.board_width ** 2
        opt =  toggle * boardSize + decision[0][1] + decision[0][0] * self.game.board_width

        # check if inside
        if opt in self.root.childrenOptions:
            self.root = self.root.genChild(opt)
        else:
            # If not, must be one of the children. Search for it
            res = None
            for child in self.root.children:
                if child.opt == opt:
                    res = child
            self.root = res
    
        # reset the root
        self.root.parent = None
        self.root.opt = -1


    # Run the simulation
    def playerDecision(self):
        # Wrap everything in a timer
        endTime = time.time() + self.secondsPerMove

        while time.time() < endTime or not self.root.allChildrenGenerated():
            # Start from the root
            curNode = self.root

            # Selection
            curNode = self.selection(curNode)

            # Expansion
            genNode = curNode.randGenChild()


            # Simulation, if false then that means we hit
            # leaf node prematurely.
            if genNode:
                curNode = self.simulation(genNode)

            # Backpropagate result
            self.backpropagate(curNode)
        
        # Times up! Time to make a decision
        # Pick the one with the highest UCB
        ucbs = self.childrenUCB(self.root)
        print(ucbs)
        act = 'u'
        if len(self.root.childrenOptions) != 0:
            print("MCTS not enough time")
            return act
        
        # Before we leave, update new root
        self.root = self.root.children[np.argmax(ucbs)]
        seloption = self.root.opt

        # Reset
        self.root.parent = None
        self.opt = -1
        
        # Convert int key into a letter
        return DIR_VAL[seloption]
    
    def backpropagate(self, node):
        win = 1 if node.game.isWin() else 0
        while node is not None:
            node.totalGames += 1
            node.wins += win

            # Probabilities are updated
            # Only works on player nodes because of definition of the game.
            self.updateProbabilities(node)

            node = node.parent


    def simulation(self, node):
        # Use heavy heuristics
        while not node.isLeaf():
            node = node.randGenChild()

        return node

    # Select each node based off its UCB
    def selection(self, curNode):
        # while we have the statistics
        while curNode.allChildrenGenerated() and not len(curNode.children)==0:
            # Select the best children for UCB
            childrenUCB = self.childrenUCB(curNode)
            arg = np.argmax(childrenUCB)
            curNode = curNode.children[arg]

        return curNode

    # Calculate ucb of the children of node
    def childrenUCB(self, node):
        childrenUCB = []
        for child in node.children:
            ucb = child.wins / child.totalGames + 1.6 * math.log(node.totalGames / child.totalGames)
            childrenUCB.append(ucb)
        return childrenUCB
    
    # Update the action-state probabilities assigned to each decision for a node
    # Only works on a node that is a player since that is the only action-state
    # that have different probability distributions
    def updateProbabilities(self, node):
        # If we are the player, we want to makesure that
        # our probabilities update based off our wins
        # We don't need to update for non-player since prob is fixed
        if node.isPlayerDecision:
            # keep track of which probabilities were updated
            probAccum = 0
            notUpdated = [x for x in range(0, 4)]
            for child in node.children:
                prob = child.wins / node.totalGames
                node.stateActProbs[child.opt] = prob
                probAccum += prob
                notUpdated.remove(child.opt)
            
            if len(notUpdated) == 0:
                return

            # Update the rest of the values
            # They should take up the remaining probabilities not assigned
            prob = (1 - probAccum) / len(notUpdated)
            for x in notUpdated:
                node.stateActProbs[x] = prob