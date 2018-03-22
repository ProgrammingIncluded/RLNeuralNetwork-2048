############################################
# Project: MCT-TFE
# File: MCT.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

from MCTNode import *
from mct_config import *
import time
import math
import operator
import numpy as np
import random

# Try a simpler MCT
class MCTEZ:
    def __init__(self, secondsPerMove, trainer):
        # Number of seconds you can make per move.
        self.secondsPerMove = secondsPerMove

        # Tracking stats
        self.gamesPlayed = 0
        self.gamesWon = 0
        self.probs = {}
        self.expectedDir = {}
        self.expected = 0
        # Accumulated max values
        self.accum = 0
        self.trainer = trainer
        

    # Make a player move
    def playerDecision(self, game):
        self.curGameState = game
        # Keep stats for root node.
        self.gamesPlayed = 0
        self.gamesWon = 0

        self.expected = 0
        self.accum = 0

        # Keep states for action-state probs
        wonDir = {k: 0 for k,v in DIR_KEY.items()}
        accumDir = {k: 0 for k,v in DIR_KEY.items()}

        # Our time loop
        endTime = time.time() + self.secondsPerMove
        availDir = self.curGameState.availDir()
        while time.time() < endTime:
            # Sample once from each game
            # We want to first manually go down one level if possible
            for dirc, val in availDir.items():
                # Copy the game state
                tempGame = self.curGameState.copy()

                # Force down one path before we do full rollout
                tempGame.moveGrid(dirc)

                # Generate a grid
                tempGame.putNew()

                # Perform a full rollout now.
                res, val = self.rolloutOnPlayerDecision(tempGame)
                # Update accumulator
                self.accum += val
                accumDir[dirc] += val

                # Increase stats if we won
                if res == 1:
                    self.gamesPlayed += 1
                    wonDir[dirc] += 1

                self.gamesPlayed += 1
        
        # Calculate the probabilities and stats
        # TODO: Use these probs for training
        self.probs = {k: v / self.gamesPlayed for k, v in wonDir.items()}
        # Caculate the expected value for each child node
        self.expectedDir = {k: v / self.gamesPlayed for k, v in accumDir.items()}
        self.expected = self.accum / self.gamesPlayed

        # Pass it to the trainer
        self.trainer([(self.expectedDir, self.expected, self.curGameState)])

        print(self.expectedDir)

        # Return largest direction with highest prob
        best = max(self.expectedDir.items(), key=operator.itemgetter(1))[0]

        # If max is zero then everything is zero, we rather just random guess
        if self.expectedDir[best] == 0:
            return DIR_VAL[random.randint(0, 3)]
        
        return best

    # Just pass in the game state
    def adversaryDecision(self, game):
        self.curGameState = game.copy()
    
    # Simulate a full roll-out game
    # Assume game given is on player decision
    # Returns 1 if game won, 0 otherwise, also returns max for the rollout
    def rolloutOnPlayerDecision(self, game):
        curState = game.copy()

        availDir = curState.availDir()
        won = curState.isWin()
        lost = len(availDir) == 0
        while not lost:
            dirc = random.choice(list(availDir.keys()))
            
            # Check to see if direction is proper
            curState.moveGrid(dirc)
            curState.putNew()

            availDir = curState.availDir()
            won = curState.isWin()
            lost = len(availDir) == 0
        
        return (1 if won else 0, curState.grid.sum())


# This monte carlo implementation assumes MCT is retained every move.
class MCTZero:
    def __init__(self, game, secondsPerMove, genValFunc, policyUpdateFunc):
        # Create a new root
        self.root = MCTNode(None, game, -1, True)
        self.game = game
        self.secondsPerMove = secondsPerMove
        self.genValFunc = genValFunc
        self.policyUpdateFunc = policyUpdateFunc
        self.tracker = []
    
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

        self.tracker.append(self.root)

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
                # Add to tracker
                self.tracker.append(genNode)
                # Estimate q
                genNode.guessProbs, genNode.guessQ = self.genValFunc(genNode.game.grid)

                curNode = self.simulation(genNode)

            # Backpropagate result
            self.backpropagate(curNode)

            # Also backpropogate what we learned
            actions = []
            for t in self.tracker:
                # Only backprop for player decisions
                if t.isPlayerDecision:
                    actions.append((t.stateActProbs, t.wins / t.totalGames, t))
            # Call function to train NN
            self.policyUpdateFunc(actions)
            # Reset trackers 
            self.tracker = []
                
        
        # Times up! Time to make a decision
        # Pick the one with the highest UCB
        ucbs = self.childrenUCB(self.root)
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
            
            self.tracker.append(node)
            # Estimate q
            node.guessProbs, node.guessQ = self.genValFunc(node.game.grid)

        return node

    # Select each node based off its UCB
    def selection(self, curNode):
        # while we have the statistics
        while curNode.allChildrenGenerated() and not len(curNode.children)==0:
            # Select the best children for UCB
            childrenUCB = self.childrenUCB(curNode)
            arg = np.argmax(childrenUCB)
            curNode = curNode.children[arg]

            self.tracker.append(curNode)
            curNode.guessProbs, curNode.guessQ = self.genValFunc(curNode.game.grid)

        return curNode

    # Calculate ucb of the children of node
    def childrenUCB(self, node):
        childrenUCB = []
        for child in node.children:
            bias = ((child.wins / child.totalGames) + child.guessQ) / 2
            ucb =  bias + 1.6 * math.log(node.totalGames / child.totalGames)
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