############################################
# Project: MCT-TFE
# File: MCT.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

import TFE as tfe
from MCTNode import *
from mct_config import *

import numpy as np
import random as rnd
import math
import time

class MCT:

    def __init__(self, board_width, valueFunc, policyUpdate):
        if board_width == 0:
            self.board_width = 4
        else:
            self.board_width = board_width
        self.sim = tfe.TFE(board_width)
        # Set the neural network for the q-value.
        self.valueFunc = valueFunc
        # Tracking neural network nodes ran on
        self.nnTracker = []
        self.policyUpdate = policyUpdate

    # Greedy algorithm for faster approach
    # Select the direction with highest direct yield
    def greedy(self, tfe):
        res = tfe.availDir()
        scores = []
        dir_save = []
        for k, v in res.items():
            v = v.flatten("K")
            res = [0 if x==0 else math.log(x, 2) for x in v]
            scores.append(sum(res))
            dir_save.append(k)
        
        arg = np.argmax(scores)
        return dir_save[arg]

    # Run the AI
    # Returns best direction to select next.
    def run(self, tfe, sec, noNone = False):
        global GREEDY_CONTROL
        # Greedy algorithm control flow
        if GREEDY_CONTROL and np.argwhere(tfe.grid.flatten() == 0).size > GREEDY_THRESH:
            return self.greedy(tfe)

        # Check if we should run greedy only once.
        GREEDY_CONTROL = GREEDY_CONTROL & ~GREEDY_INIT_ONLY 

        # Root node setting up.
        root = Node(None, self.sim, -1, tfe.grid, self.board_width, True, self.valueFunc)
        # print("\nBRANCHES: " + str(root.children_options.size))
        
        t_end = time.time() + sec
        # Monte Carlo loop
        while time.time() <= t_end or noNone:
            trav = []
            # gurnatees us a leaf node unless we run out of time.
            cur_node = self.forwardPropagate(root,trav, t_end, noNone)

            # End if time is ended. However if noNone, we need to gurantee that
            # at least first layer has all options available.
            # So we either exited because time and all children solved.
            if time.time() > t_end and not noNone:
                    break

            # Check if leaf is win or lose state. Propagate accordingly.
            self.sim.grid = cur_node.grid
            if self.sim.isWin():
                self.backPropagate(trav, 1)
            else:
                self.backPropagate(trav, 0)
            
            # check if all children for root, if so, we can set noNone to false.
            if root.children_options.size == 0:
                noNone = False

        # Select a result
        # Check makesure we have all children
        # Otherwise not enough time for first level
        if root.children_options.size != 0:
            print("NO TIME")
            return None

        highest = self.getHighestUCB(root.children)
        opt = highest.option
        direc = opt

        # Now we need to do backproping.
        # Prep data
        dataset = []
        for x in self.nnTracker:
            dataset.append((x.grid, DIR_VAL[x.option], x.UCB, x.val, x))

        self.policyUpdate(dataset)

        # Remove all data
        self.nnTracker = []

        return DIR_VAL[direc]

    def forwardPropagate(self, root, trav, t_end, noNone = False):
        cur_node = root

        # No need to check if leaf node here. The inner loop will take care of that.
        # check time again in inner loop
        # either error because node with no children even though no win
        # or reached leaf and have lost/won. Either way, leave loop.
        while time.time() <= t_end or (noNone):

            # Check win node.
            if self.sim.isWin():
                break

            # Try to create a child if possible, if not, two possibilities.
            res = cur_node.create_child()


            # We create all the children, go down one level.
            # OR we have no children.
            if res == False:
                # check if leaf
                if cur_node.children.size == 0 and cur_node.children_options.size == 0:
                    # Force noNone to false so we can escape loop, since we reached leaf.
                    noNone = False
                    break

                # Note a leaf node, let us go down via UCB
                # save snapshot
                trav.append(cur_node)
                # Go down the highest ucb.
                cur_node = self.getHighestUCB(cur_node.children)
            else:
                # otherwise, we have created a child

                # Here, we are trying to keep track of which nodes we have ran our NN.
                if cur_node.isPlayer:
                    self.nnTracker.append(res)

                trav.append(res)
                cur_node = res
            self.sim.grid = cur_node.grid

        # Save last node
        trav.append(cur_node)
        return cur_node


    def backPropagate(self, trav, win):
        # Backpropagate
        v_accum = 0
        while len(trav) != 0:
            n = trav.pop()
            n.total_games += 1
            n.total_wins += win

            # accum the heuristics.
            v_accum += n.val

            # calculate ucb
            if n.parent != None:
                # Basic UCB
                n.UCB  = 1.6 * math.sqrt(math.log(n.parent.total_games + 1) / n.total_games) 

                # Heuristics update. Note the accum
                if VAL_H:
                    n.val = v_accum
                    n.UCB += v_accum + n.guess_val
                else:
                    n.UCB += (n.total_wins /n.total_games) + n.guess_val
                
                # update siblings as well
                if VAL_H:
                    # v is not updated in siblings, only parent totals.
                    for s in n.parent.children:
                        s.UCB = s.guess_val + s.val + 1.6 * math.sqrt(math.log(s.parent.total_games + 1) / s.total_games)
                else:
                    for s in n.parent.children:
                        s.UCB = s.guess_val + (s.total_wins /s.total_games) + 1.6 * math.sqrt(math.log(s.parent.total_games + 1) / s.total_games)

    def getHighestUCB(self, children):
        # Check UCB, select highest UCB
        sel = None
        sel_val = float("-inf")
        parent = None
        if children.size != 0:
            parent = children[0].parent

        for child in children:
            # We only operate on neural net if we are the player.
            if parent.isPlayer:
                val = child.UCB + parent.children_action[DIR_VAL[child.option]]
            else:
                val = child.UCB
            if sel_val < val:
                sel = child
                sel_val = val
        
        return sel