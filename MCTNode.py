
############################################
# Project: MCT-TFE
# File: MCTNode.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

from mct_config import *
import numpy as np
import math

class Node:
    option = 0
    parent = None
    total_games = 0
    total_wins = 0
    UCB = 0
    # Value flag used for values.
    val = 0
    grid = None
    children = np.array([])

    # Array to store what options have already been done.
    # Should be at most 128
    children_options = np.array([])

    # Each node must contain a move and new tile appearance option.
    # can also be done to save a snapshot of the grid, but for now, no need.
    def __init__(self, parent, sim, option, grid, board_width, isPlayer):
        self.isPlayer = isPlayer
        self.option = option
        self.parent = parent
        self.sim = sim
        self.board_width = board_width
        # Generate options based off availability
        self.grid = np.copy(grid)
        self.children_options = self.genOpt(grid)

        # If res empty and heuristics is on, try to check what type of leaf.
        self.sim.grid = grid
        if VAL_H:
            if self.children_options.size != 0:
                self.val = np.max(self.valFromGrid(self.grid))
            elif self.sim.isWin():
                self.val = LEAF_WIN_WEIGHT
            else:
                self.val = -LEAF_WIN_WEIGHT

    
    # Returns a grid
    def optToGrid(self, opt):
        self.sim.grid = np.copy(self.grid)
        res = []
            
        if self.isPlayer:
            direc = opt
            self.sim.moveGrid(direc)
            res = self.sim.grid
        else:
            board_size = self.board_width * self.board_width
            val = opt // ( board_size)
            loc = opt - val * board_size
            res = self.sim.grid
            if val == 0:
                res[loc // 4, int(loc % 4)] = 2
            else:
                res[loc // 4, int(loc % 4)] = 4

        return res


    # given current grid, generate some options
    def genOpt(self, grid):
        res = []
        if self.isPlayer:
            # Can save if necessary
            self.sim.grid = np.copy(grid)
            after_grid = self.sim.availDir()

            # Move
            dir = [DIR_KEY[k] for k, v in after_grid.items()]
            # Then generate
            board_size = self.board_width * self.board_width
            for k, v in after_grid.items():
                v_f = v.flatten("K")
                # Multiply to have unique ID for each range.
                # Think of it as an unique array index for each config.
                res += [DIR_KEY[k]]
        else:
            v_f = grid.flatten("K")
            board_size = self.board_width * self.board_width
            res += [0 + y[0] for y in np.argwhere(v_f == 0).tolist()]
            res += [1 * board_size + y[0] for y in np.argwhere(v_f == 0).tolist()]
    
        return np.array(res)
        
    # Generate heuristic value from given grid
    def valFromGrid(self, grid):
        s = 0
        # Try to take monotonically increasing
        res = np.diff(grid.flatten("K"))
        res = np.clip(res, 0, 1)
        s += sum(res)

        # check rows
        res = np.diff(np.rot90(grid,1).flatten("K"))
        res = np.clip(res, 0, 1)
        s += sum(res) * 4
        return [s]
        result = []
        
        for x in range(0, V_DIR):
            v = math.log(np.sum(np.multiply(grid, FILTERS[x]).flatten("K")))
            result.append(v)
        
        return result
        
        

    # Create a new child for the node, if possible.
    # Does a check to see if possible.
    # Returns false if cannot create child.
    # Mutates tfe
    def create_child(self):
        if self.children_options.size == 0:
            return False

        arg = rnd.randint(0, self.children_options.size - 1)
        opt = self.children_options[arg]

        # Delete the option.
        self.children_options = np.delete(self.children_options, arg)
        grid = self.optToGrid(opt)
        result = Node(self, self.sim, opt, grid, self.board_width, not self.isPlayer)

        self.children = np.append(self.children, result)
        return result

    def hasUCB(self):
        return self.total_games != 0

    def UCB(self):
        return self.UCB
