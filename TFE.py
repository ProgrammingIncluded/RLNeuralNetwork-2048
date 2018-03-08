############################################
# Project: MCT-TFE
# File: TFE.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

import numpy as np
import random as rnd

# Game Settings
# Probability of 4 appearing
FOUR_PROB = 10
MAX_VALUE = 2048

# cannot be changed for now
MOV_OPT = ["d", "u", "l", "r"]

# 2048 Class
class TFE:
    def __init__(self, board_width):
        self.board_width = board_width
        self.grid = np.zeros((self.board_width, self.board_width), np.int64)

    def setGrid(self, grid):
        print(self.board_width)
        self.grid = grid

    # Attempt to put a new number
    def putNew(self):
        grid = self.grid
        zero = np.argwhere(grid == 0)
        if zero.size == 0:
            return False
        
        sel = rnd.randint(0, zero.shape[0] - 1)
        selK = zero[sel, :]
        grid[selK[0], selK[1]] =  2 if rnd.randint(0, 100) > 10 else 4
        return True

    # Move a single cell, merges if possible.
    def moveCell(self, x, y, dir):
        grid = self.grid
        if grid[y, x] == 0:
            return 
        # check boundary case
        if x <= 0 and dir == "l":
            return
        elif x >= (self.board_width - 1) and dir == "r":
            return
        elif y <= 0 and dir == "u":
            return
        elif y >= (self.board_width-1) and dir == "d":
            return

        if dir == "l":
            xval = -1
            yval = 0
            bound = lambda v, u: v >= 0
        elif dir == "r":
            xval = 1
            yval = 0
            bound = lambda v, u: v < self.board_width
        elif dir == "d":
            xval = 0
            yval = 1
            bound = lambda v, u: u < self.board_width
        else:
            xval = 0
            yval = -1
            bound = lambda v, u: u >= 0

        dx = x + xval
        dy = y + yval
        while bound(dx, dy):
            if grid[dy, dx] == 0:
                dx += xval
                dy += yval
            elif grid[dy, dx] == grid[y, x]:
                grid[dy, dx] *= 2
                grid[y, x] = 0
                # all done
                return
            else:
                break
        grid[dy-yval, dx-xval] = grid[y, x]
        if dy-yval != y or dx-xval != x:
            grid[y, x] = 0


    # Move a direction
    def moveGrid(self, dir):
        grid = self.grid
        if dir == "l":
            evalO = lambda v, u: u < self.board_width
            evalI = lambda v, u: v < self.board_width
            x, y = 0, 0
            incI = lambda v, u: (v+1, u)
            incO = lambda v, u: (v, u + 1)
        elif dir == "r":
            evalO = lambda v, u: u >= 0
            evalI = lambda v, u: v >= 0
            x, y = (self.board_width - 1), (self.board_width - 1)
            incI = lambda v, u: (v-1, u)
            incO = lambda v, u: (v, u - 1)
        elif dir == "d":
            evalO = lambda v, u: v >= 0
            evalI = lambda v, u: u >= 0
            x, y = (self.board_width - 1), (self.board_width - 1 )
            incI = lambda v, u: (v, u-1)
            incO = lambda v, u: (v-1, u)
        else:
            evalO = lambda v, u: v < self.board_width
            evalI = lambda v, u: u < self.board_width
            x, y = 0, 0
            incI = lambda v, u: (v, u+1)
            incO = lambda v, u: (v+1, u)

        reset = lambda dx, dy, x, y: (x, dy) if dir == "l" or dir == "r" else (dx, y)
        dx, dy = x, y
        while evalO(dx, dy):
            dx, dy = reset(dx, dy, x, y)
            while evalI(dx, dy):
                self.moveCell(dx, dy, dir)
                dx, dy = incI(dx, dy)
            dx, dy = incO(dx, dy)
                
    def restart(self):
        grid = np.zeros((self.board_width,self.board_width))

    def isWin(self):
        return self.grid.max() >= MAX_VALUE

    # Check if loosing state. Expensive! Calls availDir
    def isLose(self):
        return (len(self.availDir()) == 0)

    # check available directions. Expensive! Takes O(n^2 * 4)
    # Saves a snapshot of each grid. Key and grid.
    def availDir(self):
        choice = ["u", "d", "l", "r"]
        # check if empyt
        if self.grid.max() == 0:
            return {k: np.copy(self.grid) for k in choice}
    
        result = {}
        gridDup = np.copy(self.grid)
        for c in choice:
            self.moveGrid(c)
            if not np.array_equal(self.grid, gridDup):
                result[c] = self.grid
            self.grid = np.copy(gridDup)
        return result