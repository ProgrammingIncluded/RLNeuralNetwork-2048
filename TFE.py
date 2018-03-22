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
	@staticmethod
	def fromBits(v):
		ret = TFE(4)
		ret.grid = np.zeros((ret.board_width, ret.board_width), np.int64)
		for i in [3,2,1,0]:
			for j in [3,2,1,0]:
				if v % 16 != 0:
					ret.grid[i,j] = 2 ** (v % 16)
				v = v // 16
		return ret
		
	def __init__(self, board_width):
		self.board_width = board_width
		self.grid = np.zeros((self.board_width, self.board_width), np.int64)
	
	def setGrid(self, grid):
		print(self.board_width)
		self.grid = grid
	
	def copy(self):
		ret = TFE(self.board_width)
		ret.grid = self.grid.copy()
		return ret

	def eqStates(self, d):
		ret = []
		flip = self.copy()
		flip.grid = np.fliplr(flip.grid)
		if d == 0 or d == 2:
			df = 2 - d
		else:
			df = d
		for i in range(4):
			temp = self.copy()
			temp.grid = np.rot90(temp.grid, 3 * i)
			ret.append((temp, (d + i) % 4))
			#print(temp.grid, (d + i) % 4)
			temp = flip.copy()
			temp.grid = np.rot90(temp.grid, 3 * i)
			ret.append((temp, (df + i) % 4))
			#print(temp.grid, (df + i) % 4)
		return ret
		
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
		if dir == "l" or dir == 0:
			evalO = lambda v, u: u < self.board_width
			evalI = lambda v, u: v < self.board_width
			x, y = 0, 0
			incI = lambda v, u: (v+1, u)
			incO = lambda v, u: (v, u + 1)
		elif dir == "r" or dir == 2:
			evalO = lambda v, u: u >= 0
			evalI = lambda v, u: v >= 0
			x, y = (self.board_width - 1), (self.board_width - 1)
			incI = lambda v, u: (v-1, u)
			incO = lambda v, u: (v, u - 1)
		elif dir == "d" or dir == 3:
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

	def isWin(self, max_value = MAX_VALUE):
		return self.grid.max() >= max_value

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
	
	def score(self):
		ret = 0
		for i in range(4):
			for j in range(4):
				if self.grid[i, j] > 0:
					ret += np.log2(self.grid[i, j]) * self.grid[i, j]
		return ret