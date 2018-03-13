#import torch
#from torch.autograd import Variable
import numpy as np
from TFE import *
from CNNOHUtil import *

TREE_STATE_PLAYER_BRANCH_COUNT = 4
TREE_STATE_SYSTEM_BRANCH_COUNT = 4

class MCTState:
	def __init__(self, boardState, p):
		self.boardState = boardState
		self.p = p

class MCMC:
	def __init__(self, model):
		self.model = model
	
	def sampleFromMCT(self, initialBoardState, terminationThreshold = 10, p = 1.0):
		self.outputs = []
		self.targets = []
		self.terminationThreshold = terminationThreshold
		self.proceed(MCTState(initialBoardState, p))
		return (torch.cat(self.outputs, 0), Variable(torch.FloatTensor(self.targets)))
	
	def proceed(self, currentState):
		if currentState.boardState.isWin((2 ** self.terminationThreshold) * 2):
			return 1
		if currentState.boardState.isLose():
			return 0
		if np.random.rand() < currentState.p:
			return self.proceedTreeState(currentState)
		else:
			return self.proceedLinearState(currentState)
	
	def proceedTreeState(self, currentState):
		move = ['u', 'd', 'l', 'r']
		netState = convertBoardToNet(currentState.boardState)
		valueVariable = self.model(netState)
		maxTarget = 0
		for i in range(TREE_STATE_PLAYER_BRANCH_COUNT):
			tempboard = currentState.boardState.copy()
			tempboard.moveGrid(move[i])
			target = 0
			for j in range(TREE_STATE_SYSTEM_BRANCH_COUNT):
				temptempboard = tempboard.copy()
				temptempboard.putNew()
				target += self.proceed(MCTState(temptempboard, currentState.p / (TREE_STATE_PLAYER_BRANCH_COUNT * TREE_STATE_SYSTEM_BRANCH_COUNT))) / TREE_STATE_SYSTEM_BRANCH_COUNT
			self.outputs.append(valueVariable[0, i])
			self.targets.append(target)
			maxTarget = max(target, maxTarget)
		return maxTarget
	
	def proceedLinearState(self, currentState):
		move = ['u', 'd', 'l', 'r']
		netState = convertBoardToNet(currentState.boardState)
		valueVariable = self.model(netState)
		i = np.random.randint(TREE_STATE_PLAYER_BRANCH_COUNT)
		tempboard = currentState.boardState.copy()
		tempboard.moveGrid(move[i])
		tempboard.putNew()
		target = self.proceed(MCTState(tempboard, currentState.p))
		self.outputs.append(valueVariable[0, i])
		self.targets.append(target)
		return target