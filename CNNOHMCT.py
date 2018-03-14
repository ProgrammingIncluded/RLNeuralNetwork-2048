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
		self.totalReward = 0
		self.gameCnt = 0
		self.outputs = []
		self.targets = []
		self.terminationThreshold = terminationThreshold
		self.proceed(MCTState(initialBoardState, p))
		return (torch.cat(self.outputs, 0), Variable(torch.FloatTensor(self.targets)))
	
	def proceed(self, currentState, d = 0):
		'''
		if currentState.boardState.isWin((2 ** self.terminationThreshold) * 2):
			self.winCount += 1
			return 1
		'''
#		print(d)
#		print(currentState.boardState.grid)
		if currentState.boardState.isLose():
			target = currentState.boardState.grid.max().astype(float)
			self.totalReward += target
			self.gameCnt += 1
			return target
		if np.random.rand() < currentState.p:
			return self.proceedTreeState(currentState, d + 1)
		else:
			return self.proceedLinearState(currentState, d + 1)
	
	def proceedTreeState(self, currentState, d):
		moves = ['u', 'd', 'l', 'r']
		netState = convertBoardToNet(currentState.boardState)
		valueVariable = self.model(netState)
		maxTarget = 0
		availDirs = currentState.boardState.availDir()
		for i, move in enumerate(moves):
			if move not in availDirs:
				self.outputs.append(valueVariable[0, i])
				self.targets.append(0)
			else:
				tempboard = currentState.boardState.copy()
				tempboard.moveGrid(move)
				target = 0
				for j in range(TREE_STATE_SYSTEM_BRANCH_COUNT):
					temptempboard = tempboard.copy()
					temptempboard.putNew()
					target += self.proceed(MCTState(temptempboard, currentState.p / (TREE_STATE_PLAYER_BRANCH_COUNT * TREE_STATE_SYSTEM_BRANCH_COUNT)), d) / TREE_STATE_SYSTEM_BRANCH_COUNT
				self.outputs.append(valueVariable[0, i])
				self.targets.append(np.log2(target))
				maxTarget = max(target, maxTarget)
		return maxTarget
	
	def proceedLinearState(self, currentState, d):
		moves = ['u', 'd', 'l', 'r']
		netState = convertBoardToNet(currentState.boardState)
		valueVariable = self.model(netState)
		valueNP = valueVariable.data.numpy()
		availDirs = currentState.boardState.availDir()
		for i, move in enumerate(moves):
			if move not in availDirs:
				valueNP[0,i] = -1e9
		i = np.argmax(valueNP)
		tempboard = currentState.boardState.copy()
		tempboard.moveGrid(moves[i])
		tempboard.putNew()
		target = self.proceed(MCTState(tempboard, currentState.p), d)
		self.outputs.append(valueVariable[0, i])
		self.targets.append(np.log2(target))
		return target