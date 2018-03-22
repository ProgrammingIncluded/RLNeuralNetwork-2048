#import torch
#from torch.autograd import Variable
import numpy as np
from TFE import *
from CNNOHUtil import *

TREE_STATE_PLAYER_BRANCH_COUNT = 4
TREE_STATE_SYSTEM_BRANCH_COUNT = 4

class MCMC:
	def __init__(self, model):
		self.model = model
		self.T = 0
	
	def sampleFromMCT(self, initialBoardState, terminationTarget = 16, terminationThreshold = 10):
		self.outputs = []
		self.targets = []
		self.terminationTarget = terminationTarget
		self.terminationThreshold = terminationThreshold
		self.proceed(initialBoardState))
		return (torch.cat(self.outputs, 0), Variable(torch.LongTensor(self.targets)))
	
	def proceed(self, currentState):
		if currentState.isLose():
			return False
		
		moves = ['l', 'u', 'r', 'd']
		#netState = convertBoardToNet(currentState)
		#valueVariable = self.model(netState)
		#valueNP = valueVariable.data.numpy()
		valueNP = np.zeros((1,4))
		availDirs = currentState.availDir()
		for i, move in enumerate(moves):
			if move not in availDirs:
				valueNP[0,i] = 0
			else:
				for j in range(100):
					valueNP[0,i] += self.sample(currentState)
		i = np.argmax(valueNP)
		currentState.moveGrid(moves[i])
		currentState.putNew()
		print(currentState)
		return True
	
	def sample(self, startState):
		currentState = startState.copy()
		while not currentState.isLose():
			moves = ['l', 'u', 'r', 'd']
			netState = convertBoardToNet(currentState)
			valueVariable = self.model(netState)
			valueNP = valueVariable.data.numpy()
			availDirs = currentState.availDir()
			for i, move in enumerate(moves):
				if move not in availDirs:
					valueNP[0,i] = 0
				else:
					valueNP[0,i] += 0.25
			valueNP /= np.sum(valueNP)
			i = np.random.choice(4, valueNP)
			currentState.moveGrid(moves[i])
			currentState.putNew()
		return currentState.score()
	
	def proceedTreeState(self, currentState, d):
		moves = ['u', 'd', 'l', 'r']
		netState = convertBoardToNet(currentState.boardState)
		valueVariable = self.model(netState)
		maxTarget = 0
		sumTarget = 0
		cnt = 0
		tie = []
		availDirs = currentState.boardState.availDir()
		for i, move in enumerate(moves):
			if move not in availDirs:
				pass
#				self.outputs.append(valueVariable[0, i])
#				self.targets.append(0)
			else:
				tempboard = currentState.boardState.copy()
				tempboard.moveGrid(move)
				target = 0
				for j in range(TREE_STATE_SYSTEM_BRANCH_COUNT):
					temptempboard = tempboard.copy()
					temptempboard.putNew()
					target += self.proceed(MCTState(temptempboard, currentState.p / (TREE_STATE_PLAYER_BRANCH_COUNT * TREE_STATE_SYSTEM_BRANCH_COUNT)), d) / TREE_STATE_SYSTEM_BRANCH_COUNT
#				self.outputs.append(valueVariable[0, i])
#				self.targets.append(np.log2(target))
				if target == maxTarget:
					tie.append(i)
				if target > maxTarget:
					maxTarget = target
					tie = [i]
				cnt += 1
				sumTarget += target
		self.outputs.append(valueVariable)
		self.targets.append(tie[np.random.randint(len(tie))])
		return sumTarget / cnt
	
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
#		self.outputs.append(valueVariable[0, i])
#		self.targets.append(np.log2(target))
		return target