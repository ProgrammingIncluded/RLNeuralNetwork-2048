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
	
	def sampleFromMCT(self, initialBoardState):
		self.outputs = []
		self.targets = []
		while self.proceed(initialBoardState):
			pass
		return ([], [])
		#return (torch.cat(self.outputs, 0), Variable(torch.LongTensor(self.targets)))
	
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
				valueNP[0,i] = self.sample(currentState, 20)
		i = np.argmax(valueNP)
		currentState.moveGrid(moves[i])
		currentState.putNew()
		print(currentState.grid)
		return True
	
	def sample(self, startState, size):
		moves = ['l', 'u', 'r', 'd']
		ret = 0;
		currentStates = []
		for i in range(size):
			currentStates.append(startState.copy())
		while len(currentStates) > 0:
			netStates = []
			nextStates = []
			for currentState in currentStates:
				netStates.append(convertBoardToNet(currentState))
			netStates = torch.cat(netStates, 0).cuda()
			valueVariable = self.model(netStates)
			valueNPs = valueVariable.data.cpu().numpy()
			for i, currentState in enumerate(currentStates):
				valueNP = valueNPs[i]
				availDirs = currentState.availDir()
				valueNP = np.exp(valueNP)
				valueNP /= np.sum(valueNP)
				for i, move in enumerate(moves):
					if move not in availDirs:
						valueNP[i] = 0
					else:
						valueNP[i] += 0
				valueNP /= np.sum(valueNP)
				i = np.random.choice(4, p=valueNP)
				currentState.moveGrid(moves[i])
				currentState.putNew()
				if currentState.isLose():
					ret += currentState.score()
				else:
					nextStates.append(currentState)
			currentStates = nextStates
		print(ret)
		return ret
	
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