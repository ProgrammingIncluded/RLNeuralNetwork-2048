#import torch
#from torch import nn
import datetime
import sys
from CNNOHUtil import *
from CNNOHMCT import MCMC
sys.setrecursionlimit(10000)

model = nn.Sequential(nn.Conv3d(2, 16, kernel_size=2, stride=1, padding=0), nn.Tanh(),
					  nn.Conv3d(16, 32, kernel_size=2, stride=1, padding=0), nn.Tanh(),
					  Flatten(), nn.Linear(2*2*(MODEL_DEPTH-2)*32, 64), nn.Tanh(), nn.Dropout(), nn.Linear(64, 4))

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.00001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def printTime():
	print(datetime.datetime.now().isoformat())

def genTimeStamp():
	timestamp = ''.join(c for c in datetime.datetime.now().isoformat() if c.isdigit())
	return timestamp

def saveModel(model, filename):
	torch.save(model, filename)

def loadModel(filename):
	model = torch.load(filename)
	return model

def trainOneRound(output, target):
	optimizer.zero_grad()
	loss = loss_fn(output, target)
	print("Training Loss =", loss.data[0])
	loss.backward()
	optimizer.step()

def trainingWorkflow(nRound, threshold, p, goal, batch):
	#model = loadModel('16model_latest.pt')
	model.train(True)
	mcmc = MCMC(model)
	for i in range(nRound):
		print('round', i)
		batch_output = []
		batch_target = []
		winCnt = 0
		gameCnt = 0
		for j in range(batch):
			initBoard = TFE(4)
			# generate a new
			initBoard.putNew()
			initBoard.putNew()
	#		print('threshold', threshold)
			output, target = mcmc.sampleFromMCT(initBoard, goal, threshold, p)
			batch_output.append(output)
			batch_target.append(target)
			winCnt += mcmc.winCnt
			gameCnt += mcmc.gameCnt
		output = torch.cat(batch_output, 0)
		target = torch.cat(batch_target, 0)
		print('winning rate', winCnt / gameCnt)
		print('total', gameCnt, 'games')
#		print(output, target)
		npTarget = target.data.numpy()
		print(np.sum(npTarget), npTarget.size)
		printTime()
		trainOneRound(output, target)
		printTime()
		saveModel(model, 'model' + genTimeStamp() + '.pt')
		saveModel(model, '16model_latest.pt')
		print('')
trainingWorkflow(100, 12, 1, 16, 20)