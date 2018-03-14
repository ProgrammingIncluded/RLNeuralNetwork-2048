#import torch
#from torch import nn
import datetime
import sys
from CNNOHUtil import *
from CNNOHMCT import MCMC
sys.setrecursionlimit(10000)

model = nn.Sequential(nn.Conv3d(2, 8, kernel_size=2, stride=1, padding=0), nn.ReLU(),
					  nn.Conv3d(8, 16, kernel_size=2, stride=1, padding=0), nn.ReLU(),
					  Flatten(), nn.Linear(2*2*(MODEL_DEPTH-2)*16, 64), nn.Tanh(), nn.Dropout(), nn.Linear(64, 4))

loss_fn = nn.MSELoss()
learning_rate = 0.0001

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

def trainingWorkflow(nRound, threshold, p):
	#model = loadModel('model_latest.pt')
	mcmc = MCMC(model)
	for i in range(nRound):
		initBoard = TFE(4)
		# generate a new
		initBoard.putNew()
		initBoard.putNew()
		
		print('round', i)
#		print('threshold', threshold)
		output, target = mcmc.sampleFromMCT(initBoard, threshold, p)
		print('average score', mcmc.totalReward / mcmc.gameCnt)
		print('total', mcmc.gameCnt, 'games')
		npTarget = target.data.numpy()
		print(np.sum(npTarget), npTarget.size)
		printTime()
		trainOneRound(output, target)
		printTime()
		
		saveModel(model, 'model' + genTimeStamp() + '.pt')
		saveModel(model, 'model_latest.pt')
		print('')
trainingWorkflow(100, 6, 0.5)