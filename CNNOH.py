#import torch
#from torch import nn
import datetime
import sys
from CNNOHUtil import *
from CNNOHMCT import MCMC
sys.setrecursionlimit(10000)

model = nn.Sequential(nn.Conv1d(65, 64, kernel_size=3, padding=1), nn.ReLU(),
					  BottleNeck(64, nn.Sequential(ResModule(64), ResModule(64), ResModule(64), ResModule(64), ResModule(64))),
					  BottleNeck(64, nn.Sequential(ResModule(64), ResModule(64), ResModule(64), ResModule(64), ResModule(64))),
					  Flatten(), nn.Linear(256, 32), nn.Tanh(), nn.Dropout(), nn.Linear(32, 4))

loss_fn = nn.CrossEntropyLoss()
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

def trainingWorkflow(nRound, batch):
	model = loadModel('pretrained_best')
	model.train(True)
	#model.cpu()
	mcmc = MCMC(model)
	for i in range(nRound):
		print('round', i)
		batch_output = []
		batch_target = []
		for j in range(batch):
			initBoard = TFE(4)
			# generate a new
			initBoard.putNew()
			initBoard.putNew()
	#		print('threshold', threshold)
			output, target = mcmc.sampleFromMCT(initBoard)
			batch_output.append(output)
			batch_target.append(target)
		output = torch.cat(batch_output, 0)
		target = torch.cat(batch_target, 0)
		printTime()
		#trainOneRound(output, target)
		printTime()
		#saveModel(model, 'model' + genTimeStamp() + '.pt')
		#saveModel(model, '16model_latest.pt')
		print('')
trainingWorkflow(1, 1)