#import torch
#from torch import nn
from CNNOHUtil import *
from CNNOHMCT import MCMC

model = nn.Sequential(nn.Conv3d(2, 8, kernel_size=2, stride=1, padding=0), nn.ReLU(),
					  nn.Conv3d(8, 16, kernel_size=2, stride=1, padding=0), nn.ReLU(),
					  Flatten(), nn.Linear(2*2*(MODEL_DEPTH-2)*16, 64), nn.Tanh(), nn.Dropout(), nn.Linear(64, 4))

loss_fn = nn.MSELoss()
learning_rate = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def trainOneRound(output, target):
	optimizer.zero_grad()
	loss = loss_fn(output, target)
	loss.backward()
	optimizer.step()

def trainingWorkflow(nRound, threshold, p):
	mcmc = MCMC(model)
	for i in range(nRound):
		initBoard = TFE(4)
		# generate a new
		initBoard.putNew()
		initBoard.putNew()
		
		output, target = mcmc.sampleFromMCT(initBoard, threshold, p)
		trainOneRound(output, target)

trainingWorkflow(1, 7, 1)