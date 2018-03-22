import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
from TFE import *

MODEL_DEPTH = 16

class Flatten(nn.Module):
	def forward(self, x):
		x = x.view(x.size()[0],-1)
		return x

class BottleNeck(nn.Module):
	def __init__(self, numChannel, submodule):
		super(BottleNeck, self).__init__()
		self.submodule = submodule
		self.conv = nn.Conv1d(numChannel, numChannel, kernel_size=4, stride=2, padding=1)
		self.pool = nn.MaxPool1d(kernel_size=2)
		
	def forward(self, x):
		return F.relu(self.pool(self.submodule(x)) + self.conv(x))

class ResModule(nn.Module):
	def __init__(self, numChannel):
		super(ResModule, self).__init__()
		self.conv1 = nn.Conv1d(numChannel, numChannel, kernel_size=3, padding=1, bias=False)
		self.conv2 = nn.Conv1d(numChannel, numChannel, kernel_size=3, padding=1, bias=False)
	
	def forward(self, x):
		return F.relu(x+self.conv2(F.relu(self.conv1(x))))

def convertBoardToNet(board):
	grid = board.grid
	layers = []
	for k in range(16):
		occ = np.zeros((4,4))
		reg = np.zeros((4,4))
		hot = np.zeros((4,4))
		lvl = np.zeros((4,4))
		for i in range(4):
			for j in range(4):
				if grid[i, j] != 0:
					v = np.round(np.log2(grid[i, j]) - 1).astype(int)
					occ[i, j] = 1
					reg[i, j] = v + 1
					if v == k:
						hot[i, j] = 1
						lvl[i, j] = v + 1
		layers.append(np.concatenate((np.asarray([[[k + 1]]]), occ.reshape((1, 16, 1)), reg.reshape((1, 16, 1)), hot.reshape((1, 16, 1)), lvl.reshape((1, 16, 1))), axis=1))
	return Variable(torch.from_numpy(np.concatenate(layers, axis=2)).type(torch.FloatTensor), requires_grad=False)