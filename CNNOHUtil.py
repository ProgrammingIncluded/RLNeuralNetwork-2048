import torch
from torch import nn
from torch.autograd import Variable
from TFE import *

MODEL_DEPTH = 16

class Flatten(nn.Module):
	def forward(self, x):
		x = x.view(x.size()[0],-1)
		return x

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