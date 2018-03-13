import torch
from torch import nn
from torch.autograd import Variable
from TFE import *

MODEL_DEPTH = 11

class Flatten(nn.Module):
	def forward(self, x):
		x = x.view(x.size()[0],-1)
		return x

def convertBoardToNet(board):
	width = board.board_width
	grid = board.grid
	tensorNP = np.zeros((1, 2, width, width, MODEL_DEPTH))
	for i in range(width):
		for j in range(width):
			if grid[i, j] != 0:
				tensorNP[0, 0, i, j, np.round(np.log2(grid[i, j]) - 1).astype(int)] = 1
				tensorNP[0, 0, i, j, :] = 1
	return Variable(torch.from_numpy(tensorNP).type(torch.FloatTensor), requires_grad=False)