# Convolutional Neural Network
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

NUM_DIR = 4
VALUE_STATE = 1

class CNN(nn.Module):

    def __init__(self, inputDim):
        super(CNN, self).__init__()
        self.hidden = nn.Linear(inputDim, NUM_DIR + VALUE_STATE)

    def foward(self, x):
        x = self.hidden(x)
        x = F.sigmoid(x)
        return x