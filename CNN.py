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
        #
        self.hidden1 = nn.Linear(inputDim, inputDim)
        self.hidden2 = nn.Linear(inputDim, inputDim)
        self.hidden3 = nn.Linear(inputDim, inputDim)
        self.hidden4 = nn.Linear(inputDim, NUM_DIR + VALUE_STATE)
        # self.softmax = nn.Softmax(dim=0)
    def foward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)

        x = self.hidden2(x)
        x = F.relu(x)

        x = self.hidden3(x)
        x = F.relu(x)

        x = self.hidden4(x)
        x = F.relu(x)
        # x_stateActionProbabilities = self.softmax(x[0,0:4])
        # x_stateValue = x[0,4]
        # x = torch.unsqueeze(torch.cat((x_stateActionProbabilities,x_stateValue),0),0)
        return x
