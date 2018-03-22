import datetime
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

from CNNOHUtil import *
from TFE import TFE

dtype = torch.FloatTensor
timestamp = ''.join(c for c in datetime.datetime.now().isoformat() if c.isdigit())

data = []
target = []
with open("input.txt", "r") as ins:
	cnt = 0
	for line in ins:
		cnt += 1
		if cnt % 1000 == 0:
			print("line number = ", cnt)
		a = line.split()

		for board, direction in TFE.fromBits(int(a[0])).eqStates(int(a[1])):
			data.append(convertBoardToNet(board))
			target.append(direction)
data = torch.cat(data, 0).data.numpy()
target = torch.LongTensor(target).numpy()
print(data.shape, target.shape)

hold_out_size = 500
bits = np.concatenate((np.ones((hold_out_size)), np.zeros((data.shape[0] - hold_out_size))))
np.random.shuffle(bits)

train_data = data[np.where(bits == 0)]
train_target = target[np.where(bits == 0)]
holdout_data = data[np.where(bits == 1)]
holdout_target = target[np.where(bits == 1)]

sample_size = 500
bits = np.concatenate((np.ones((sample_size)), np.zeros((train_data.shape[0] - sample_size))))
np.random.shuffle(bits)

train_sample_data = train_data[np.where(bits == 1)]
train_sample_target = train_target[np.where(bits == 1)]

# Create torch variable
train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_target = torch.from_numpy(train_target).type(torch.LongTensor)
holdout_data = torch.from_numpy(holdout_data).type(torch.FloatTensor)
holdout_target = torch.from_numpy(holdout_target).type(torch.LongTensor)
train_sample_data = torch.from_numpy(train_sample_data).type(torch.FloatTensor)
train_sample_target = torch.from_numpy(train_sample_target).type(torch.LongTensor)

class BottleNeck(nn.Module):
	def __init__(self, numChannel, submodule):
		super(BottleNeck, self).__init__()
		self.submodule = submodule
		self.conv = nn.Conv1d(numChannel, numChannel, kernel_size=4, stride=2, padding=1)
		self.bn = nn.BatchNorm1d(numChannel)
		self.pool = nn.MaxPool1d(kernel_size=2)
		
	def forward(self, x):
		return F.relu(self.pool(self.submodule(x)) + self.bn(self.conv(x)))

class ResModule(nn.Module):
	def __init__(self, numChannel):
		super(ResModule, self).__init__()
		self.conv1 = nn.Conv1d(numChannel, numChannel, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(numChannel)
		self.conv2 = nn.Conv1d(numChannel, numChannel, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(numChannel)
	
	def forward(self, x):
		return F.relu(x+self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

batch_size = 128

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_data, train_target), batch_size=batch_size, shuffle=True)

"""ResNet"""
model = nn.Sequential(nn.Conv1d(65, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
					  BottleNeck(64, nn.Sequential(ResModule(64), ResModule(64), ResModule(64), ResModule(64), ResModule(64))),
					  BottleNeck(64, nn.Sequential(ResModule(64), ResModule(64), ResModule(64), ResModule(64), ResModule(64))),
					  Flatten(), nn.Linear(256, 32), nn.Tanh(), nn.Dropout(), nn.Linear(32, 4))

model.cuda()

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

hist_train_loss = []
hist_train_accuracy = []
hist_holdout_loss = []
hist_holdout_accuracy = []

for epoch in range(50):
	for i, (data, target) in enumerate(train_loader):
		data = Variable(data.cuda(), requires_grad=False)
		target = Variable(target.cuda(), requires_grad=False)
		optimizer.zero_grad()
		output = model(data)
		loss = loss_fn(output, target)
		loss.backward()
		optimizer.step()
	
	print("Epoch", epoch)
	train_sample_output = model(Variable(train_sample_data.cuda(), requires_grad=False))
	train_loss = loss_fn(train_sample_output, Variable(train_sample_target.cuda(), requires_grad=False))
	print("Training Loss =", train_loss.data[0])
	
	_, predicted = torch.max(train_sample_output, 1)
	correct = (predicted.data.cpu() == train_sample_target).sum()
	total = len(train_sample_target)
	train_accuracy = correct / total
	print("Training Accuracy =", 100 * train_accuracy)
	
	holdout_output = model(Variable(holdout_data.cuda(), requires_grad=False))
	holdout_loss = loss_fn(holdout_output, Variable(holdout_target.cuda(), requires_grad=False))
	print("Holdout Loss =", holdout_loss.data[0])
	
	_, predicted = torch.max(holdout_output, 1)
	correct = (predicted.data.cpu() == holdout_target).sum()
	total = len(holdout_target)
	holdout_accuracy = correct / total
	print("Holdout Accuracy =", 100 * holdout_accuracy)
	print()
	torch.save(model, timestamp)
