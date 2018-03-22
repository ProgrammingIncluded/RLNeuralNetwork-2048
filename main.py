############################################
# Project: MCT-TFE
# File: main.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

import TFE as tfet
import mct_config as mctconfig
from MCT import *
import time
import random as rnd
import sys
import random
import CNN as cnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
import torch

# All the lovely neural network stuff goes here.
BOARD_WIDTH = 4
NN = cnn.CNN(BOARD_WIDTH * BOARD_WIDTH)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(NN.parameters(), lr=0.1)

# Function to generate tuples of size two:
# (
#    UCB to next state taking direction i,
#    current node value
# )
# Input is specified NxN play field.
# UCB should be in dictionary form where
# keys are shown in DIR_KEY within mct_config

def generateValue(grid):
    # Convert numpy grid to input for NN.
    input = torch.from_numpy(grid.flatten())

    # Get the batch shape
    input = input.unsqueeze(0)
    output = NN(Variable(input).float())
    output = output.squeeze(0)

    # Get results into numpy
    output = output.data.numpy()

    # Prepare for output
    stateActionProbabilities = {i:output[i] for i in range(4)}
    stateValue = output[-1]
    result = (stateActionProbabilities,stateValue)

    return result

# Function called for backprop. Arguments are archived list of actions
# queuedActions is an array of (action-state probs, state value, node)
def policyUpdate(actions):
    # Again, run the neural network, this is lazy coding
    # We back prop this time.
    for v in actions:
        inVect = torch.from_numpy(v[2].game.grid.flatten())

        # Get the batch shape
        inVect = inVect.unsqueeze(0)
        results = NN.foward(Variable(inVect).float())

        # Prepare training.
        resultsCopy = torch.FloatTensor([0,0,0,0,0])
        # If probs does not have value, should be zero

        # Normalize the action-state probs
        probs = v[0]
        for key,val in probs.items():
            resultsCopy[key] = val

        # copy down whatever we have for the state value
        resultsCopy[4] = v[1]

        loss = criterion(results, Variable(resultsCopy))
        loss.backward()
        optimizer.step()

# Trainer is a function designed specifically for MCTEZ
# Passes in a list of tuples:
# (normalizedScoresPerDirection, gamesWon/gamesPlayed, grid)
input_list = []
target_list = []

def trainer(values):
    input_list.append(np.array(values[0][2].grid.flatten()))
    target_list.append(np.append(list(values[0][0].values()),values[0][1]))

def train():
    inputs = np.array([input for input in input_list])
    targets = np.array([target for target in target_list])

    numSamples = inputs.shape[0]
    val_idx = np.random.choice(numSamples,int(numSamples/10))
    train_idx = np.delete(np.arange(numSamples),val_idx)

    print(inputs[:,0:5])
    print(targets[:,0:5])

    inputs_train = inputs[train_idx,:]
    inputs_val = inputs[val_idx,:]
    targets_train = targets[train_idx,:]
    targets_val = targets[val_idx,:]

    dataset_train = TensorDataset(torch.FloatTensor(inputs_train),torch.FloatTensor(targets_train))
    dataset_val = TensorDataset(torch.FloatTensor(inputs_val),torch.FloatTensor(targets_val))

    dataset_sizes = {}
    dataset_sizes['train'] = inputs_train.shape[0]
    dataset_sizes['val'] = inputs_val.shape[0]

    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset_train,batch_size=32,shuffle=True)
    dataloaders['val'] = DataLoader(dataset_val,batch_size=32,shuffle=True)

    model = NN.cuda()

    num_epochs = 1000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train','val']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)

            running_loss = 0.0

            for data in dataloaders[phase]:
                inputs, labels = data
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if(phase=='train'):
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0] * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, "n/a"))
            
    model = NN.cpu()
    #
    # for values in values_list:
    #     for v in values:
    #         NN.zero_grad()
    #         inVect = torch.from_numpy(v[2].grid.flatten())
    #
    #         # Get the batch shape
    #         inVect = inVect.unsqueeze(0)
    #         results = NN.foward(Variable(inVect).float())
    #
    #         # Prepare training.
    #         resultsCopy = torch.FloatTensor([0,0,0,0,0])
    #         # If probs does not have value, should be zero
    #
    #         # Normalize the action-state probs
    #         probs = v[0]
    #         for key,val in probs.items():
    #             resultsCopy[DIR_KEY[key]] = val
    #
    #         # copy down whatever we have for the state value
    #         resultsCopy[4] = v[1]
    #
    #         loss = criterion(results, Variable(resultsCopy))
    #         print("PREDICTION: ")
    #         print(results)
    #         print("\n")
    #         print("ACTUAL: ")
    #         print("State Action Val: ", v[0])
    #         print("State Val: ", v[1])
    #         print("\n")
    #         print("LOSS: ")
    #         print(loss)
    #         loss.backward()
    #         optimizer.step()

def main():
    board_width = BOARD_WIDTH
    for epochs in range(0, 10):
        print("NUM EPOCHS:", epochs)
        tfe = tfet.TFE(board_width)
        # generate a new
        tfe.putNew()
        tfe.putNew()
        print("STARTING BOARD: ")
        # tfe.grid = np.array([0, 0, 2, 16, 0 ,0 , 64, 4, 0, 8, 16, 256, 2,4, 2, 16]).reshape((4,4))
        print(tfe.grid)
        print("")

        # mct = MCTZero(tfe, MONTE_CARLO_RUN, genValueFunction, policyUpdate)
        mct = MCTEZ(MONTE_CARLO_RUN, trainer, generateValue)
        while (not tfe.isWin()) and (not tfe.isLose()):

            start = time.clock()

            print("*********************")

            # For the MCTZero
            # act = mct.playerDecision()

            # For MCTEZ
            act = mct.playerDecision(tfe)


            print("AI SELECT ACTION: " + str(act))
            print("*********************")
            print("BEFORE: ")
            print(tfe.grid)
            print("")

            print("*********************")
            print("AFTER: ")

            # move grid
            tfe.moveGrid(act)

            # generate a new
            # advDecision = tfe.putNew()
            # mct.adversaryDecision(advDecision)

            # Try MCTEZ
            tfe.putNew()
            mct.adversaryDecision(tfe)

            print(tfe.grid)
            print("")

            print("TIME TAKEN FOR STEP: " + str(time.clock() - start))
            print("")
            # Flush it
            sys.stdout.flush()

        print("FINISHED: ")
        print(tfe.grid)
        print("")

        print("IS WIN?: ")
        print(tfe.isWin())
        print("")

        print("IS LOSE?: ")
        print(tfe.isLose())
        print("")

        train()

        # Save the model
        torch.save(NN, "model2.torch")


if __name__ == '__main__':
    main()
