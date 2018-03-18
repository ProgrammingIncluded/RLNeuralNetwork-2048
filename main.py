############################################
# Project: MCT-TFE
# File: main.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

import TFE as tfet
import numpy as np
import time
import random as rnd
import sys
import random
import matplotlib.pyplot as plt


def main(n):
    tfet.BOARD_WIDTH = n
    tfet.MAX_VALUE = 2048*64
    tfe = tfet.TFE()
    # generate a new
    tfe.putNew()
    tfe.putNew()
    # print("STARTING BOARD: ")
    # tfe.grid = np.array([0, 0, 2, 16, 0 ,0 , 64, 4, 0, 8, 16, 256, 2,4, 2, 16]).reshape((4,4))
    # print(tfe.grid)
    # print("")

    # np.set_printoptions(threshold=np.nan)

    while (not tfe.isWin()) and (not tfe.isLose()):
        act = tfet.MOV_OPT[random.randint(0, 3)]

        start = time.clock()

        # print("*********************")

        # print("AI SELECT ACTION: " + str(act))
        # print("*********************")
        # print("BEFORE: ")
        # print(tfe.grid)
        # print("")

        # print("*********************")
        # print("AFTER: ")

        # move grid
        tfe.moveGrid(act)

        # generate a new
        tfe.putNew()

        # print(tfe.grid)
        # print("")

        # print("TIME TAKEN FOR STEP: " + str(time.clock() - start))
        # print("")
        # Flush it
        sys.stdout.flush()
    return np.amax(tfe.grid)

    # print("FINISHED: ")
    # print(tfe.grid)
    # print("")

    # print("IS WIN?: ")
    # print(tfe.isWin())
    # print("")

    # print("IS LOSE?: ")
    # print(tfe.isLose())
    # print("")


if __name__ == '__main__':
    size = np.arange(1,7)
    mean = np.zeros(size.shape)
    for i in range(size.shape[0]):
        print(size[i])
        runningavg = 0.0
        for j in range(100):
            runningavg += main(size[i])
        mean[i] = runningavg/1
    plt.plot(size,np.log2(mean),'b')
    plt.xlabel("Dimension of grid (NxN)")
    plt.ylabel("Log2(Average performance)")
    plt.title("N-Dimensional 2048 Problem")
    plt.grid()
    plt.show()
