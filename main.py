############################################
# Project: MCT-TFE
# File: main.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

from TFE import *
import time
import random as rnd
import sys


def main():
    tfe = TFE()
    # generate a new
    tfe.putNew()
    tfe.putNew()
    print "STARTING BOARD: "
    # tfe.grid = np.array([0, 0, 2, 16, 0 ,0 , 64, 4, 0, 8, 16, 256, 2,4, 2, 16]).reshape((4,4))
    print tfe.grid
    print ""

    act = 'd'
    while (not tfe.isWin()) and (not tfe.isLose()):

        start = time.clock() 

        print "*********************"

        print "AI SELECT ACTION: " + act
        print "*********************"
        print "BEFORE: "
        print tfe.grid
        print ""

        print "*********************"
        print "AFTER: "

        # move grid
        tfe.moveGrid(act)

        # generate a new
        tfe.putNew()

        print tfe.grid
        print ""

        print "TIME TAKEN FOR STEP: " + str(time.clock() - start)
        print ""
        # Flush it
        sys.stdout.flush()
    
    print "FINISHED: "
    print tfe.grid
    print ""

    print "IS WIN?: "
    print tfe.isWin()
    print ""
    
    print "IS LOSE?: "
    print tfe.isLose()
    print ""


if __name__ == '__main__':
    main()