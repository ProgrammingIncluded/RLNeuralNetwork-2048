############################################
# Project: MCT-TFE
# File: mct_config.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

from TFE import *
import numpy as np
import sys

# System arguments.
if len(sys.argv) >= 2:
    MONTE_CARLO_RUN = int(sys.argv[1])
else:
    MONTE_CARLO_RUN = 0

# Number of 0's minimum before starting non-greedy approach
if len(sys.argv) >= 3:
    GREEDY_THRESH = int(sys.argv[2])
else:
    GREEDY_THRESH = 16

# Allow greedy algorithm for the first few squares.
GREEDY_CONTROL = False
GREEDY_INIT_ONLY = True

# TODO: Value Heuristic
VAL_H = True
# Directions of corners to check, across.
V_DIR = 2

# Branch Weight
LEAF_WIN_WEIGHT = 1000

# Max children for each turn. 4 directions, 16 available, 2 possible gen values.
MAX_CHILDREN = 2 * 4 * 16

# Keys for direction
DIR_KEY = {"u": 0, "d": 1, "l": 2, "r":3}
DIR_VAL = {v: k for k, v in DIR_KEY.items()}

# Heuristic filter used for weighting corners. n^2 weighted.
VFB = np.array([[64, 18, 4, 1], [18, 18, 4, 1], [18, 18, 18, 1], [1, 1, 1, 1]]).reshape((4,4))
FILTERS = [VFB, np.rot90(VFB, 1), np.rot90(VFB, 3), np.rot90(VFB, 2), np.rot90(VFB, 4)]