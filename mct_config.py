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
    MONTE_CARLO_RUN = float(sys.argv[1])
else:
    MONTE_CARLO_RUN = 0

# Branch Weight
LEAF_WIN_WEIGHT = 100

# Keys for direction
DIR_KEY = {"u": 0, "d": 1, "l": 2, "r":3}
DIR_VAL = {v: k for k, v in DIR_KEY.items()}