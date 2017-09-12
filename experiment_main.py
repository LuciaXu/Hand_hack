
from config import handConfig
from Datareader import Datareader
from train_models import test_input_full,test_input_tree

import os
import numpy as np


def main():
    config = handConfig()
    #seqconfig = Datareader(config)
    #test_input_full(config,seqconfig)
    return 0

if __name__ == '__main__':
    main()
