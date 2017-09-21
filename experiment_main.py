
from config import handConfig
from Datareader import Datareader
from train_models import test_input_full,test_input_tree
from train_cnn_networks import train_model, test_model

import os
import numpy as np


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = handConfig()
    seqconfig = Datareader(config)
    #test_input_full(config,seqconfig)
    train_model(config,seqconfig)
    #test_model(config,seqconfig)

    return 0

if __name__ == '__main__':
    main()
