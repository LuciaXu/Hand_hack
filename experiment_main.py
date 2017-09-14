
from config import handConfig
from Datareader import Datareader
#from train_models import test_input_full,test_input_tree
from train_dense_hier_networks import train_model

import os
import numpy as np


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = handConfig()
    seqconfig = Datareader(config)
    print("tfrecords saved")
    train_model(config,seqconfig)
    return 0

if __name__ == '__main__':
    main()
