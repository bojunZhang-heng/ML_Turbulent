# test.py

import os
import numpy as np
import torch
import logging
from utils.utils import setup_logger, knn, get_graph_feature
from colorama import Fore, Style

#! alias for colorful output
R = Fore.RED
Y = Fore.YELLOW
G = Fore.GREEN
M = Fore.MAGENTA
C = Fore.CYAN
RESET = Style.RESET_ALL

def main():
    """ Test knn() and get_graph_feature() function. """

    exp_dir = os.path.join('experiments', "Test")
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, 'test.log')
    setup_logger(log_file)
    logging.info(f"{Fore.RED}*******************************Starting Test knn() and get_graph_feature() function. {Style.RESET_ALL}")

    x = torch.tensor([[
        [0., 0.],
        [1., 1.],
        [-2, -2],
        [3., 3.]
    ]], dtype=torch.float32)                      # x.shape = [1, 4, 2]
    kk = 2
    logging.info(f"x.shape: {x.shape}")
    logging.info(f"x value: {x}")


    idx = knn(x, kk)
    feature = get_graph_feature(x, kk, idx, )

if __name__=="__main__":
    main()
