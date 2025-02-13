import os
import time
import torch
import random
import logging
import numpy as np
import torch.backends.cudnn as cudnn

from config import get_args
from DataLoader.MSC import MSC
from DataLoader.QuickEval import QuickEval

def set_seed_logger(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

    return args.seed


def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger


if __name__ == "__main__":
    global logger
    args = get_args()
    date = time.strftime("%Y%m%d_%H:%M", time.localtime())
    logger_file = f"{args.dataset}_{date}.log"

    logger = get_logger(f"logs/{logger_file}")
    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))


    if args.dataset == "msc":
        msc = MSC(args, logger)
        msc.evaluation()

    elif args.dataset == "quickeval":
        quick_eval = QuickEval(args, logger)
        quick_eval.evaluation()
