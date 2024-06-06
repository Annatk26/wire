import logging
import random

import numpy as np
import torch


def setup_logger(log_level: str) -> None:
    logging.basicConfig(
        level=log_level,
        format=
        "%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True)

    logging.info(f"Logger set to {log_level}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f"Seed set to {seed}")


def setup():
    setup_logger("INFO")
    seed_everything(0)
