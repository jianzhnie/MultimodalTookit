'''
Author: jianzhnie
Date: 2021-11-18 16:28:01
LastEditTime: 2021-11-18 16:54:27
LastEditors: jianzhnie
Description:

'''
from transformers import Trainer, logging


class MMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        log_level = args.get_process_log_level()
        print(log_level)
        # set the correct log level depending on the node
        logging.set_verbosity(logging.INFO)


if __name__ == '__main__':
    logger = logging.get_logger(__name__)
    log_levels = logging.get_log_levels_dict().copy()
    trainer_log_levels = dict(**log_levels, passive=-1)
    print(trainer_log_levels)
