'''
Author: jianzhnie
Date: 2021-11-17 12:15:26
LastEditTime: 2021-11-17 12:16:38
LastEditors: jianzhnie
Description:

'''

import logging

logger = logging.getLogger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)
print(trainer_log_levels)
