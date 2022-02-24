'''
Author: jianzhnie
Date: 2021-11-17 09:29:59
LastEditTime: 2022-02-24 12:27:30
LastEditors: jianzhnie
Description:

'''
import re
from os import makedirs
from os.path import abspath, dirname, exists, join


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_data_path():
    return join(get_root_path(), 'datasets')


def get_log_path():
    return join(get_log_path(), 'logs')


def create_dir_if_not_exists(folder):
    if not exists(folder):
        makedirs(folder)


def get_args_info_as_str(config_flags):
    rtn = []
    d = vars(config_flags)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        if type(v) is dict:
            for k2, v2 in v.items():
                s = '{0:26} : {1}'.format(k + '_' + k2, v2)
                rtn.append(s)
        else:
            s = '{0:26} : {1}'.format(k, v)
            rtn.append(s)
    return '\n'.join(rtn)


def sorted_nicely(sort_keys, reverse=False):

    def tryint(s):
        return int(s)

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(
                s, sort_keys))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(sort_keys, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn
