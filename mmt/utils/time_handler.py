'''
Author: jianzhnie
Date: 2021-11-18 18:22:32
LastEditTime: 2021-12-03 17:21:47
LastEditors: jianzhnie
Description:

'''
import signal
import time

import torch
import torch.distributed as dist


def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):

    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start

    return _timed_function


def first_n(n, generator):
    for i, d in zip(range(n), generator):
        yield d


class TimeoutHandler:

    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig
        self.device = torch.device('cuda')

    @property
    def interrupted(self):
        if not dist.is_initialized():
            return self._interrupted

        interrupted = torch.tensor(self._interrupted).int().to(self.device)
        dist.broadcast(interrupted, 0)
        interrupted = bool(interrupted.item())
        return interrupted

    def __enter__(self):
        self._interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def master_handler(signum, frame):
            self.release()
            self._interrupted = True
            print('Received SIGTERM')

        def ignoring_handler(signum, frame):
            self.release()
            print('Received SIGTERM, ignoring')

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            signal.signal(self.sig, master_handler)
        else:
            signal.signal(self.sig, ignoring_handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


def calc_ips(batch_size, time):
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized() else 1)
    tbs = world_size * batch_size
    return tbs / time
