import logging
import os
import shutil
from collections import OrderedDict

import torch
from torch import distributed as dist


def save_checkpoint(state,
                    is_best,
                    checkpoint_dir,
                    filename='checkpoint.pth.tar'):
    if (not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0:
        file_path = os.path.join(checkpoint_dir, filename)
        torch.save(state, file_path)
        if is_best:
            shutil.copyfile(file_path,
                            os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized() else 1)
    return rt


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30
    epochs."""
    lr = args.lr * (0.1**(epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


_logger = logging.getLogger(__name__)


def load_checkpoint(model, checkpoint_path, log_info=True):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        else:
            model.load_state_dict(checkpoint, strict=False)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def resum_checkpoint(args, log_info=True):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(
            args.resume,
            map_location=lambda storage, loc: storage.cuda(args.gpu))
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model_state = checkpoint['state_dict']
        optimizer_state = checkpoint['optimizer']
        if 'state_dict_ema' in checkpoint:
            model_state_ema = checkpoint['state_dict_ema']
        else:
            model_state_ema = None
        print("=> loaded checkpoint '{}' (epoch {})".format(
            args.resume, checkpoint['epoch']))
        if start_epoch >= args.epochs:
            print(
                f'Launched training for {args.epochs}, checkpoint already run {start_epoch}'
            )
            exit(1)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        model_state = None
        model_state_ema = None
        optimizer_state = None

    return model_state, model_state_ema, optimizer_state, start_epoch, best_prec1


def test_load_checkpoint(args):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(
            args.resume,
            map_location=lambda storage, loc: storage.cuda(args.gpu))
        checkpoint = {
            k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in checkpoint.items()
        }
        optimizer_state = checkpoint['optimizer']
        model_state = checkpoint['state_dict']
        if 'state_dict_ema' in checkpoint:
            model_state_ema = checkpoint['state_dict_ema']
        else:
            model_state_ema = None
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        model_state = None
        model_state_ema = None
        optimizer_state = None

    return model_state, model_state_ema, optimizer_state


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
