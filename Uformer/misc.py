#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch


# def save_checkpoint(model1, model2, model3, optimizer1, optimizer2, optimizer3, epoch, step, checkpoint_dir):
def save_checkpoint(model1, optimizer1, epoch, step, checkpoint_dir):
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model.ckpt-{}-{}.pt'.format(epoch,step))
    torch.save({'model1': model1.state_dict(),
                # 'model2': model2.state_dict(),
                # 'model3': model3.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                # 'optimizer2': optimizer2.state_dict(),
                # 'optimizer3': optimizer3.state_dict(),
                'epoch': epoch,
                'step': step}, checkpoint_path)
    with open(os.path.join(checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write('model.ckpt-{}-{}.pt'.format(epoch,step))
    print("=> Save checkpoint:", checkpoint_path)


# def reload_model(model1, model2, model3, optimizer1, optimizer2, optimizer3, checkpoint_dir, use_cuda=True):
def reload_model(model1, optimizer1, checkpoint_dir, use_cuda=True):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model1.load_state_dict(checkpoint['model1'])#, strict=False
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        print('=> Reload previous model and optimizer.',model_name)
    else:
        print('[!] checkpoint directory is empty. Train a new model ...')
        epoch = 0
        step = 0
    return epoch, step


# def reload_for_eval(model1, model2, model3, checkpoint_dir, use_cuda):
def reload_for_eval(model1, checkpoint_dir, use_cuda):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model1.load_state_dict(checkpoint['model']) # model1
        # model2.load_state_dict(checkpoint['model2'])
        # model3.load_state_dict(checkpoint['model3'])
        print('=> Reload well-trained model {} for decoding.'.format(
            model_name))


def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def learning_rate_decaying(optimizer, rate):
    """decaying the learning rate"""
    lr = get_learning_rate(optimizer) * rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_learning_rate(optimizer):
    """Get learning rate"""
    return optimizer.param_groups[0]["lr"]
