#!/usr/bin/env python
# -*- coding:utf-8 -*-
import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, shift_limit=0.1, p=0.5),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = []
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn=None):
    """不使用预处理函数，因为输入是多通道数据"""
    _transform = []
    return albu.Compose(_transform) 