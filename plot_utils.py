# -*- encoding: utf-8 -*-
'''
Author: Z.T. Zhu
Date  : September 6th, 2022
'''
from tensorflow import reshape, cast, int8
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 13

def plot_train_process(history):
    epoch = history.epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(epoch, loss, label='train')
    ax1.plot(epoch, val_loss, label='test')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(epoch, acc, label='train')
    ax2.plot(epoch, val_acc, label='test')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend()

def plot_result(test_set, model):
    plot_ds = test_set.unbatch().take(9).batch(9)
    for image, _ in plot_ds.take(1):
        masks_pred = model(image)

    masks_pred = masks_pred > 0.5
    masks_pred = cast(masks_pred, dtype=int8)

    fig = plt.figure(figsize=(6, 6))
    for i, (image, _) in enumerate(plot_ds.unbatch().take(9)):
        ax = fig.add_subplot(3, 3, i + 1)
        plot_mask(ax, image, masks_pred[i, ...])