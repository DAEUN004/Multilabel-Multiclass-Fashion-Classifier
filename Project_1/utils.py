import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def converter(labels):
    num_classes = [7, 3, 3, 4, 6, 3]

    argmax_values = []

    # Starting index for slicing
    start_index = 0

    # Iterate over num_classes
    for num_class in num_classes:
        # Get the slice corresponding to the current category
        slice_labels = labels[:, start_index:start_index + num_class]
        
        # Apply argmax along dimension 1
        argmax_value = slice_labels.argmax(dim=1)
        
        # Append to the list
        argmax_values.append(argmax_value)
        
        # Update the start index for the next category
        start_index += num_class

    # Stack the argmax values along dimension 1 to form the final tensor
    argmax_values_tensor = torch.stack(argmax_values, dim=1)

    return argmax_values_tensor


def plot_loss(train_loss, val_loss, fig_name):
    x = np.arange(len(train_loss))
    max_loss = max(max(train_loss), max(val_loss))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_ylim([0,max_loss+1])
    lns1 = ax1.plot(x, train_loss, 'yo-', label='train_loss')
    lns2 = ax1.plot(x, val_loss, 'go-', label='val_loss')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    fig.tight_layout()
    fig_name_loss = fig_name.replace('.png','_loss.png')  # Modify fig_name to include _loss
    plt.title(fig_name_loss)

    plt.savefig(os.path.join('./diagram', fig_name_loss))  # Save with modified fig_name

    np.savez(os.path.join('./diagram', fig_name.replace('.png ', '.npz')), train_loss=train_loss, val_loss=val_loss)

def plot_acc(train_acc, val_acc, fig_name):
    x = np.arange(len(train_acc))
    max_loss = max(max(train_acc), max(val_acc))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('acc')
    ax1.set_ylim([0,1])
    lns1 = ax1.plot(x, train_acc, 'yo-', label='train_acc')
    lns2 = ax1.plot(x, val_acc, 'go-', label='val_acc')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    fig.tight_layout()
    fig_name_acc = fig_name.replace('.png','_acc.png')  # Modify fig_name to include _acc
    plt.title(fig_name_acc)

    plt.savefig(os.path.join('./diagram', fig_name_acc))  # Save with modified fig_name

    np.savez(os.path.join('./diagram', fig_name.replace('.png ', '.npz')), train_loss=train_acc, val_loss=val_acc)

def plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, fig_name):
    plot_acc(stat_training_acc, stat_val_acc, fig_name)
    plot_loss(stat_training_loss, stat_val_loss,fig_name)
