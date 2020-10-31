import torch
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

plt.switch_backend('agg')


def shuffle_in_union(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_minibatches_idx(n, minibatch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(math.ceil(n / minibatch_size)):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return minibatches


def get_data(positions, labels):
    data = []
    for idx in range(len(positions)):
        inputs = torch.FloatTensor([positions[idx]])
        targets = torch.LongTensor([labels[idx]])
        data.append((inputs, targets))
    return data


def plot_scatter(random_positions, random_y_pred_np, path, option='2D'):
    if option == '3D':
        fig = plt.figure()
        plt.rcParams.update({'font.size': 23})
        ax = fig.add_subplot(111, projection='3d')
        random_xs = random_positions[:, 0]
        random_ys = random_positions[:, 1]
        random_zs = random_positions[:, 2]
        prob = random_y_pred_np[:, 0] * -1
        ax.scatter(random_xs, random_ys, random_zs, c=prob, cmap='bwr', vmin=-1.0, vmax=0.0, marker='.')
        ax.set_xlabel('X', labelpad=15)
        ax.set_ylabel('Y', labelpad=15)
        ax.set_zlabel('Z', labelpad=15)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    else:
        plt.figure()
        plt.rcParams.update({'font.size': 23})
        random_xs = random_positions[:, 0]
        random_ys = random_positions[:, 1]
        prob = random_y_pred_np[:, 0] * -1
        plt.scatter(random_xs, random_ys, c=prob, cmap='bwr', vmin=-1.0, vmax=0.0, marker='.')
        plt.xlabel('X', labelpad=15)
        plt.ylabel('Y', labelpad=15)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def plot_delta_versus_distance(total_delta, total_dist, path, xlabel='Geodesic Distance', ylabel='Relative Similarity'):
    plt.rcParams.update({'font.size': 23})
    plt.scatter(total_dist, total_delta, marker='.', color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
