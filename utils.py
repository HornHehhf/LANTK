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


def add_one_extra_dim(data):
    new_data = []
    for (inputs, targets) in data:
        inputs = inputs.cpu().numpy().reshape(-1)
        inputs = inputs.tolist()
        # add one dimension
        inputs.append(1)
        inputs = torch.Tensor(np.array(inputs).reshape(1, -1))
        new_data.append((inputs, targets))
    return new_data
