import numpy as np
import sys
import copy
import time
from numpy import linalg as LA

from ntk import get_kernel_matrix, process_data_numpy, kernel_regression, get_test_accuracy, get_transform_matrix,  \
    get_same_class_relative_ratio
from data import save_sampled_binary_data, load_sampled_binary_data_from_pickle
from utils import set_random_seed, add_one_extra_dim


def get_positive_hull_indicator_one(theta_test_train, theta_train_train, a, b, option='train'):
    # inputs: theta_test_train (b, n), theta_train_train (n, n), z (4, s), a: test_index, b: train_index
    # outputs: V (n, n, 4, 4)
    n = theta_test_train.shape[1]
    V = np.zeros((4, 4, n, n))
    # (4, 4, n, n)
    V[:2] = (np.array([[1, 0, 0, 0],
                      [np.cos(theta_test_train[a][b]), np.sin(theta_test_train[a][b]), 0, 0]]).reshape(2, 4, 1, 1) +
             np.zeros((2, 4, n, n)))
    # (2, 4, n, n)
    theta_a_n = theta_test_train[a].reshape(n, 1) + np.zeros((n, n))
    theta_b_n = theta_train_train[b].reshape(n, 1) + np.zeros((n, n))
    theta_n_a = theta_test_train[a].reshape(1, n) + np.zeros((n, n))
    theta_n_b = theta_train_train[b].reshape(1, n) + np.zeros((n, n))
    # (n, n)
    V[2][0] = np.cos(theta_a_n)
    if np.sin(theta_test_train[a][b]) == 0:
        V[2][1] = np.sin(theta_a_n)
    else:
        V[2][1] = (np.cos(theta_b_n) - np.cos(theta_test_train[a][b]) * np.cos(theta_a_n)) / \
                  np.sin(theta_test_train[a][b])
    V[2][2] = np.sqrt(np.maximum(1 - V[2][0] * V[2][0] - V[2][1] * V[2][1], 0))

    V[3][0] = np.cos(theta_n_a)
    if np.sin(theta_test_train[a][b]) == 0:
        V[3][1] = (np.cos(theta_train_train) - np.cos(theta_a_n) * np.cos(theta_n_a)) / np.sin(theta_a_n)
        if option == 'train':
            V[3][1][a] = np.sin(theta_test_train[a])
    else:
        V[3][1] = (np.cos(theta_n_b) - np.cos(theta_test_train[a][b] * np.cos(theta_n_a))) / \
                  np.sin(theta_test_train[a][b])

    if np.sin(theta_test_train[a][b]) == 0:
        V[3][2] = np.sqrt(np.maximum(1 - V[3][0] * V[3][0] - V[3][1] * V[3][1], 0))
    else:
        V[3][2] = (np.cos(theta_train_train) - V[2][1] * V[3][1]) / V[2][2]

    V[3][3] = np.sqrt(np.maximum(1 - V[3][0] * V[3][0] - V[3][1] * V[3][1] - V[3][2] * V[3][2], 0))

    V = np.transpose(V, (2, 3, 0, 1))
    # (n, n, 4, 4)
    # VZ = (np.sum(np.dot(V, Z), axis=2) == 4)
    # (n, n, s)
    # print(np.min(V), np.max(V), np.mean(V))
    return V


def get_positive_hull_indicator_two(theta_test_train, theta_train_train, a, b, option='train'):
    # inputs: theta_test_train (b, n), theta_train_train (n, n), z (4, s), a: test_index, b: train_index
    # outputs: hull_indicator (n, n, s)
    n = theta_test_train.shape[1]
    V = np.zeros((4, 4, n, n))
    # (4, 4, n, n)
    V[0] = (np.array([1, 0, 0, 0]).reshape(1, 4, 1, 1) + np.zeros((1, 4, n, n)))
    # (1, 4, n, n)
    theta_a_n = theta_test_train[a].reshape(n, 1) + np.zeros((n, n))
    theta_b_n = theta_train_train[b].reshape(n, 1) + np.zeros((n, n))
    theta_n_a = theta_test_train[a].reshape(1, n) + np.zeros((n, n))
    theta_n_b = theta_train_train[b].reshape(1, n) + np.zeros((n, n))
    # (n, n)
    V[1][0] = np.cos(theta_a_n)
    V[1][1] = np.sin(theta_a_n)

    V[2][0] = np.cos(theta_test_train[a][b]).reshape(1, 1) + np.zeros((n, n))
    V[2][1] = (np.cos(theta_b_n) - np.cos(theta_a_n) * np.cos(theta_test_train[a][b])) / np.sin(theta_a_n)
    if option == 'train':
        V[2][1][a] = np.sin(theta_test_train[a][b])
    V[2][2] = np.sqrt(np.maximum(1 - V[2][0] * V[2][0] - V[2][1] * V[2][1], 0))

    V[3][0] = np.cos(theta_n_a)
    V[3][1] = (np.cos(theta_train_train) - np.cos(theta_a_n) * np.cos(theta_n_a)) / np.sin(theta_a_n)
    if option == 'train':
        if np.sin(theta_test_train[a][b]) == 0:
            V[3][1][a] = np.sin(theta_n_a)[0]
        else:
            V[3][1][a] = (np.cos(theta_n_b)[0] - np.cos(theta_test_train[a][b]) * np.cos(theta_n_a[0])) / \
                         np.sin(theta_test_train[a][b])
    V[3][2] = (np.cos(theta_n_b) - V[2][1] * V[3][1]) / V[2][2]
    if option == 'train':
        V[3][2][a] = np.sqrt(np.maximum(1 - V[3][0][a] * V[3][0][a] - V[3][1][a] * V[3][1][a], 0))
    V[3][3] = np.sqrt(np.maximum(1 - V[3][0] * V[3][0] - V[3][1] * V[3][1] - V[3][2] * V[3][2], 0))

    V = np.transpose(V, (2, 3, 0, 1))
    # (n, n, 4, 4)
    # VZ = (np.sum(np.dot(V, Z), axis=2) == 4)
    # (n, n, s)
    # print(np.min(V), np.max(V), np.mean(V))
    return V


def get_positive_hull_indicator_three(theta_test_train, theta_train_train, a, b):
    # inputs: theta_test_train (b, n), theta_train_train (n, n), z (4, s), a: test_index, b: train_index
    # outputs: hull_indicator (n, n, s)
    n = theta_test_train.shape[1]
    V = np.zeros((4, 4, n, n))
    # (4, 4, n, n)
    V[0] = (np.array([1, 0, 0, 0]).reshape(1, 4, 1, 1) + np.zeros((1, 4, n, n)))
    # (1, 4, n, n)
    theta_a_n = theta_test_train[a].reshape(n, 1) + np.zeros((n, n))
    theta_b_n = theta_train_train[b].reshape(n, 1) + np.zeros((n, n))
    theta_n_a = theta_test_train[a].reshape(1, n) + np.zeros((n, n))
    theta_n_b = theta_train_train[b].reshape(1, n) + np.zeros((n, n))
    # (n, n)
    V[1][0] = np.cos(theta_b_n)
    V[1][1] = np.sin(theta_b_n)

    V[2][0] = np.cos(theta_test_train[a][b]).reshape(1, 1) + np.zeros((n, n))
    V[2][1] = (np.cos(theta_a_n) - np.cos(theta_b_n) * np.cos(theta_test_train[a][b])) / np.sin(theta_b_n)
    V[2][1][b] = np.sin(theta_test_train[a][b])
    V[2][2] = np.sqrt(np.maximum(1 - V[2][0] * V[2][0] - V[2][1] * V[2][1], 0))

    V[3][0] = np.cos(theta_n_b)
    V[3][1] = (np.cos(theta_train_train) - np.cos(theta_b_n) * np.cos(theta_n_b)) / np.sin(theta_b_n)
    if np.sin(theta_test_train[a][b]) == 0:
        V[3][1][b] = np.sin(theta_n_b)[0]
    else:
        V[3][1][b] = (np.cos(theta_n_a)[0] - np.cos(theta_test_train[a][b]) * np.cos(theta_n_b[0])) / \
                         np.sin(theta_test_train[a][b])
    V[3][2] = (np.cos(theta_n_a) - V[2][1] * V[3][1]) / V[2][2]
    V[3][2][b] = np.sqrt(np.maximum(1 - V[3][0][b] * V[3][0][b] - V[3][1][b] * V[3][1][b], 0))
    V[3][3] = np.sqrt(np.maximum(1 - V[3][0] * V[3][0] - V[3][1] * V[3][1] - V[3][2] * V[3][2], 0))

    V = np.transpose(V, (2, 3, 0, 1))
    # (n, n, 4, 4)
    # VZ = (np.sum(np.dot(V, Z), axis=2) == 4)
    # (n, n, s)
    # print(np.min(V), np.max(V), np.mean(V))
    return V


def get_fourth_Kab(inner_products_test_train, inner_products_train_train, norm_products_train_train, X_train_norm,
                   X_test_norm, theta_test_train, theta_train_train, a, b, Z, option='train'):
    # inputs: X_train (n, d), X_test (b, d), a: train index, b: test index
    # outputs: E_K^4(a, b) (n, n)
    n = X_train_norm.shape[0]
    first_item_ratio = 2 * inner_products_test_train[a][b] * inner_products_train_train + \
                       np.dot(inner_products_test_train[a].reshape(n, 1), inner_products_train_train[b].reshape(1, n)) \
                       + np.dot(inner_products_train_train[b].reshape(n, 1), inner_products_test_train[a].reshape(1, n))
    # (n, n)
    second_item_ratio = 2 * inner_products_test_train[a][b] * norm_products_train_train
    # (n, n)
    third_item_ratio = X_train_norm[b] * np.dot(inner_products_test_train[a].reshape(n, 1), X_train_norm.reshape(1, n))
    # (n, n)
    fourth_item_ratio = X_test_norm[a] * np.dot(inner_products_train_train[b].reshape(n, 1), X_train_norm.reshape(1, n))
    # (n, n)
    V_one = get_positive_hull_indicator_one(theta_test_train, theta_train_train, a, b, option)
    V_two = get_positive_hull_indicator_two(theta_test_train, theta_train_train, a, b, option)
    V_three = get_positive_hull_indicator_three(theta_test_train, theta_train_train, a, b)
    # (n, n, 4, 4)
    VZ_one = (np.sum(np.dot(V_one, Z) >= 0, axis=2) == 4)
    VZ_two = (np.sum(np.dot(V_two, Z) >= 0, axis=2) == 4)
    VZ_three = (np.sum(np.dot(V_three, Z) >= 0, axis=2) == 4)
    # (n, n, s)
    first_item = np.mean(VZ_one, axis=2)
    # (n, n)
    second_item = np.mean(np.dot(V_one[:, :, 2, :], Z) * np.dot(V_one[:, :, 3, :], Z) * VZ_one, axis=2)
    # (n, n)
    third_item = np.mean(np.dot(V_two[:, :, 2, :], Z) * np.dot(V_two[:, :, 3, :], Z) * VZ_two, axis=2)
    # (n, n)
    fourth_item = np.mean(np.dot(V_three[:, :, 2, :], Z) * np.dot(V_three[:, :, 3, :], Z) * VZ_three, axis=2)
    # (n, n)
    Kab = first_item_ratio * first_item + second_item_ratio * second_item + third_item_ratio * third_item + \
          fourth_item_ratio * fourth_item
    return Kab


def get_ntl_simple_acc(X_train, X_test, Y_train, Z, m, option):
    expected_second_order = get_kernel_matrix(X_train, X_test)
    # (b, n)
    H = get_kernel_matrix(X_train, X_train)
    # (n, n)
    eigenvalues, eigen_P = LA.eig(H)
    eigen_D = np.diag(eigenvalues)
    eigen_D_inv = np.linalg.inv(eigen_D)
    eigen_P_transpose = np.transpose(eigen_P)
    print('eigenvalues', eigenvalues)
    print('eigen_P', eigen_P)
    print('eigen_D', eigen_D)
    print('eigen_D_inv', eigen_D_inv)
    print('eigen_P_transpose', eigen_P_transpose)
    print('double check', np.dot(np.dot(eigen_P, eigen_D), eigen_P_transpose))
    print('H', H)
    transform_matrix = np.linalg.solve(H, Y_train)
    # (n, 1)
    print('HT', np.dot(H, transform_matrix))
    expected_fourth_order = np.zeros((X_test.shape[0], X_train.shape[0]))
    # (b, n)
    X_train_norm = np.linalg.norm(X_train, axis=1).reshape(-1, 1)
    # (n, 1)
    X_test_norm = np.linalg.norm(X_test, axis=1).reshape(-1, 1)
    # (b, 1)
    inner_products_test_train = np.dot(X_test, np.transpose(X_train))
    # (b, n)
    inner_products_train_train = np.dot(X_train, np.transpose(X_train))
    # (n, n)
    norm_products_test_train = np.dot(X_test_norm, np.transpose(X_train_norm))
    # (b, n)
    norm_products_train_train = np.dot(X_train_norm, np.transpose(X_train_norm))
    # (n, n)
    cos_values_test_train = np.clip(inner_products_test_train / norm_products_test_train, -1.0, 1.0)
    # (b, n)
    cos_values_train_train = np.clip(inner_products_train_train / norm_products_train_train, -1.0, 1.0)
    # (n, n)
    theta_test_train = np.arccos(cos_values_test_train)
    # (b, n)
    theta_train_train = np.arccos(cos_values_train_train)
    for a in range(len(X_test)):
        # print(a, end='\r')
        for b in range(len(X_train)):
            # time_start = time.time()
            Kab = get_fourth_Kab(inner_products_test_train, inner_products_train_train, norm_products_train_train,
                                 X_train_norm, X_test_norm, theta_test_train, theta_train_train, a, b, Z, option)
            Kab /= m
            item_Q = np.dot(np.dot(eigen_P_transpose, Kab), np.dot(eigen_P, eigen_D_inv))
            item_Q_deno = eigenvalues.reshape(-1, 1) + eigenvalues.reshape(1, -1)
            item_Q = item_Q / item_Q_deno
            Qab = np.dot(np.dot(eigen_P, item_Q), eigen_P_transpose)
            if a == b == 0:
                print('Qab', Qab)
            # print(Kab)
            # time_end = time.time()
            # print('time per ab', time_end - time_start)
            expected_fourth_order[a][b] = np.dot(np.transpose(transform_matrix), np.dot(Kab, transform_matrix)) - \
                                          np.dot(np.transpose(Y_train), np.dot(Qab, Y_train))
    # print('expected_second_order', expected_second_order)
    # print('expected_fourth_order', expected_fourth_order)
    kernel_values = expected_second_order + expected_fourth_order
    # kernel_values = expected_second_order
    # print('kernel values', kernel_values)
    return kernel_values, expected_second_order, expected_fourth_order


def run_ntl_simple_experiments():
    neg_label = int(sys.argv[1].split('=')[1])
    pos_label = int(sys.argv[2].split('=')[1])
    # binary_sample_size = int(sys.argv[3].split('=')[1])
    print('neg label', neg_label)
    print('pos label', pos_label)
    # print('binary sample size', binary_sample_size)
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    config = {'sample_size': 10000, 'seed': 66, 'dataset': 'MNIST', 'input_size': 1 * 28 * 28 + 1,
              'hidden_size': 40960, 'epoch_num': 500, 'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'lr': 0.5, 'binary_labels': [str(neg_label), str(pos_label)],
              'dir_path': '/path/to/working/dir', 'iid_sample_size': 400}

    # print('set random seed', config['seed'])
    # set_random_seed(config['seed'])
    # print('sample data')
    # save_sampled_binary_data(config)
    print('set random seed', config['seed'])
    set_random_seed(config['seed'])
    print('load data')
    train_data, test_data = load_sampled_binary_data_from_pickle(config)
    train_data = add_one_extra_dim(train_data)
    test_data = add_one_extra_dim(test_data)
    train_data = train_data[:20]
    test_data = test_data[:20]
    print('train data size', len(train_data))
    print('test data size', len(test_data))
    print('process data to numpy format')
    total_inputs_train, total_outputs_train = process_data_numpy(train_data)
    total_inputs_test, total_outputs_test = process_data_numpy(test_data)
    print('train data size', total_inputs_train.shape, total_outputs_train.shape)
    print('test data size', total_inputs_test.shape, total_outputs_test.shape)
    print('train inputs', total_inputs_train)
    print('train outputs', total_outputs_train)
    Y_train = copy.deepcopy(total_outputs_train)
    Y_test = copy.deepcopy(total_outputs_test)
    print('Y_train', Y_train)
    print('random sample Z')
    Z = np.random.normal(size=(4, config['iid_sample_size']))
    print('Z', Z)
    # (4, s)
    # get train kernel values
    train_kernel_values, train_second, train_fourth = \
        get_ntl_simple_acc(total_inputs_train, total_inputs_train, Y_train, Z, config['hidden_size'], option='train')
    Y_train_train = np.dot(Y_train, np.transpose(Y_train))
    print('train fourth', train_fourth)
    print('Y_train_train', Y_train_train)
    print('intra-class', np.mean(train_fourth * (Y_train_train >= 0)))
    print('inter-class', np.mean(train_fourth * (Y_train_train <= 0)))
    # get test kernel values
    test_kernel_values, test_second, test_fourth = \
        get_ntl_simple_acc(total_inputs_train, total_inputs_test, Y_train, Z, config['hidden_size'], option='test')
    Y_test_train = np.dot(Y_test, np.transpose(Y_train))
    print('test fourth', test_fourth)
    print('Y_test_train', Y_test_train)
    print('intra-class', np.mean(test_fourth * (Y_test_train >= 0)))
    print('inter-class', np.mean(test_fourth * (Y_test_train <= 0)))
    '''# get transform matrix
    print('train kernel values size', train_kernel_values)
    transform_matrix = get_transform_matrix(train_kernel_values, total_outputs_train)
    print('transform matrix size', transform_matrix)
    # get test kernel values
    test_kernel_values = get_ntl_simple_acc(total_inputs_train, total_inputs_test, Y_train, Z, config['hidden_size'],
                                            option='test')
    print('test kernel values size', test_kernel_values)
    # get test outputs
    test_outputs = kernel_regression(test_kernel_values, transform_matrix)
    print('test outputs size', test_outputs)
    # get test accuracy
    test_accuracy = get_test_accuracy(test_outputs, total_outputs_test)
    print('test accuracy', test_accuracy)
    # get same class ratio
    train_same_class_ratio = get_same_class_relative_ratio(train_kernel_values, train_kernel_values,
                                                           train_kernel_values, total_outputs_train,
                                                           total_outputs_train)
    print('train same class ratio', train_same_class_ratio)
    test_test_kernel_values = get_kernel_matrix(total_inputs_test, total_inputs_test)
    test_train_same_class_ratio = get_same_class_relative_ratio(test_kernel_values, test_test_kernel_values,
                                                                train_kernel_values, total_outputs_test,
                                                                total_outputs_train)
    print('test train same class ratio', test_train_same_class_ratio)
    '''


if __name__ == '__main__':
    run_ntl_simple_experiments()




