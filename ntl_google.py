import numpy as np
import sys
import pickle

from data import save_sampled_binary_data, load_sampled_binary_data_from_pickle
from utils import set_random_seed, add_one_extra_dim
from ntk import get_same_class_relative_ratio, process_data_numpy, get_kernel_matrix, kernel_regression, \
    get_transform_matrix, get_test_accuracy

from second_regression import get_pair_predictions, get_pca_features


def get_ntl_kernel_V1_pre(X_train, Y_train, H_train_train, H_test_train, config):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    n = X_train.shape[0]
    # H_train_train = get_kernel_matrix(X_train, X_train)
    # (n, n)
    # H_test_train = get_kernel_matrix(X_train, X_test)
    # (b, n)
    label_inner_products = np.dot(Y_train, np.transpose(Y_train))
    # (n, n)
    sigma_square = np.square(np.max(H_train_train) - np.min(H_train_train))
    const_H_square_y = np.sum(H_train_train * H_train_train * label_inner_products)
    const_H_y = np.sum(H_train_train * label_inner_products)
    const_y = np.sum(label_inner_products)
    const_H_square = np.sum(H_train_train * H_train_train)
    const_H = np.sum(H_train_train)

    train_label_kernels = (
                          sigma_square * const_y - const_y * H_train_train * H_train_train + 2 * const_H_y * H_train_train -
                          const_H_square_y) / (n * n * sigma_square - n * n * H_train_train * H_train_train +
                                               2 * const_H * H_train_train - const_H_square)

    test_label_kernels = (sigma_square * const_y - const_y * H_test_train * H_test_train + 2 * const_H_y * H_test_train -
                     const_H_square_y) / (n * n * sigma_square - n * n * H_test_train * H_test_train +
                                          2 * const_H * H_test_train - const_H_square)
    # for i in range(label_kernels.shape[0]):
    #    print('label dia', label_kernels[i][i])
    # label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    # label_ratio = 1.0
    return train_label_kernels, test_label_kernels


def get_ntl_kernel_V2_origin(X_train, X_test, Y_train, config):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    H_train_train = get_kernel_matrix(X_train, X_train)
    # (n, n)
    H_inv = np.linalg.inv(H_train_train)
    # (n, n)
    H_inv_two = np.dot(H_inv, H_inv)
    # (n, n)
    # normalize H_inv_two
    H_inv_two = (H_inv_two - np.min(H_inv_two)) / (np.max(H_inv_two) - np.min(H_inv_two))
    print('H_inv_two', H_inv_two)
    H_test_train = get_kernel_matrix(X_train, X_test)
    # (b, n)
    label_inner_products = np.dot(Y_train, np.transpose(Y_train))
    # (n, n)
    sigma_square = np.square(np.max(H_train_train) - np.min(H_train_train))
    const_H_square_inv_y = np.sum(H_train_train * H_train_train * H_inv_two * label_inner_products)
    const_H_inv_y = np.sum(H_train_train * H_inv_two * label_inner_products)
    const_inv_y = np.sum(H_inv_two * label_inner_products)
    const_H_square_inv = np.sum(H_train_train * H_train_train * H_inv_two)
    const_H_inv = np.sum(H_train_train * H_inv_two)
    const_inv = np.sum(H_inv_two)

    train_label_kernels = (sigma_square * const_inv_y - const_inv_y * H_train_train * H_train_train +
                           2 * const_H_inv_y * H_train_train - const_H_square_inv_y) / \
                          (sigma_square * const_inv - const_inv * H_train_train * H_train_train +
                           2 * const_H_inv * H_train_train - const_H_square_inv)

    label_kernels = (sigma_square * const_inv_y - const_inv_y * H_test_train * H_test_train +
                     2 * const_H_inv_y * H_test_train - const_H_square_inv_y) / \
                    (sigma_square * const_inv - const_inv * H_test_train * H_test_train +
                     2 * const_H_inv * H_test_train - const_H_square_inv)
    label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    kernel_values = H_test_train + label_ratio * label_kernels
    # for i in range(label_kernels.shape[0]):
    #     print('label dia', label_kernels[i, i])
    #     if label_kernels[i, i] < 0 and config['train']:
    #        label_kernels[i, i] = 1.0
    #        print('hello')
    #    print('label dia', label_kernels[i, i])
    print('H', H_test_train)
    print('label kernels', label_kernels * label_ratio)
    print('kernel values', kernel_values)
    return kernel_values


def get_ntl_kernel_V2_pre(X_train, Y_train, H_train_train, H_test_train, H_inv, config):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    n = X_train.shape[0]
    # H_train_train = get_kernel_matrix(X_train, X_train)
    # (n, n)
    # H_test_train = get_kernel_matrix(X_train, X_test)
    # (b, n)
    label_inner_products = np.dot(np.dot(H_inv, np.dot(Y_train, np.transpose(Y_train))), H_inv)
    # (n, n)
    sigma_square = np.square(np.max(H_train_train) - np.min(H_train_train))
    const_H_square_y = np.sum(H_train_train * H_train_train * label_inner_products)
    const_H_y = np.sum(H_train_train * label_inner_products)
    const_y = np.sum(label_inner_products)
    const_H_square = np.sum(H_train_train * H_train_train)
    const_H = np.sum(H_train_train)

    train_label_kernels = (sigma_square * const_y - const_y * H_train_train * H_train_train + 2 * const_H_y * H_train_train -
                     const_H_square_y) / (n * n * sigma_square - n * n * H_train_train * H_train_train +
                                          2 * const_H * H_train_train - const_H_square)

    test_label_kernels = (sigma_square * const_y - const_y * H_test_train * H_test_train + 2 * const_H_y * H_test_train -
                     const_H_square_y) / (n * n * sigma_square - n * n * H_test_train * H_test_train +
                                          2 * const_H * H_test_train - const_H_square)
    # for i in range(label_kernels.shape[0]):
    #    print('label dia', label_kernels[i][i])
    # label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    # label_ratio = 1.0

    return train_label_kernels, test_label_kernels


def get_ntl_kernel_V3_pre(X_train, X_test, Y_train, H_train_train, H_test_train, H_inv, config):
    n = X_train.shape[0]
    b = X_test.shape[0]
    F_train_train = []
    F_test_train = []
    for i in range(n):
        for j in range(n):
            cur_feature = []
            cur_feature.extend(list(X_train[i]))
            cur_feature.extend(list(X_train[j]))
            cur_feature.extend(list(X_train[i] * X_train[j]))
            cur_feature.append(H_train_train[i, j])
            F_train_train.append(cur_feature)
    for i in range(b):
        for j in range(n):
            cur_feature = []
            cur_feature.extend(list(X_test[i]))
            cur_feature.extend(list(X_train[j]))
            cur_feature.extend(list(X_test[i] * X_train[j]))
            cur_feature.append(H_test_train[i, j])
            F_test_train.append(cur_feature)
    F_train_train = np.array(F_train_train)
    # (n, n, p)
    F_test_train = np.array(F_test_train)
    # (b, n, p)
    F_train_train = np.transpose(F_train_train, (2, 0, 1))
    # (p, n, n)
    F_test_train = np.transpose(F_test_train, (0, 2, 1))
    # (b, p, n)
    print('F train train', F_train_train.shape)
    print('F test train', F_test_train)
    H_inv_y = np.dot(H_inv, Y_train).reshape(-1)
    # (n)
    Z = np.dot(np.dot(H_inv_y, F_train_train), H_inv_y)
    # (p)
    print('Z', Z.shape)
    Z_norm = np.dot(np.dot(np.ones(n), F_train_train), np.ones(n))
    # (p)
    print('Z_norm', Z_norm.shape)
    kernel_norm = np.sum(np.dot(Z_norm, F_test_train))
    train_kernel_norm = np.sum(np.dot(Z_norm, F_train_train))
    test_label_kernels = np.dot(Z, F_test_train) / kernel_norm
    train_label_kernels = np.dot(Z, F_train_train) / train_kernel_norm
    print('train label kernels shape', train_label_kernels.shape)
    print('test label kernels shape', test_label_kernels.shape)
    label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    # label_ratio = 1.0
    return train_label_kernels, test_label_kernels, label_ratio


def get_ntl_kernel(label_kernels, label_ratio, H_test_train):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    kernel_values = H_test_train + label_ratio * label_kernels
    # kernel_values = label_kernels
    print('H', H_test_train)
    print('label kernels', label_kernels * label_ratio)
    print('kernel values', kernel_values)
    return kernel_values


def get_ntl_kernel_V4_pre(X_train, Y_train, H_train_train, H_test_train, H_inv, config):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    n = X_train.shape[0]
    b = H_test_train.shape[0]
    # H_train_train = get_kernel_matrix(X_train, X_train)
    # (n, n)
    # H_test_train = get_kernel_matrix(X_train, X_test)
    # (b, n)
    label_inner_products = np.dot(np.dot(H_inv, np.dot(Y_train, np.transpose(Y_train))), H_inv)
    # (n, n)
    label_mean = np.mean(label_inner_products, axis=1)
    # (n)
    print('label mean', label_mean)
    H_mean = np.mean(H_test_train, axis=1)
    # (b)
    print('H mean', H_mean)
    H_mean_train = np.mean(H_train_train, axis=1)
    # (n)
    H_mean_row = H_mean.reshape(-1, 1) + np.zeros((b, n))
    # (b, n)
    H_mean_column = H_mean_train.reshape(1, -1) + np.zeros((b, n))
    # (b, n)
    print('H mean train', H_mean_train)
    H_mean_row_train = H_mean_train.reshape(1, -1) + np.zeros((n, n))
    # (n, n)
    H_mean_column_train = H_mean_train.reshape(-1, 1) + np.zeros((n, n))
    # (n, n)

    H_y_mean = np.dot(H_test_train, label_mean).reshape(-1)
    # (b)
    print('H_y_mean', H_y_mean)
    H_y_mean_train = np.dot(H_train_train, label_mean).reshape(-1)
    # (n)
    H_y_mean_row = H_y_mean.reshape(-1, 1) + np.zeros((b, n))
    # (b, n)
    H_y_mean_column = H_y_mean_train.reshape(1, -1) + np.zeros((b, n))
    # (b, n)
    print('H_y_mean_train', H_y_mean_train)
    H_y_mean_row_train = H_y_mean_train.reshape(1, -1) + np.zeros((n, n))
    # (n, n)
    H_y_mean_column_train = H_y_mean_train.reshape(-1, 1) + np.zeros((n, n))
    # (n, n)

    train_label_kernels = (H_y_mean_row_train + H_y_mean_column_train) / (H_mean_row_train + H_mean_column_train)
    train_label_kernels /= n
    # (n, n)

    test_label_kernels = (H_y_mean_row + H_y_mean_column) / (H_mean_row + H_mean_column)
    test_label_kernels /= n
    # (b, n)

    label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    # label_ratio = 1.0
    print('train label kernels', train_label_kernels.shape)
    print('test label kernels', test_label_kernels.shape)
    return train_label_kernels, test_label_kernels, label_ratio


def run_ntl_experiments():
    neg_label = int(sys.argv[1].split('=')[1])
    pos_label = int(sys.argv[2].split('=')[1])
    # ntl_version = sys.argv[3].split('=')[1]
    # binary_sample_size = int(sys.argv[3].split('=')[1])
    print('neg label', neg_label)
    print('pos label', pos_label)
    # print('ntl version', ntl_version)
    assert neg_label < pos_label
    # print('binary sample size', binary_sample_size)
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    config = {'sample_size': 10000, 'seed': 66, 'dataset': 'CIFAR10', 'input_size': 3 * 32 * 32 + 1,
              'hidden_size': 4096, 'epoch_num': 500, 'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'lr': 0.5, 'binary_labels': [str(neg_label), str(pos_label)],
              'dir_path': '/path/to/working/dir', 'label_ratio': 0.0005, 'ntl_version': 'V6',
              'train': True}
    print('label ratio', config['label_ratio'])
    train_kernel_path = config['dir_path'] + '/data/kernel_matrix/train_kernel_CNN_' + str(neg_label) + '_' + \
                        str(pos_label) + '.pickle'
    test_kernel_path = config['dir_path'] + '/data/kernel_matrix/test_kernel_CNN_' + str(neg_label) + '_' + \
                       str(pos_label) + '.pickle'
    train_label_kernel_path = config['dir_path'] + '/data/kernel_matrix/train_label_kernel_CNN_' + str(neg_label) + \
                              '_' + str(pos_label) + '_global_fine.pickle'
    test_label_kernel_path = config['dir_path'] + '/data/kernel_matrix/test_label_kernel_CNN_' + str(neg_label) + '_' \
                             + str(pos_label) + '_global_fine.pickle'
    pickle_in_train = open(train_kernel_path, 'rb')
    pickle_in_test = open(test_kernel_path, 'rb')
    pickle_in_train_label = open(train_label_kernel_path, 'rb')
    pickle_in_test_label = open(test_label_kernel_path, 'rb')
    H_train_train = pickle.load(pickle_in_train)
    pickle_in_train.close()
    H_test_train = pickle.load(pickle_in_test)
    pickle_in_test.close()

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
    # train_data = train_data[:binary_sample_size]
    # test_data = test_data[:100]
    # train_data = train_data[:100]
    # test_data = test_data[:100]
    print('train data size', len(train_data))
    print('test data size', len(test_data))
    print('process data to numpy format')
    total_inputs_train, total_outputs_train = process_data_numpy(train_data)
    total_inputs_test, total_outputs_test = process_data_numpy(test_data)
    print('train data size', total_inputs_train.shape, total_outputs_train.shape)
    print('test data size', total_inputs_test.shape, total_outputs_test.shape)
    print('train inputs', total_inputs_train)
    print('train outputs', total_outputs_train)
    X_train_norm = np.linalg.norm(total_inputs_train, axis=1).reshape(-1, 1)
    X_test_norm = np.linalg.norm(total_inputs_test, axis=1).reshape(-1, 1)
    norm_products_train = np.dot(X_train_norm, np.transpose(X_train_norm))
    norm_products_test = np.dot(X_test_norm, np.transpose(X_train_norm))
    inner_products_train = np.dot(total_inputs_train, np.transpose(total_inputs_train))
    inner_products_test = np.dot(total_inputs_test, np.transpose(total_inputs_train))
    cos_values_train = np.clip(inner_products_train / norm_products_train, -1.0, 1.0)
    cos_values_test = np.clip(inner_products_test / norm_products_test, -1.0, 1.0)
    norm_sum_train = np.square(X_train_norm).reshape(-1, 1) + np.square(X_train_norm).reshape(1, -1)
    norm_sum_test = np.square(X_test_norm).reshape(-1, 1) + np.square(X_train_norm).reshape(1, -1)
    euc_distance_train = norm_sum_train - 2 * inner_products_train
    euc_distance_test = norm_sum_test - 2 * inner_products_test
    # get train kernel values
    # config['train'] = True
    print(config['ntl_version'])
    if config['ntl_version'] == 'V1':
        train_label_kernels, test_label_kernels = \
            get_ntl_kernel_V1_pre(total_inputs_train, total_outputs_train, H_train_train, H_test_train, config)
        # train_label_kernels = ((train_label_kernels >= 0) * 2 - 1)
        # train_label_kernels = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        # test_label_kernels = ((test_label_kernels >= 0) * 2 - 1)
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    elif config['ntl_version'] == 'V2':
        H_inv = np.linalg.inv(H_train_train)
        train_label_kernels, test_label_kernels = \
            get_ntl_kernel_V2_pre(total_inputs_train, total_outputs_train, H_train_train, H_test_train, H_inv,
                                  config)
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    elif config['ntl_version'] == 'V3':
        H_inv = np.linalg.inv(H_train_train)
        train_label_kernels, test_label_kernels, label_ratio = \
            get_ntl_kernel_V3_pre(total_inputs_train, total_inputs_test, total_outputs_train, H_train_train,
                                  H_test_train, H_inv, config)
    elif config['ntl_version'] == 'V4':
        H_inv = np.linalg.inv(H_train_train)
        train_label_kernels, test_label_kernels, label_ratio = \
            get_ntl_kernel_V4_pre(total_inputs_train, total_outputs_train, H_train_train, H_test_train, H_inv, config)
    elif config['ntl_version'] == 'V5':
        train_label_kernels = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        train_kernel_ntk = get_kernel_matrix(total_inputs_train, total_inputs_train)
        test_kernel_ntk = get_kernel_matrix(total_inputs_train, total_inputs_test)
        transform_matrix_ntk = np.linalg.solve(train_kernel_ntk, total_outputs_train)
        test_output_pred = np.dot(test_kernel_ntk, transform_matrix_ntk)
        print('test output pred', test_output_pred)
        test_output_pred = ((test_output_pred >= 0) * 2 - 1)
        print('test output pred', test_output_pred)
        print('test accuracy', np.mean(test_output_pred == total_outputs_test))
        test_label_kernels = np.dot(test_output_pred, np.transpose(total_outputs_train))
        # test_label_kernels = np.dot(total_outputs_test, np.transpose(total_outputs_train))
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    elif config['ntl_version'] == 'V6':
        '''F_train_train, F_test_train = get_pca_features(total_inputs_train, total_inputs_test)
        train_label_kernels = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        train_train_features = np.concatenate((H_train_train.reshape(-1, 1), inner_products_train.reshape(-1, 1)),
                                              axis=1)
        train_train_features = np.concatenate((train_train_features, cos_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, euc_distance_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, F_train_train), axis=1)
        test_train_features = np.concatenate((H_test_train.reshape(-1, 1), inner_products_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, cos_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, euc_distance_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, F_test_train), axis=1)
        train_label_kernels, test_label_kernels = get_pair_predictions(train_train_features, test_train_features,
                                                                       train_label_kernels.reshape(-1, 1))
        train_label_kernels = train_label_kernels.reshape(-1, H_train_train.shape[1])
        test_label_kernels = test_label_kernels.reshape(-1, H_test_train.shape[1])'''
        train_label_kernels = pickle.load(pickle_in_train_label)
        pickle_in_train_label.close()
        test_label_kernels = pickle.load(pickle_in_test_label)
        pickle_in_test_label.close()
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    elif config['ntl_version'] == 'V7':
        train_label_kernels = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        test_label_kernels = pickle.load(pickle_in_test_label)
        test_label_kernels = (test_label_kernels >= 0) * 2 - 1
        pickle_in_test_label.close()
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    elif config['ntl_version'] == 'V8':
        F_train_train, F_test_train = get_pca_features(total_inputs_train, total_inputs_test)
        train_label_products = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        train_train_features = np.concatenate((H_train_train.reshape(-1, 1), inner_products_train.reshape(-1, 1)),
                                                      axis=1)
        train_train_features = np.concatenate((train_train_features, cos_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, euc_distance_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, F_train_train), axis=1)
        test_train_features = np.concatenate((H_test_train.reshape(-1, 1), inner_products_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, cos_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, euc_distance_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, F_test_train), axis=1)
        # neg=4 pos=5 weights
        weights = [[3.93637967e+01], [-1.03960164e-03], [3.35983230e-01], [-1.64769767e-04], [2.15114289e-04],
                   [1.97015315e-04], [8.07369052e-04], [-2.04074264e-04], [3.82644368e-04], [5.12423102e-05],
                   [5.18928807e-04], [2.31160302e-04], [4.29041704e-04], [-1.03988695e-03], [-4.22337358e-04],
                   [-3.59228823e-04], [5.93590440e-04], [9.35286507e-03], [-2.21214699e-05], [1.99646283e-03],
                   [1.75735997e-04], [4.45975028e-04], [-5.09241795e-04], [8.03083360e-03], [1.81681319e-03],
                   [-1.77126432e-03], [7.61548256e-04], [-1.34766112e-03], [-1.78664585e-03], [6.69717093e-04],
                   [1.60599489e-03], [-2.58497242e-03], [4.30134899e-03], [3.81611411e-04], [3.71937740e-05],
                   [4.72220704e-04], [-3.85077832e-03], [-3.48423904e-03], [-1.67094920e-04], [6.95591567e-04],
                   [-1.23503008e-03], [-1.75690109e-03], [-9.36692840e-04], [-7.60809579e-03], [-5.47028707e-04],
                   [-1.19679562e-03], [1.10879859e-03], [-2.36457944e-03], [3.62845427e-04], [2.83950067e-03],
                   [3.95116944e-03], [-4.29303095e-03], [-2.23469204e-03], [5.21000625e-03], [3.33396223e-03],
                   [-1.37641279e-03], [4.97276760e-03], [3.62236619e-04], [-5.93734440e-04], [-6.98010647e-04],
                   [5.30712130e-03], [-1.20188186e-03], [6.19141611e-03], [-1.26182035e-03]]
        test_label_kernels = np.dot(test_train_features, weights)
        train_label_kernels = np.dot(train_train_features, weights)
        print('train accuracy', np.mean((train_label_kernels >= 0) == (train_label_products >= 0)))
        train_label_kernels = train_label_kernels.reshape(-1, H_train_train.shape[1])
        test_label_kernels = test_label_kernels.reshape(-1, H_test_train.shape[1])
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    elif config['ntl_version'] == 'V9':
        '''F_train_train, F_test_train = get_pca_features(total_inputs_train, total_inputs_test)
        train_label_kernels = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        train_train_features = np.concatenate((H_train_train.reshape(-1, 1), inner_products_train.reshape(-1, 1)),
                                              axis=1)
        train_train_features = np.concatenate((train_train_features, cos_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, euc_distance_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, F_train_train), axis=1)
        test_train_features = np.concatenate((H_test_train.reshape(-1, 1), inner_products_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, cos_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, euc_distance_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, F_test_train), axis=1)
        train_label_kernels, test_label_kernels = get_pair_predictions(train_train_features, test_train_features,
                                                                       train_label_kernels.reshape(-1, 1))
        train_label_kernels = train_label_kernels.reshape(-1, H_train_train.shape[1])
        test_label_kernels = test_label_kernels.reshape(-1, H_test_train.shape[1])'''
        train_label_kernels = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        test_label_products = np.dot(total_outputs_test, np.transpose(total_outputs_test))
        pickle_in_train_label.close()
        test_label_kernels = pickle.load(pickle_in_test_label)
        test_label_kernels = (test_label_kernels >= 0) * 2 - 1
        print('regression accuracy', np.mean((test_label_kernels >= 0) == (test_label_products >= 0)))
        pickle_in_test_label.close()
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']

    train_kernel_values = get_ntl_kernel(train_label_kernels, label_ratio, H_train_train)
    print('train kernel values size', train_kernel_values)
    # get transform matrix
    transform_matrix = get_transform_matrix(train_kernel_values, total_outputs_train)
    print('transform matrix size', transform_matrix)
    # get test kernel values
    # config['train'] = False
    test_kernel_values = get_ntl_kernel(test_label_kernels, label_ratio, H_test_train)
    print('test kernel values size', test_kernel_values)
    # get test outputs
    test_outputs = kernel_regression(test_kernel_values, transform_matrix)
    print('test outputs size', test_outputs)
    # get test accuracy
    test_accuracy = get_test_accuracy(test_outputs, total_outputs_test)
    print('test accuracy', test_accuracy)
    '''
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
    run_ntl_experiments()

