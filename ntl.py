import numpy as np
import sys


from data import save_sampled_binary_data, load_sampled_binary_data_from_pickle
from utils import set_random_seed, add_one_extra_dim
from ntk import get_same_class_relative_ratio, process_data_numpy, get_kernel_matrix, kernel_regression, \
    get_transform_matrix, get_test_accuracy


def get_ntl_kernel_V1(X_train, X_test, Y_train, config):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    n = X_train.shape[0]
    H_train_train = get_kernel_matrix(X_train, X_train)
    # (n, n)
    H_test_train = get_kernel_matrix(X_train, X_test)
    # (b, n)
    label_inner_products = np.dot(Y_train, np.transpose(Y_train))
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

    label_kernels = (sigma_square * const_y - const_y * H_test_train * H_test_train + 2 * const_H_y * H_test_train -
                     const_H_square_y) / (n * n * sigma_square - n * n * H_test_train * H_test_train +
                                          2 * const_H * H_test_train - const_H_square)
    # for i in range(label_kernels.shape[0]):
    #    print('label dia', label_kernels[i][i])
    label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
    kernel_values = H_test_train + label_ratio * label_kernels
    print('H', H_test_train)
    print('label kernels', label_kernels * label_ratio)
    print('kernel values', kernel_values)
    return kernel_values


def get_ntl_kernel_V2(X_train, X_test, Y_train, config):
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


def get_ntl_kernel(X_train, X_test, Y_train, config):
    if config['ntl_version'] == 'V1':
        return get_ntl_kernel_V1(X_train, X_test, Y_train, config)
    elif config['ntl_version'] == 'V2':
        return get_ntl_kernel_V2(X_train, X_test, Y_train, config)


def run_ntl_experiments():
    neg_label = int(sys.argv[1].split('=')[1])
    pos_label = int(sys.argv[2].split('=')[1])
    ntl_version = sys.argv[3].split('=')[1]
    # binary_sample_size = int(sys.argv[3].split('=')[1])
    print('neg label', neg_label)
    print('pos label', pos_label)
    print('ntl version', ntl_version)
    assert neg_label < pos_label
    # print('binary sample size', binary_sample_size)
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    config = {'sample_size': 10000, 'seed': 66, 'dataset': 'CIFAR10', 'input_size': 3 * 32 * 32 + 1,
              'hidden_size': 4096, 'epoch_num': 500, 'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'lr': 0.5, 'binary_labels': [str(neg_label), str(pos_label)],
              'dir_path': '/path/to/working/dir', 'label_ratio': 0.1, 'ntl_version': ntl_version,
              'train': True}

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
    # get train kernel values
    # config['train'] = True
    train_kernel_values = get_ntl_kernel(total_inputs_train, total_inputs_train, total_outputs_train, config)
    print('train kernel values size', train_kernel_values)
    # get transform matrix
    transform_matrix = get_transform_matrix(train_kernel_values, total_outputs_train)
    print('transform matrix size', transform_matrix)
    # get test kernel values
    # config['train'] = False
    test_kernel_values = get_ntl_kernel(total_inputs_train, total_inputs_test, total_outputs_train, config)
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


if __name__ == '__main__':
    run_ntl_experiments()

