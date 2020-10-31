import numpy as np
import sys
import pickle

from data import save_sampled_data, load_sampled_data_from_pickle
from utils import set_random_seed, add_one_extra_dim
from ntk import get_same_class_relative_ratio, get_kernel_matrix, kernel_regression, \
    get_transform_matrix, get_test_accuracy
from ntk_google_multi import process_data_numpy_multi, get_test_accuracy_multi


def get_ntl_kernel_V1_pre(X_train, Y_train, H_train_train, H_test_train, config):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    n = X_train.shape[0]
    # H_train_train = get_kernel_matrix(X_train, X_train)
    # (n, n)
    # H_test_train = get_kernel_matrix(X_train, X_test)
    # (b, n)
    label_inner_products = (np.dot(Y_train, np.transpose(Y_train)) - 0.5) * 2
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


def get_ntl_kernel_V2_pre(X_train, Y_train, H_train_train, H_test_train, H_inv, config):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    n = X_train.shape[0]
    # H_train_train = get_kernel_matrix(X_train, X_train)
    # (n, n)
    # H_test_train = get_kernel_matrix(X_train, X_test)
    # (b, n)
    label_inner_products = (np.dot(Y_train, np.transpose(Y_train)) - 0.5) * 2
    label_inner_products = np.dot(np.dot(H_inv, label_inner_products), H_inv)
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


def get_ntl_kernel(label_kernels, label_ratio, H_test_train):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    kernel_values = H_test_train + label_ratio * label_kernels
    # kernel_values = label_kernels
    print('H', H_test_train)
    print('label kernels', label_kernels * label_ratio)
    print('kernel values', kernel_values)
    return kernel_values


def run_ntl_experiments():
    sample_size = int(sys.argv[1].split('=')[1])
    ntl_version = sys.argv[2].split('=')[1]
    label_ratio = float(sys.argv[3].split('=')[1])
    print('sample size', sample_size)
    print('label ratio', label_ratio)
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    config = {'sample_size': sample_size, 'seed': 66, 'dataset': 'CIFAR10', 'input_size': 3 * 32 * 32 + 1,
              'hidden_size': 4096, 'epoch_num': 500, 'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'lr': 0.5,
              'dir_path': '/path/to/working/dir', 'label_ratio': label_ratio, 'ntl_version': ntl_version,
              'train': True, 'one-hot': True}
    train_kernel_path = config['dir_path'] + '/data/kernel_matrix/train_kernel_CNN_' + str(config['sample_size']) + \
                        '.pickle'
    test_kernel_path = config['dir_path'] + '/data/kernel_matrix/test_kernel_CNN_' + str(config['sample_size']) + \
                       '.pickle'
    train_label_kernel_path = config['dir_path'] + '/data/kernel_matrix/train_label_kernel_CNN_' + \
                              str(config['sample_size']) + ntl_version + '_global.pickle'
    test_label_kernel_path = config['dir_path'] + '/data/kernel_matrix/test_label_kernel_CNN_' + \
                             str(config['sample_size']) + ntl_version + '_global.pickle'
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
    # save_sampled_data(config, config['dir_path'])
    print('set random seed', config['seed'])
    set_random_seed(config['seed'])
    print('load data')
    train_data, test_data = load_sampled_data_from_pickle(config, config['dir_path'])
    # train_data = add_one_extra_dim(train_data)
    # test_data = add_one_extra_dim(test_data)
    # train_data = train_data[:binary_sample_size]
    # test_data = test_data[:100]
    # train_data = train_data[:1000]
    # test_data = test_data[:1000]
    print('train data size', len(train_data))
    print('test data size', len(test_data))
    print('process data to numpy format')
    total_inputs_train, total_outputs_train = process_data_numpy_multi(train_data, option='reshape')
    total_inputs_test, total_outputs_test = process_data_numpy_multi(test_data, option='reshape')
    print('train data size', total_inputs_train.shape, total_outputs_train.shape)
    print('test data size', total_inputs_test.shape, total_outputs_test.shape)
    print('train inputs', total_inputs_train)
    print('train outputs', total_outputs_train)
    print(config['ntl_version'])
    if config['ntl_version'] == 'V1':
        train_label_kernels, test_label_kernels = \
            get_ntl_kernel_V1_pre(total_inputs_train, total_outputs_train, H_train_train, H_test_train, config)
        # train_label_kernels = ((train_label_kernels >= 0) * 2 - 1)
        # train_label_kernels = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        # test_label_kernels = ((test_label_kernels >= 0) * 2 - 1)
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
        # label_ratio = 1.0
    elif config['ntl_version'] == 'V2':
        H_inv = np.linalg.inv(H_train_train)
        train_label_kernels, test_label_kernels = \
            get_ntl_kernel_V2_pre(total_inputs_train, total_outputs_train, H_train_train, H_test_train, H_inv,
                                  config)
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
        # label_ratio = 1.0
    elif config['ntl_version'] == 'V7' or 'V8':
        print('load label kernels')
        train_label_kernels = pickle.load(pickle_in_train_label)
        pickle_in_train_label.close()
        test_label_kernels = pickle.load(pickle_in_test_label)
        pickle_in_test_label.close()
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
        # label_ratio = 1.0
    elif config['ntl_version'] == 'V7' or 'V8':
        train_label_kernels = (np.dot(total_outputs_train, np.transpose(total_outputs_train)) - 0.5) * 2
        test_label_kernels = pickle.load(pickle_in_test_label)
        test_label_kernels = (test_label_kernels >= 0) * 2 - 1
        pickle_in_test_label.close()
        label_ratio = np.mean(np.abs(H_train_train)) / np.mean(np.abs(train_label_kernels)) * config['label_ratio']
        # label_ratio = 1.0

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
    test_accuracy = get_test_accuracy_multi(test_outputs, total_outputs_test)
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

