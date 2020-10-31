import sys
import numpy as np
import pickle


from data import save_sampled_binary_data, load_sampled_binary_data_from_pickle
from utils import set_random_seed
from ntk import get_transform_matrix, get_test_accuracy, get_same_class_relative_ratio, kernel_regression
from jax_models import MLPNet, WideResnet, CNN


def process_data_numpy(data, option='normal'):
    # print(data[0][0].size())
    total_inputs = np.zeros((len(data), data[0][0].size()[1], data[0][0].size()[2], data[0][0].size()[3]))
    total_outputs = np.zeros((len(data), 1))
    for x in range(len(data)):
        total_inputs[x] = data[x][0].cpu().data.numpy()
        # total_outputs[x] = data[x][1].cpu().data.numpy().reshape(-1) - np.ones(data[0][1].size()[1]) * 0.1
        total_outputs[x] = (data[x][1].cpu().data.numpy().reshape(-1) - 0.5) * 2
    total_inputs = np.transpose(total_inputs, (0, 2, 3, 1))
    if option == 'reshape':
        total_inputs = total_inputs.reshape(len(data), -1)
    return total_inputs, total_outputs


def get_kernel_matrix_google(X_train, X_test):
    # inputs: X_train (n, d), X_test (b, d)
    # outputs: kernel_values (b, n)
    # init_fn, apply_fn, kernel_fn = MLPNet()
    init_fn, apply_fn, kernel_fn = CNN(block_size=1, k=1, num_classes=1)
    # kernel_values = kernel_fn(X_test, X_train, 'ntk')
    kernel_values = np.zeros((X_test.shape[0], X_train.shape[0]))
    batch_size = 1000
    # batch_size = 500
    for i in range(X_test.shape[0] // batch_size):
        print(i)
        # for j in range(X_train.shape[0] // batch_size):
        # print(i, j)
        # kernel_values[i * batch_size: i * batch_size + batch_size, j * batch_size: j * batch_size + batch_size] = \
        #       kernel_fn(X_test[i * batch_size: i * batch_size + batch_size], X_train[j * batch_size:
        #        j * batch_size + batch_size], 'ntk')  # (b, n) np.ndarray
        kernel_values[i * batch_size: i * batch_size + batch_size] = \
            kernel_fn(X_test[i * batch_size: i * batch_size + batch_size], X_train, 'ntk')
    print('zero num in kernel values', np.sum(kernel_values == 0))
    return kernel_values


def run_ntk_experiments():
    neg_label = int(sys.argv[1].split('=')[1])
    pos_label = int(sys.argv[2].split('=')[1])
    # binary_sample_size = int(sys.argv[3].split('=')[1])
    print('neg label', neg_label)
    print('pos label', pos_label)
    assert neg_label < pos_label
    # print('binary sample size', binary_sample_size)
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    config = {'sample_size': 10000, 'seed': 66, 'dataset': 'CIFAR10', 'input_size': 3 * 32 * 32 + 1,
              'hidden_size': 4096, 'epoch_num': 500, 'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'lr': 0.5, 'binary_labels': [str(neg_label), str(pos_label)],
              'dir_path': '/path/to/working/dir'}
    train_kernel_path = config['dir_path'] + '/data/kernel_matrix/train_kernel_CNN_' + str(neg_label) + '_' + \
                        str(pos_label) + '.pickle'
    test_kernel_path = config['dir_path'] + '/data/kernel_matrix/test_kernel_CNN_' + str(neg_label) + '_' + \
                       str(pos_label) + '.pickle'
    pickle_out_train = open(train_kernel_path, 'wb')
    pickle_out_test = open(test_kernel_path, 'wb')
    pickle_in_train = open(train_kernel_path, 'rb')
    pickle_in_test = open(test_kernel_path, 'rb')

    print('set random seed', config['seed'])
    set_random_seed(config['seed'])
    print('sample data')
    save_sampled_binary_data(config)
    print('set random seed', config['seed'])
    set_random_seed(config['seed'])
    print('load data')
    train_data, test_data = load_sampled_binary_data_from_pickle(config)
    # train_data = train_data[:binary_sample_size]
    # test_data = test_data[:100]
    # train_data = train_data[:1]
    # test_data = test_data[:1]
    print('train data size', len(train_data))
    print('test data size', len(test_data))
    print('process data to numpy format')
    total_inputs_train, total_outputs_train = process_data_numpy(train_data, 'normal')
    total_inputs_test, total_outputs_test = process_data_numpy(test_data, 'normal')
    print('train data size', total_inputs_train.shape, total_outputs_train.shape)
    print('test data size', total_inputs_test.shape, total_outputs_test.shape)
    print('train inputs', total_inputs_train)
    print('train outputs', total_outputs_train)
    # get train kernel values
    train_kernel_values = get_kernel_matrix_google(total_inputs_train, total_inputs_train)
    pickle.dump(train_kernel_values, pickle_out_train)
    pickle_out_train.close()
    train_kernel_values = pickle.load(pickle_in_train)
    pickle_in_train.close()
    print('train kernel values size', train_kernel_values)
    # get transform matrix
    transform_matrix = get_transform_matrix(train_kernel_values, total_outputs_train)
    print('transform matrix size', transform_matrix)
    # get test kernel values
    test_kernel_values = get_kernel_matrix_google(total_inputs_train, total_inputs_test)
    pickle.dump(test_kernel_values, pickle_out_test)
    pickle_out_test.close()
    test_kernel_values = pickle.load(pickle_in_test)
    pickle_in_test.close()
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
    test_test_kernel_values = get_kernel_matrix_google(total_inputs_test, total_inputs_test)
    test_train_same_class_ratio = get_same_class_relative_ratio(test_kernel_values, test_test_kernel_values,
                                                                train_kernel_values, total_outputs_test,
                                                                total_outputs_train)
    print('test train same class ratio', test_train_same_class_ratio)
    '''


if __name__ == '__main__':
    run_ntk_experiments()
