import numpy as np
import sys


from data import save_sampled_binary_data, load_sampled_binary_data_from_pickle
from utils import set_random_seed, add_one_extra_dim


def get_same_class_relative_ratio(kernel_values, square_a, square_b, Y_a, Y_b):
    # kernel_values: b x n, square_a: b x b, square_b: nxn, Y_a: b x 1, Y_b: n x 1
    print('kernel values', kernel_values.shape)
    normalized_kernel_values = np.zeros_like(kernel_values)
    for a in range(len(kernel_values)):
        for b in range(len(kernel_values[0])):
            normalized_kernel_values[a][b] = kernel_values[a][b] / np.sqrt(square_a[a][a] * square_b[b][b])
            if normalized_kernel_values[a][b] is np.NaN:
                print('kernel', kernel_values[a][b])
                print('square_a', square_a[a][a])
                print('square_b', square_b[b][b])

    same_class = []
    between_class = []
    for a in range(len(Y_a)):
        for b in range(len(Y_b)):
            if Y_a[a][0] == Y_b[b][0]:
                same_class.append(normalized_kernel_values[a][b])
            else:
                between_class.append(normalized_kernel_values[a][b])
    print('same class', len(same_class), np.min(same_class), np.mean(same_class), np.max(same_class))
    print('between class', len(between_class), np.min(between_class), np.mean(between_class), np.max(between_class))
    ratio = np.mean(same_class) / (np.mean(same_class) + np.mean(between_class))
    print('relative ratio', ratio)
    return ratio


def process_data_numpy(data):
    # print(data[0][0].size())
    total_inputs = np.zeros((len(data), data[0][0].size()[1]))
    total_outputs = np.zeros((len(data), 1))
    for x in range(len(data)):
        total_inputs[x] = data[x][0].cpu().data.numpy().reshape(-1)
        # total_outputs[x] = data[x][1].cpu().data.numpy().reshape(-1) - np.ones(data[0][1].size()[1]) * 0.1
        total_outputs[x] = (data[x][1].cpu().data.numpy().reshape(-1) - 0.5) * 2
    return total_inputs, total_outputs


def get_kernel_matrix(X_train, X_test):
    # inputs: X_train (n, d), X_test (b, d)
    # outputs: kernel_values (b, n)
    X_train_norm = np.linalg.norm(X_train, axis=1).reshape(-1, 1)
    # (n, 1)
    # print('X_train_norm', X_train_norm)
    X_test_norm = np.linalg.norm(X_test, axis=1).reshape(-1, 1)
    # (b, 1)
    # print('X_test_norm', X_test_norm)
    inner_products = np.dot(X_test, np.transpose(X_train))
    # (b, n)
    # print('inner products', inner_products)
    norm_products = np.dot(X_test_norm, np.transpose(X_train_norm))
    # (b, n)
    # print('norm products', norm_products)
    cos_values = np.clip(inner_products / norm_products, -1.0, 1.0)
    # (b, n)
    # print('cos values', cos_values)
    sin_values = np.sqrt(1 - np.square(cos_values))
    # (b, n)
    # print('sin values', sin_values)
    theta_values = np.arccos(cos_values)
    # (b, n)
    # print('theta values', theta_values)
    kernel_values = inner_products * (np.pi - theta_values) / (np.pi * 2) + \
                    norm_products * (sin_values + (np.pi - theta_values) * cos_values) / (np.pi * 2)
    # (b, n)
    # for i in range(len(kernel_values)):
    #     for j in range(len(kernel_values[0])):
    #         if np.isnan(kernel_values[i][j]) or np.isinf(kernel_values[i][j]):
    #             print(kernel_values[i][j])
    #             print(i, j)
    print('min', np.min(kernel_values))
    print('max', np.max(kernel_values))
    print('mean', np.mean(kernel_values))
    return kernel_values


def kernel_regression(test_kernel_values, transform_matrix):
    # inputs: kernel_values (b, n), transform_matrix (n, c)
    # outputs: test_outputs (b, c)
    test_outputs = np.dot(test_kernel_values, transform_matrix)
    # (b, c)
    return test_outputs


def get_transform_matrix(train_kernel_values, train_outputs):
    # inputs: train_kernel_values (n, n), train_outputs (n, c)
    # outputs: transform_matrix (n, c)
    # get the inverse
    # inverse_H = np.linalg.inv(train_kernel_values)
    # print('inverse_H', inverse_H)
    # (n, n)
    # transform_matrix = np.dot(inverse_H, train_outputs)
    # print('train kernel values matrix rank', np.linalg.matrix_rank(train_kernel_values))
    transform_matrix = np.linalg.solve(train_kernel_values, train_outputs)
    # (n, c)
    print('HT', np.dot(train_kernel_values, transform_matrix))
    return transform_matrix


def get_test_accuracy(test_outputs, test_targets):
    total = test_outputs.shape[0]
    pred = (test_outputs >= 0)
    # (n, 1)
    gold = (test_targets >= 0)
    # (n, 1)
    print('pred', pred)
    print('gold', gold)
    correct = np.sum(pred == gold)
    return correct / total


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

    print('set random seed', config['seed'])
    set_random_seed(config['seed'])
    print('sample data')
    save_sampled_binary_data(config)
    print('set random seed', config['seed'])
    set_random_seed(config['seed'])
    print('load data')
    train_data, test_data = load_sampled_binary_data_from_pickle(config)
    train_data = add_one_extra_dim(train_data)
    test_data = add_one_extra_dim(test_data)
    # train_data = train_data[:binary_sample_size]
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
    train_kernel_values = get_kernel_matrix(total_inputs_train, total_inputs_train)
    print('train kernel values size', train_kernel_values)
    # get transform matrix
    transform_matrix = get_transform_matrix(train_kernel_values, total_outputs_train)
    print('transform matrix size', transform_matrix)
    # get test kernel values
    test_kernel_values = get_kernel_matrix(total_inputs_train, total_inputs_test)
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
    run_ntk_experiments()

