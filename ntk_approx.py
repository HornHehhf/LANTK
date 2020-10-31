import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


from models import MLPNetNoBiasFalse
from data import save_sampled_binary_data, load_sampled_binary_data_from_pickle
from utils import set_random_seed, add_one_extra_dim, get_minibatches_idx
from ntk import process_data_numpy, get_transform_matrix, kernel_regression, get_test_accuracy, \
    get_same_class_relative_ratio

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def relu(X):
    return np.maximum(0, X)


def build_model(config):
    model = MLPNetNoBiasFalse(config['input_size'], config['hidden_size'])
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    return model, loss_function, optimizer


def simple_train_batch(trainloader, testloader, model, loss_function, optimizer, config):
    total_loss_list = []
    for epoch in range(config['epoch_num']):
        model.train()
        # print('current epoch: ', epoch)
        total_loss = 0
        minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=config['simple_train_batch_size'],
                                              shuffle=True)
        # BCE, (100, 1, 1) doesn't matter
        # MSE, (100, 1, 1) matter
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(), Variable(targets.float().cuda()).squeeze()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
        total_loss_list.append(total_loss.cpu().data.item())
        if epoch % (config['epoch_num'] // 10) == 0 or epoch == config['epoch_num'] - 1:
            print('train loss', total_loss)
            train_accuracy, pred_np = simple_test_batch(trainloader, model, config)
            print('train accuracy', train_accuracy)
            test_accuracy, pred_np = simple_test_batch(testloader, model, config)
            print('test accuracy', test_accuracy)
    return total_loss_list


def simple_test_batch(testloader, model, config):
    model.eval()
    total = 0.0
    correct = 0.0
    pred_np = []
    minibatches_idx = get_minibatches_idx(len(testloader), minibatch_size=config['simple_test_batch_size'],
                                          shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([list(testloader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(np.array([list(testloader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.cuda()).squeeze(1)
        outputs = model(inputs)
        pred_np.extend(list(outputs.cpu().data.numpy()))
        predicted = (outputs >= 0.5).long().squeeze()
        total += targets.size(0)
        correct += predicted.eq(targets.long()).sum().item()
    test_accuracy = correct / total
    pred_np = np.array(pred_np)
    return test_accuracy, pred_np


def get_ntk_approx(X_train, X_test, w, a):
    # inputs: X_train (n, d), X_test (b, d), w (m, d), a (1, m)
    # outputs: kernel_values (b, n)
    extend_a_train = np.zeros((X_train.shape[0], a.shape[1])) + a
    # (n, m)
    Hidden_train = np.dot(X_train, np.transpose(w))
    # (n, m)
    H_train = relu(Hidden_train)
    # (n, m)
    L_train = (Hidden_train >= 0) * extend_a_train
    # (n, m)

    extend_a_test = np.zeros((X_test.shape[0], a.shape[1])) + a
    # (b, m)
    Hidden_test = np.dot(X_test, np.transpose(w))
    # (b, m)
    H_test = relu(Hidden_test)
    # (b, m)
    L_test = (Hidden_test >= 0) * extend_a_test
    # (b, m)

    X_products = np.dot(X_test, np.transpose(X_train))
    # (b, n)
    H_products = np.dot(H_test, np.transpose(H_train))
    # (b, n)
    L_products = np.dot(L_test, np.transpose(L_train))
    # (b, n)

    kernel_values = (X_products * L_products + H_products) / a.shape[1]
    # (b, n)

    for i in range(len(kernel_values)):
        for j in range(len(kernel_values[0])):
            if np.isnan(kernel_values[i][j]) or np.isinf(kernel_values[i][j]):
                print(kernel_values[i][j])
                print(i, j)
    print('min', np.min(kernel_values))
    print('max', np.max(kernel_values))
    print('mean', np.mean(kernel_values))

    return kernel_values


def run_ntk_approx_experiments():
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
              'hidden_size': 40960, 'epoch_num': 10, 'simple_train_batch_size': 128, 'simple_test_batch_size': 100,
              'lr': 3e-4, 'binary_labels': [str(neg_label), str(pos_label)],
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
    print('build model')
    model, loss_function, optimizer = build_model(config)
    print('test model')
    test_accuracy, pred_np = simple_test_batch(test_data, model, config)
    print('test accuracy', test_accuracy)
    w = model.fc1.weight.cpu().data.numpy()
    a = model.fc2.weight.cpu().data.numpy()
    print('w', w.shape, w)
    print('a', a.shape, a)
    # get train kernel values
    train_kernel_values = get_ntk_approx(total_inputs_train, total_inputs_train, w, a)
    print('train kernel values size', train_kernel_values)
    # get transform matrix
    transform_matrix = get_transform_matrix(train_kernel_values, total_outputs_train)
    print('transform matrix size', transform_matrix)
    # get test kernel values
    test_kernel_values = get_ntk_approx(total_inputs_train, total_inputs_test, w, a)
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
    print('initialized parameters')
    print('train same class ratio', train_same_class_ratio)
    test_test_kernel_values = get_ntk_approx(total_inputs_test, total_inputs_test, w, a)
    test_train_same_class_ratio = get_same_class_relative_ratio(test_kernel_values, test_test_kernel_values,
                                                                train_kernel_values, total_outputs_test,
                                                                total_outputs_train)
    print('test train same class ratio', test_train_same_class_ratio)
    print('train model')
    total_loss_list = simple_train_batch(train_data, test_data, model, loss_function, optimizer, config)
    print('total loss list', total_loss_list)
    print('test model')
    test_accuracy, pred_np = simple_test_batch(test_data, model, config)
    print('test accuracy', test_accuracy)
    w = model.fc1.weight.cpu().data.numpy()
    a = model.fc2.weight.cpu().data.numpy()
    print('w', w.shape, w)
    print('a', a.shape, a)
    # get train kernel values
    train_kernel_values = get_ntk_approx(total_inputs_train, total_inputs_train, w, a)
    print('train kernel values size', train_kernel_values)
    # get transform matrix
    transform_matrix = get_transform_matrix(train_kernel_values, total_outputs_train)
    print('transform matrix size', transform_matrix)
    # get test kernel values
    test_kernel_values = get_ntk_approx(total_inputs_train, total_inputs_test, w, a)
    print('test kernel values size', test_kernel_values)
    # get test outputs
    test_outputs = kernel_regression(test_kernel_values, transform_matrix)
    print('test outputs size', test_outputs)
    # get test accuracy
    test_accuracy = get_test_accuracy(test_outputs, total_outputs_test)
    print('test accuracy', test_accuracy)
    # get same class ratio
    train_same_class_ratio_new = get_same_class_relative_ratio(train_kernel_values, train_kernel_values,
                                                               train_kernel_values, total_outputs_train,
                                                               total_outputs_train)
    print('trained parameters')
    print('train same class ratio', train_same_class_ratio_new)
    test_test_kernel_values = get_ntk_approx(total_inputs_test, total_inputs_test, w, a)
    test_train_same_class_ratio_new = get_same_class_relative_ratio(test_kernel_values, test_test_kernel_values,
                                                                    train_kernel_values, total_outputs_test,
                                                                    total_outputs_train)
    print('test train same class ratio', test_train_same_class_ratio_new)

    print('train same class ratio diff', train_same_class_ratio_new - train_same_class_ratio)
    print('test train same class ratio diff', test_train_same_class_ratio_new - test_train_same_class_ratio)


if __name__ == '__main__':
    run_ntk_approx_experiments()
