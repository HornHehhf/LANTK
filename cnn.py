import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys

from models import CNN
from data import save_sampled_binary_data, load_sampled_binary_data_from_pickle
from utils import set_random_seed, get_minibatches_idx

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def build_model(config):
    model = CNN()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
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
        # if epoch % (config['epoch_num'] // 1) == 0 or epoch == config['epoch_num'] - 1:
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


def run_nn_experiments():
    neg_label = int(sys.argv[1].split('=')[1])
    pos_label = int(sys.argv[2].split('=')[1])
    # binary_sample_size = int(sys.argv[3].split('=')[1])
    print('neg label', neg_label)
    print('pos label', pos_label)
    # print('binary sample size', binary_sample_size)
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    config = {'sample_size': 10000, 'seed': 66, 'dataset': 'CIFAR10', 'input_size': 3 * 32 * 32 + 1,
              'hidden_size': 40960, 'epoch_num': 300, 'simple_train_batch_size': 128, 'simple_test_batch_size': 100,
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
    # train_data = train_data[:binary_sample_size]
    print('train data size', len(train_data))
    print('test data size', len(test_data))
    print('build model')
    model, loss_function, optimizer = build_model(config)
    print('test model')
    test_accuracy, pred_np = simple_test_batch(test_data, model, config)
    print('test accuracy', test_accuracy)
    print('train model')
    total_loss_list = simple_train_batch(train_data, test_data, model, loss_function, optimizer, config)
    print('total loss list', total_loss_list)
    print('test model')
    test_accuracy, pred_np = simple_test_batch(test_data, model, config)
    print('test accuracy', test_accuracy)
    # for index in range(len(test_data)):
    #     print(test_data[index][1].cpu().data.item())
    #     print(pred_np[index][0])


if __name__ == '__main__':
    run_nn_experiments()
