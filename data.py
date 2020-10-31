import numpy as np
import pickle
import torchvision.transforms as transforms
import torchvision
import torch


def process_one_hot(data):
    new_data = []
    for (inputs, targets) in data:
        # change targets to one-hot
        zero_vector = np.zeros(10)
        zero_vector[targets[0].item()] = 1.0
        targets = torch.Tensor(zero_vector.reshape(1, -1))
        new_data.append((inputs, targets))
    return new_data


def save_sampled_binary_data(config):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    if config['dataset'] == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=config['dir_path'] + '/data', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=config['dir_path'] + '/data', train=False, download=True,
                                               transform=transform_test)
    else:
        trainset = torchvision.datasets.MNIST(root=config['dir_path'] + '/data', train=True, download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.MNIST(root=config['dir_path'] + '/data', train=False, download=True,
                                             transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    print('original train data size', len(trainloader))
    label_num = [0] * 2
    train_data = []
    train_data_neg = []
    train_data_pos = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if str(targets[0].item()) not in config['binary_labels']:
            continue
        if str(targets[0].item()) == config['binary_labels'][0]:
            if label_num[0] < config['sample_size'] // 2:
                train_data_neg.append((inputs, torch.LongTensor([0])))
                label_num[0] += 1
        if str(targets[0].item()) == config['binary_labels'][1]:
            if label_num[1] < config['sample_size'] // 2:
                train_data_pos.append((inputs, torch.LongTensor([1])))
                label_num[1] += 1
    np.random.shuffle(train_data_neg)
    np.random.shuffle(train_data_pos)
    for x in range(len(train_data_neg)):
        train_data.append(train_data_neg[x])
        train_data.append(train_data_pos[x])
    test_data = []
    test_data_neg = []
    test_data_pos = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if str(targets[0].item()) not in config['binary_labels']:
            continue
        if str(targets[0].item()) == config['binary_labels'][0]:
            test_data_neg.append((inputs, torch.LongTensor([0])))
        if str(targets[0].item()) == config['binary_labels'][1]:
            test_data_pos.append((inputs, torch.LongTensor([1])))
    np.random.shuffle(test_data_neg)
    np.random.shuffle(test_data_pos)
    for x in range(np.minimum(len(test_data_neg), len(test_data_pos))):
        test_data.append(test_data_neg[x])
        test_data.append(test_data_pos[x])
    print('label num', label_num)
    print('train data size', len(train_data))
    print('test data size', len(test_data))
    if config['dataset'] == 'CIFAR10':
        pickle_out_train = open(config['dir_path'] + "/data/train_cifar10_" + "_".join(config['binary_labels']) + "_" +
                                str(config['sample_size']) + ".pickle", "wb")
        pickle_out_test = open(config['dir_path'] + "/data/test_cifar10_" + "_".join(config['binary_labels']) +
                               ".pickle", "wb")
    else:
        pickle_out_train = open(config['dir_path'] + "/data/train_mnist_" + "_".join(config['binary_labels']) + "_" +
                                str(config['sample_size']) + ".pickle", "wb")
        pickle_out_test = open(config['dir_path'] + "/data/test_mnist_" + "_".join(config['binary_labels']) +
                               ".pickle", "wb")
    pickle.dump(train_data, pickle_out_train)
    pickle_out_train.close()
    pickle.dump(test_data, pickle_out_test)
    pickle_out_test.close()


def load_sampled_binary_data_from_pickle(config):
    if config['dataset'] == 'CIFAR10':
        pickle_in_train = open(config['dir_path'] + "/data/train_cifar10_" + "_".join(config['binary_labels']) + "_" +
                               str(config['sample_size']) + ".pickle", "rb")
        pickle_in_test = open(config['dir_path'] + "/data/test_cifar10_" + "_".join(config['binary_labels']) +
                              ".pickle", "rb")
    else:
        pickle_in_train = open(config['dir_path'] + "/data/train_mnist_" + "_".join(config['binary_labels']) + "_" +
                               str(config['sample_size']) + ".pickle", "rb")
        pickle_in_test = open(config['dir_path'] + "/data/test_mnist_" + "_".join(config['binary_labels']) +
                              ".pickle", "rb")
    trainloader = pickle.load(pickle_in_train)
    pickle_in_train.close()
    testloader = pickle.load(pickle_in_test)
    pickle_in_test.close()
    return trainloader, testloader


def save_sampled_data(config, dir_path):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    if config['dataset'] == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.MNIST(root=dir_path + '/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root=dir_path + '/data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    print('original train data size', len(trainloader))
    label_num = [0] * 10
    train_data = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if label_num[targets[0].item()] < config['sample_size'] // 10:
            train_data.append((inputs, targets))
            label_num[targets[0].item()] += 1
    test_data = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        test_data.append((inputs, targets))
    if config['one-hot']:
        print('process data')
        train_data = process_one_hot(train_data)
        test_data = process_one_hot(test_data)
    print('label num', label_num)
    print('train data size', len(train_data))
    print('test data size', len(test_data))
    if config['dataset'] == 'CIFAR10':
        pickle_out_train = open(dir_path + "/data/train_cifar10_" + str(config['sample_size']) + "_.pickle", "wb")
        pickle_out_test = open(dir_path + "/data/test_cifar10.pickle", "wb")
    else:
        pickle_out_train = open(dir_path + "/data/train_mnist_" + str(config['sample_size']) + "_.pickle", "wb")
        pickle_out_test = open(dir_path + "/data/test_mnist.pickle", "wb")
    pickle.dump(train_data, pickle_out_train)
    pickle_out_train.close()
    pickle.dump(test_data, pickle_out_test)
    pickle_out_test.close()


def load_sampled_data_from_pickle(config, dir_path):
    if config['dataset'] == 'CIFAR10':
        pickle_in_train = open(dir_path + "/data/train_cifar10_" + str(config['sample_size']) + "_.pickle", "rb")
        pickle_in_test = open(dir_path + "/data/test_cifar10.pickle", "rb")
    else:
        pickle_in_train = open(dir_path + "/data/train_mnist_" + str(config['sample_size']) + "_.pickle", "rb")
        pickle_in_test = open(dir_path + "/data/test_mnist.pickle", "rb")
    trainloader = pickle.load(pickle_in_train)
    pickle_in_train.close()
    testloader = pickle.load(pickle_in_test)
    pickle_in_test.close()
    return trainloader, testloader

