import numpy as np
import random
import pickle
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from os import listdir
import json

from utils import set_random_seed, shuffle_in_union


def process_one_hot(data):
    new_data = []
    for (inputs, targets) in data:
        inputs = inputs.cpu().numpy().reshape(-1)
        inputs = inputs.tolist()
        # add one dimension
        inputs.append(1)
        inputs = torch.Tensor(np.array(inputs).reshape(1, -1))
        # change targets to one-hot
        zero_vector = np.zeros(10)
        zero_vector[targets[0].item()] = 1.0
        targets = torch.Tensor(zero_vector.reshape(1, -1))
        new_data.append((inputs, targets))
    return new_data


def double_helix(size, min_z=-0.5 *np.pi, max_z=0.5 * np.pi, k=1.5):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 2):
        theta_pos = random.uniform(min_z, max_z)
        theta_neg = random.uniform(min_z, max_z)
        positions.append((np.cos(k * theta_pos), np.sin(k * theta_pos), theta_pos))
        original_positions.append(theta_pos)
        labels.append(1)
        positions.append((-np.cos(k * theta_neg), -np.sin(k * theta_neg), theta_neg))
        original_positions.append(theta_neg)
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def two_cosine(size, min_x=-0.6 * np.pi, max_x=0.6 * np.pi, k=1, delta=0.2):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 2):
        theta_pos = random.uniform(min_x, max_x)
        theta_neg = random.uniform(min_x, max_x)
        # positions.append((1, 1))
        positions.append((theta_pos, np.cos(k * theta_pos) + delta))
        original_positions.append(theta_pos)
        labels.append(1)
        # positions.append((0, 0))
        positions.append((theta_neg, np.cos(k * theta_neg) - delta))
        original_positions.append(theta_neg)
        labels.append(0)
    # add imbalance
    '''for index in range(size // 2):
        theta_pos = random.uniform(min_x, max_x)
        positions.append((theta_pos, np.cos(k * theta_pos) + delta))
        original_positions.append(theta_pos)
        labels.append(1)'''
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def two_fold_surface(size, min_x=-10, max_x=10, min_z=-1, max_z=1, k=1.2, bias=12, delta=1):
# def two_fold_surface(size, min_x=-10, max_x=10, min_z=-1, max_z=1, k=1.0, bias=10, delta=1):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 4):
        x_pos = random.uniform(min_x - delta/k, max_x + delta/k)
        x_neg = random.uniform(min_x + delta/k, max_x - delta/k)
        z_pos = random.uniform(min_z, max_z)
        z_neg = random.uniform(min_z, max_z)
        positions.append((x_pos, -k * np.abs(x_pos) + bias + delta, z_pos))
        original_positions.append((x_pos, -k, z_pos, max_x + delta))
        labels.append(1)
        positions.append((x_neg, -k * np.abs(x_neg) + bias - delta, z_neg))
        original_positions.append((x_neg, -k, z_neg, max_x - delta))
        labels.append(0)
    for index in range(size // 4):
        x_pos = random.uniform(min_x - delta / k, max_x + delta / k)
        x_neg = random.uniform(min_x + delta / k, max_x - delta / k)
        z_pos = random.uniform(min_z, max_z)
        z_neg = random.uniform(min_z, max_z)
        positions.append((x_pos, k * np.abs(x_pos) - bias - delta, z_pos))
        original_positions.append((x_pos, +k, z_pos, max_x + delta))
        labels.append(1)
        positions.append((x_neg, k * np.abs(x_neg) - bias + delta, z_neg))
        original_positions.append((x_neg, +k, z_neg, max_x - delta))
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def donut(size, min_theta=-np.pi, max_theta=np.pi, r=8, k=4):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 2):
        theta_pos = random.uniform(min_theta, max_theta)
        theta_neg = random.uniform(min_theta, max_theta)
        positions.append((r * np.cos(theta_pos) + np.sin(k * theta_pos) * np.cos(theta_pos),
                          r * np.sin(theta_pos) + np.sin(k * theta_pos) * np.sin(theta_pos),
                          np.cos(k * theta_pos)))
        original_positions.append(theta_pos)
        # original_positions.append((theta_pos, 1))
        labels.append(1)
        positions.append((r * np.cos(theta_neg) - np.sin(k * theta_neg) * np.cos(theta_neg),
                          r * np.sin(theta_neg) - np.sin(k * theta_neg) * np.sin(theta_neg),
                          - np.cos(k * theta_neg)))
        original_positions.append(theta_neg)
        # original_positions.append((theta_neg, -1))
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def two_sphere(size, min_theta=0, max_theta=1.0*np.pi, r_1=10, r_2=20):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 2):
        theta_pos = random.uniform(min_theta, max_theta)
        theta_neg = random.uniform(min_theta, max_theta)
        phi_pos = random.uniform(0, 2 * np.pi)
        phi_neg = random.uniform(0, 2 * np.pi)
        positions.append((r_1 * np.sin(theta_pos) * np.cos(phi_pos),
                          r_1 * np.sin(theta_pos) * np.sin(phi_pos),
                          r_1 * np.cos(theta_pos)))
        original_positions.append((np.sin(theta_pos) * np.cos(phi_pos), np.sin(theta_pos) * np.sin(phi_pos),
                                   np.cos(theta_pos)))
        labels.append(1)
        positions.append((r_2 * np.sin(theta_neg) * np.cos(phi_neg),
                          r_2 * np.sin(theta_neg) * np.sin(phi_neg),
                          r_2 * np.cos(theta_neg)))
        original_positions.append((np.sin(theta_neg) * np.cos(phi_neg), np.sin(theta_neg) * np.sin(phi_neg),
                                   np.cos(theta_neg)))
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def two_cycles(size, r1=10, r2=10, r3=50):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 4):
        pos_theta_one = random.uniform(-np.pi, np.pi)
        pos_theta_two = random.uniform(-np.pi, np.pi)
        neg_theta_one = random.uniform(-np.pi, np.pi)
        neg_theta_two = random.uniform(-np.pi, np.pi)

        positions.append((r1 * np.cos(pos_theta_one), r1 * np.sin(pos_theta_one)))
        # original_positions.append((r1 * np.cos(pos_theta_one), r1 * np.sin(pos_theta_one)))
        original_positions.append(pos_theta_one)
        labels.append(1)
        positions.append((r3 * np.cos(neg_theta_one), r3 * np.sin(neg_theta_one)))
        # original_positions.append((r3 * np.cos(neg_theta_one), r3 * np.sin(neg_theta_one)))
        original_positions.append(neg_theta_one)
        labels.append(0)

        positions.append((r2 * np.cos(pos_theta_two), r2 * np.sin(pos_theta_two)))
        # original_positions.append((r2 * np.cos(pos_theta_two), r2 * np.sin(pos_theta_two)))
        original_positions.append(pos_theta_two)
        labels.append(1)
        positions.append((r3 * np.cos(neg_theta_two), r3 * np.sin(neg_theta_two)))
        # original_positions.append((r3 * np.cos(neg_theta_two), r3 * np.sin(neg_theta_two)))
        original_positions.append(neg_theta_two)
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def save_data_binary(config, dir_path):
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['test_batch_size'], shuffle=False)
    print('original train data size', len(trainloader))
    train_data_pos_one = []
    train_data_pos_two = []
    train_data_neg = []
    label_num = [0] * 3
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(inputs)
        if str(targets[0].item()) not in config['labels']['all']:
            continue
        if str(targets[0].item()) in config['labels']['neg']:
            if label_num[0] < config['sample_size']:
                label_num[0] += 1
                train_data_neg.append((inputs, torch.LongTensor([0])))
        else:
            if str(targets[0].item()) == config['labels']['pos'][0]:
                if label_num[1] < config['sample_size'] // 2:
                    train_data_pos_one.append((inputs, torch.LongTensor([1])))
                    # 1
                    label_num[1] += 1
            else:
                if label_num[2] < config['sample_size'] // 2:
                    train_data_pos_two.append((inputs, torch.LongTensor([1])))
                    # 1
                    label_num[2] += 1
    trainloader = train_data_pos_one, train_data_pos_two, train_data_neg
    test_data_pos_one = []
    test_data_pos_two = []
    test_data_neg = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if str(targets[0].item()) not in config['labels']['all']:
            continue
        if str(targets[0].item()) in config['labels']['neg']:
            test_data_neg.append((inputs, torch.LongTensor([0])))
        else:
            if str(targets[0].item()) == config['labels']['pos'][0]:
                test_data_pos_one.append((inputs, torch.LongTensor([1])))
            else:
                test_data_pos_two.append((inputs, torch.LongTensor([1])))
    testloader = test_data_pos_one, test_data_pos_two, test_data_neg
    print('train data pos one size', len(train_data_pos_one))
    print('train data pos two size', len(train_data_pos_two))
    print('train data neg', len(train_data_neg))
    print('test data pos one size', len(test_data_pos_one))
    print('test data pos two size', len(test_data_pos_two))
    print('test data neg', len(test_data_neg))
    print(label_num)
    if config['dataset'] == 'CIFAR10':
        pickle_out_train = open(dir_path + "/data/train_cifar10_binary_" + "_".join(config['labels']['all']) + ".pickle",
                                "wb")
        pickle_out_test = open(dir_path + "/data/test_cifar10_binary_" + "_".join(config['labels']['all']) + ".pickle",
                               "wb")
    else:
        pickle_out_train = open(dir_path + "/data/train_mnist_binary_" + "_".join(config['labels']['all']) + ".pickle",
                                "wb")
        pickle_out_test = open(dir_path + "/data/test_mnist_binary_" + "_".join(config['labels']['all']) + ".pickle",
                               "wb")
    pickle.dump(trainloader, pickle_out_train)
    pickle_out_train.close()
    pickle.dump(testloader, pickle_out_test)
    pickle_out_test.close()


def load_data_binary_from_pickle(config, dir_path):
    if config['dataset'] == 'CIFAR10':
        pickle_in_train = open(dir_path + "/data/train_cifar10_binary_" + "_".join(config['labels']['all']) + ".pickle",
                               "rb")
        pickle_in_test = open(dir_path + "/data/test_cifar10_binary_" + "_".join(config['labels']['all']) + ".pickle",
                              "rb")
    else:
        pickle_in_train = open(dir_path + "/data/train_mnist_binary_" + "_".join(config['labels']['all']) + ".pickle",
                               "rb")
        pickle_in_test = open(dir_path + "/data/test_mnist_binary_" + "_".join(config['labels']['all']) + ".pickle",
                              "rb")
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


def load_images_dict(root_dir):
    images = {}
    img_size = (32, 32)
    img_files = np.sort(listdir(root_dir))
    print(img_files)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    to_tensor = transforms.ToTensor()
    for img_file in img_files:
        img_path = root_dir + '/' + img_file
        img = cv2.imread(img_path)
        '''# Resize
        height, width, _ = img.shape
        new_height = height * 256 // min(img.shape[:2])
        new_width = width * 256 // min(img.shape[:2])
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # Crop
        height, width, _ = img.shape
        startx = width // 2 - (224 // 2)
        starty = height // 2 - (224 // 2)
        img = img[starty:starty + 224, startx:startx + 224]
        assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)'''
        img = cv2.resize(img, img_size)
        t_img = Variable(normalize(to_tensor(img)).unsqueeze(0)).cuda().float()
        images[img_file] = t_img
    return images


def save_data_mscoco(config, dir_path):
    images = load_images_dict(dir_path + '/data/MSCOCO/chair-bench-cat-dog-50/images')
    with open(dir_path + '/data/MSCOCO/chair-bench-cat-dog-50/cat-dog.json') as json_file:
        label_one = json.load(json_file)
    with open(dir_path + '/data/MSCOCO/chair-bench-cat-dog-50/chair-bench.json') as json_file:
        label_two = json.load(json_file)
    data_one = []
    data_two = []
    labels = []
    img_files = images.keys()
    for img_file in img_files:
        if label_one[img_file] == 'cat':
            data_one.append((images[img_file], torch.LongTensor([0])))
            cur_label_one = 0
        else:
            data_one.append((images[img_file], torch.LongTensor([1])))
            cur_label_one = 1
        if label_two[img_file] == 'chair':
            data_two.append((images[img_file], torch.LongTensor([0])))
            cur_label_two = 0
        else:
            data_two.append((images[img_file], torch.LongTensor([1])))
            cur_label_two = 1
        labels.append([cur_label_one, cur_label_two])
    pickle_out_one = open(dir_path + "/data/MSCOCO/mscoco_chair_bench_cat_dog_one.pickle", "wb")
    pickle.dump(data_one, pickle_out_one)
    pickle_out_one.close()
    pickle_out_two = open(dir_path + "/data/MSCOCO/mscoco_chair_bench_cat_dog_two.pickle", "wb")
    pickle.dump(data_two, pickle_out_two)
    pickle_out_two.close()
    pickle_out_labels = open(dir_path + "/data/MSCOCO/mscoco_chair_bench_cat_dog_labels.pickle", "wb")
    pickle.dump(labels, pickle_out_labels)
    pickle_out_labels.close()


def load_data_mscoco_from_pickle(config, dir_path):
    pickle_in_one = open(dir_path + "/data/MSCOCO/mscoco_chair_bench_cat_dog_one.pickle", "rb")
    pickle_in_two = open(dir_path + "/data/MSCOCO/mscoco_chair_bench_cat_dog_two.pickle", "rb")
    pickle_in_labels = open(dir_path + "/data/MSCOCO/mscoco_chair_bench_cat_dog_labels.pickle", "rb")
    data_one = pickle.load(pickle_in_one)
    pickle_in_one.close()
    data_two = pickle.load(pickle_in_two)
    pickle_in_two.close()
    labels = pickle.load(pickle_in_labels)
    pickle_in_labels.close()
    return data_one, data_two, labels

