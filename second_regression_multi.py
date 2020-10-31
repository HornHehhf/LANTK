import numpy as np
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor

from data import save_sampled_data, load_sampled_data_from_pickle
from utils import set_random_seed, add_one_extra_dim, get_minibatches_idx
from fjlt import fjlt
from models import LinearModel
from ntk_google_multi import process_data_numpy_multi, get_test_accuracy_multi
from ntk import get_kernel_matrix

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_ntl_kernel_V1_pre(X_train, Y_train, H_train_train, H_test_train):
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
    return train_label_kernels, test_label_kernels


def get_ntl_kernel_V2_pre(X_train, Y_train, H_train_train, H_test_train):
    # inputs: X_train (n, d), X_test (b, d), Y_train (n, 1)
    # outputs: kernel_values (b, n)
    n = X_train.shape[0]
    # H_train_train = get_kernel_matrix(X_train, X_train)
    # (n, n)
    # H_test_train = get_kernel_matrix(X_train, X_test)
    # (b, n)
    label_inner_products = (np.dot(Y_train, np.transpose(Y_train)) - 0.5) * 2
    # (n, n)
    H_same_mean = np.sum(H_train_train * (label_inner_products >= 0)) / np.sum((label_inner_products >= 0))
    H_diff_mean = np.sum(H_train_train * (label_inner_products <= 0)) / np.sum((label_inner_products <= 0))

    print('H same mean', H_same_mean)
    print('H diff mean', H_diff_mean)

    train_label_kernels = (np.abs(H_train_train - H_same_mean) <= np.abs(H_train_train - H_diff_mean)) * 2 - 1
    test_label_kernels = (np.abs(H_test_train - H_same_mean) <= np.abs(H_test_train - H_diff_mean)) * 2 - 1

    return train_label_kernels, test_label_kernels


def get_pair_predictions(train_train_features, test_train_features, train_train_labels, epsilon=0.01):
    # input: train_train_features (n^2, p), test_train_features (b*n, p), train_train_labels (n^2, 1)
    # output: test_train_labels_pred (b*n, 1)
    d = train_train_features.shape[0]
    print('d', d)
    m = (train_train_features.shape[1] + 1)
    print('m', m)
    k = int(np.log(m) / epsilon / epsilon)
    print('k', k)
    q = np.log(m) * np.log(m) / d
    print('q', q)
    print('kdq', k * d * q)
    train_train_extend = np.concatenate((train_train_features, train_train_labels), axis=1)
    # (d, m)
    reduced_train_train_extend = fjlt(train_train_extend, k, q)
    # (k, m)
    print('reduced_train_train_extend', reduced_train_train_extend.shape)
    reduced_train_train_features = reduced_train_train_extend[:, 0:-1]
    # (k, m-1)
    reduced_train_train_labels = reduced_train_train_extend[:, -1].reshape(-1, 1)
    # (k, 1)
    ATA = np.dot(np.transpose(reduced_train_train_features), reduced_train_train_features)
    # (m-1, m-1)
    ATb = np.dot(np.transpose(reduced_train_train_features), reduced_train_train_labels)
    # (m-1, 1)
    weights = np.linalg.solve(ATA, ATb)
    # (m-1, 1)
    print('weights', weights)
    test_train_features_pred = np.dot(test_train_features, weights)
    # (b*n, 1)
    print('test train features pred', test_train_features_pred[:100])
    train_train_features_pred = np.dot(train_train_features, weights)
    print('train train features pred', train_train_features_pred[:100])
    print('train train labels', train_train_labels[:100])
    return train_train_features_pred, test_train_features_pred


def SGD_linear(train_train_features, test_train_features, train_train_labels, sample_ratio=1.0):
    # input: train_train_features (n^2, p), test_train_features (b*n, p), train_train_labels (n^2, 1)
    # output: test_train_labels_pred (b*n, 1)
    train_index = list(range(train_train_features.shape[0]))
    np.random.shuffle(train_index)
    train_index = train_index[: int(train_train_features.shape[0] * sample_ratio)]
    train_train_features = train_train_features[train_index]
    train_train_labels = train_train_labels[train_index]
    print('train train features', train_train_features.shape)
    print('test train features', test_train_features.shape)

    clf = SGDRegressor(max_iter=5000)
    clf.fit(train_train_features, train_train_labels.reshape(-1))
    train_train_features_pred = clf.predict(train_train_features)
    test_train_features_pred = clf.predict(test_train_features)
    print(train_train_features_pred[:100])
    print('train accuracy', np.mean((train_train_features_pred >= 0) == (train_train_labels.reshape(-1) >= 0)))
    return train_train_features_pred, test_train_features_pred


def build_model(config):
    model = LinearModel(config['input_size'])
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    return model, loss_function, optimizer


def simple_train_batch(train_features, train_labels, model, loss_function, optimizer, config):
    total_loss_list = []
    for epoch in range(config['epoch_num']):
        model.train()
        print('current epoch: ', epoch)
        total_loss = 0
        minibatches_idx = get_minibatches_idx(len(train_features), minibatch_size=config['batch_size'],
                                              shuffle=True)
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(train_features[minibatch, :])
            targets = torch.Tensor(train_labels[minibatch])
            inputs, targets = Variable(inputs.cuda()), Variable(targets.float().cuda())
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
        total_loss_list.append(total_loss.cpu().data.item())
        print('train loss', total_loss)
    return total_loss_list


def simple_test_batch(test_features, model, config):
    model.eval()
    pred_np = []
    minibatches_idx = get_minibatches_idx(len(test_features), minibatch_size=config['batch_size'],
                                          shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(test_features[minibatch, :])
        inputs = Variable(inputs.cuda())
        outputs = model(inputs)
        pred_np.extend(list(outputs.cpu().data.numpy()))
    return np.array(pred_np)


def miniGD_linear(train_train_features, test_train_features, train_train_labels, sample_ratio=1.0):
    # input: train_train_features (n^2, p), test_train_features (b*n, p), train_train_labels (n^2, 1)
    # output: test_train_labels_pred (b*n, 1)
    config = {'lr': 3e-2, 'epoch_num': 300, 'input_size': train_train_features.shape[1], 'batch_size': 20000000}
    train_index = list(range(train_train_features.shape[0]))
    np.random.shuffle(train_index)
    train_index = train_index[: int(train_train_features.shape[0] * sample_ratio)]
    train_train_features = train_train_features[train_index]
    train_train_labels = train_train_labels[train_index].reshape(-1)
    print('train train features', train_train_features.shape)
    print('test train features', test_train_features.shape)
    print('build model')
    model, loss_function, optimizer = build_model(config)
    print('train model')
    total_loss_list = simple_train_batch(train_train_features, train_train_labels, model, loss_function, optimizer,
                                         config)
    print('test model')
    train_train_pred = simple_test_batch(train_train_features, model, config)
    test_train_pred = simple_test_batch(test_train_features, model, config)
    print('train train pred', train_train_pred[:100])
    print('train train labels', train_train_labels[:100])
    print('train accuracy', np.mean((train_train_pred.reshape(-1) >= 0) == (train_train_labels >= 0)))
    return train_train_pred, test_train_pred


def get_pca_features(X_train, X_test):
    # input: X_train (n, d), X_test (b, d)
    # output: pca_features_train (n^2, 30), pca_features_test (n^2, 30)
    d = 5
    pca = PCA(n_components=d, svd_solver='full')
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print('X_train_pca', X_train_pca.shape)
    print('X_test_pca', X_test_pca.shape)
    n = X_train.shape[0]
    b = X_test.shape[0]
    # train_left = X_train_pca.reshape(n, 1, 10) + np.zeros((n, n, 10))
    # train_right = X_train_pca.reshape(1, n, 10) + np.zeros((n, n, 10))
    # test_left = X_test_pca.reshape(b, 1, 10) + np.zeros((b, n, 10))
    # test_right = X_train_pca.reshape(1, n, 10) + np.zeros((b, n, 10))
    train_joint = np.zeros((n, n, d))
    test_joint = np.zeros((b, n, d))
    train_dis = np.zeros((n, n, d))
    test_dis = np.zeros((b, n, d))
    # train_square = np.zeros((n, n, d))
    # test_square = np.zeros((b, n, d))
    for i in range(n):
        for j in range(n):
            train_joint[i, j] = X_train_pca[i] * X_train_pca[j]
            train_dis[i, j] = np.abs(X_train_pca[i] - X_train_pca[j])
            # train_square[i, j] = np.square(X_train_pca[i] - X_train_pca[j])
    for i in range(b):
        for j in range(n):
            test_joint[i, j] = X_test_pca[i] * X_train_pca[j]
            test_dis[i, j] = np.abs(X_test_pca[i] - X_train_pca[j])
            # test_square[i, j] = np.square(X_test_pca[i] - X_train_pca[j])
    # (b * n, d)
    # F_train_train = np.concatenate((train_left, train_right), axis=-1)
    # F_train_train = np.concatenate((F_train_train, train_joint), axis=-1)
    # F_test_train = np.concatenate((test_left, test_right), axis=-1)
    # F_test_train = np.concatenate((F_test_train, test_joint), axis=-1)
    F_train_train = np.concatenate((train_joint, train_dis), axis=-1)
    F_test_train = np.concatenate((test_joint, test_dis), axis=-1)
    F_train_train = F_train_train.reshape(-1, 2 * d)
    F_test_train = F_test_train.reshape(-1, 2 * d)
    print('F_train_train', F_train_train.shape)
    print('F_test_train', F_test_train.shape)
    return F_train_train, F_test_train


def get_pca_global(X_train, X_test):
    d = 5
    pca = PCA(n_components=d, svd_solver='full')
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print('X_train_pca', X_train_pca.shape)
    print('X_test_pca', X_test_pca.shape)
    H_train_train = get_kernel_matrix(X_train_pca, X_train_pca)
    H_test_train = get_kernel_matrix(X_train_pca, X_test_pca)
    X_train_norm = np.linalg.norm(X_train_pca, axis=1).reshape(-1, 1)
    X_test_norm = np.linalg.norm(X_test_pca, axis=1).reshape(-1, 1)
    norm_sum_train = np.square(X_train_norm).reshape(-1, 1) + np.square(X_train_norm).reshape(1, -1)
    norm_sum_test = np.square(X_test_norm).reshape(-1, 1) + np.square(X_train_norm).reshape(1, -1)
    norm_products_train = np.dot(X_train_norm, np.transpose(X_train_norm))
    norm_products_test = np.dot(X_test_norm, np.transpose(X_train_norm))
    inner_products_train = np.dot(X_train_pca, np.transpose(X_train_pca))
    inner_products_test = np.dot(X_test_pca, np.transpose(X_train_pca))
    cos_values_train = np.clip(inner_products_train / norm_products_train, -1.0, 1.0)
    cos_values_test = np.clip(inner_products_test / norm_products_test, -1.0, 1.0)
    euc_distance_train = norm_sum_train - 2 * inner_products_train
    euc_distance_test = norm_sum_test - 2 * inner_products_test
    sin_values_train = np.sqrt(1 - np.square(cos_values_train))
    sin_values_test = np.sqrt(1 - np.square(cos_values_test))
    theta_values_train = np.arccos(cos_values_train)
    theta_values_test = np.arccos(cos_values_test)
    sigma_square = np.mean(euc_distance_train)
    rbf_train = np.exp(- euc_distance_train / sigma_square)
    rbf_test = np.exp(- euc_distance_test / sigma_square)
    X_train_diff = X_train_pca - np.mean(X_train_pca, axis=1).reshape(-1, 1)
    X_test_diff = X_test_pca - np.mean(X_test_pca, axis=1).reshape(-1, 1)
    inner_products_train_diff = np.dot(X_train_diff, np.transpose(X_train_diff))
    inner_products_test_diff = np.dot(X_test_diff, np.transpose(X_train_diff))
    X_train_diff_norm = np.linalg.norm(X_train_diff, axis=1).reshape(-1, 1)
    X_test_diff_norm = np.linalg.norm(X_test_diff, axis=1).reshape(-1, 1)
    norm_products_train_diff = np.dot(X_train_diff_norm, np.transpose(X_train_diff_norm))
    norm_products_test_diff = np.dot(X_test_diff_norm, np.transpose(X_train_diff_norm))
    correlation_train = inner_products_train_diff / norm_products_train_diff
    correlation_test = inner_products_test_diff / norm_products_test_diff
    n = X_train_pca.shape[0]
    b = X_test_pca.shape[0]
    l1_norm_train = np.zeros((n, n))
    l1_norm_test = np.zeros((b, n))
    for i in range(n):
        for j in range(n):
            l1_norm_train[i, j] = np.linalg.norm(X_train_pca[i] - X_train_pca[j], ord=1)
    for i in range(b):
        for j in range(n):
            l1_norm_test[i, j] = np.linalg.norm(X_test_pca[i] - X_train_pca[j], ord=1)
    train_train_features = np.concatenate((H_train_train.reshape(-1, 1), inner_products_train.reshape(-1, 1)),
                                          axis=1)
    train_train_features = np.concatenate((train_train_features, cos_values_train.reshape(-1, 1)), axis=1)
    train_train_features = np.concatenate((train_train_features, euc_distance_train.reshape(-1, 1)), axis=1)
    train_train_features = np.concatenate((train_train_features, norm_products_train.reshape(-1, 1)), axis=1)
    train_train_features = np.concatenate((train_train_features, sin_values_train.reshape(-1, 1)), axis=1)
    train_train_features = np.concatenate((train_train_features, theta_values_train.reshape(-1, 1)), axis=1)
    train_train_features = np.concatenate((train_train_features, rbf_train.reshape(-1, 1)), axis=1)
    train_train_features = np.concatenate((train_train_features, correlation_train.reshape(-1, 1)), axis=1)
    train_train_features = np.concatenate((train_train_features, l1_norm_train.reshape(-1, 1)), axis=1)

    test_train_features = np.concatenate((H_test_train.reshape(-1, 1), inner_products_test.reshape(-1, 1)), axis=1)
    test_train_features = np.concatenate((test_train_features, cos_values_test.reshape(-1, 1)), axis=1)
    test_train_features = np.concatenate((test_train_features, euc_distance_test.reshape(-1, 1)), axis=1)
    test_train_features = np.concatenate((test_train_features, norm_products_test.reshape(-1, 1)), axis=1)
    test_train_features = np.concatenate((test_train_features, sin_values_test.reshape(-1, 1)), axis=1)
    test_train_features = np.concatenate((test_train_features, theta_values_test.reshape(-1, 1)), axis=1)
    test_train_features = np.concatenate((test_train_features, rbf_test.reshape(-1, 1)), axis=1)
    test_train_features = np.concatenate((test_train_features, correlation_test.reshape(-1, 1)), axis=1)
    test_train_features = np.concatenate((test_train_features, l1_norm_test.reshape(-1, 1)), axis=1)
    return train_train_features, test_train_features


def run_ntl_experiments():
    sample_size = int(sys.argv[1].split('=')[1])
    ntl_version = sys.argv[2].split('=')[1]
    print('sample size', sample_size)
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    config = {'sample_size': sample_size, 'seed': 66, 'dataset': 'CIFAR10', 'input_size': 3 * 32 * 32 + 1,
              'hidden_size': 4096, 'epoch_num': 500, 'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'lr': 0.5,
              'dir_path': '/path/to/working/dir', 'label_ratio': 0.1, 'ntl_version': ntl_version,
              'train': True, 'one-hot': True}
    train_kernel_path = config['dir_path'] + '/data/kernel_matrix/train_kernel_CNN_' + str(config['sample_size']) + \
                        '.pickle'
    test_kernel_path = config['dir_path'] + '/data/kernel_matrix/test_kernel_CNN_' + str(config['sample_size']) + \
                       '.pickle'
    train_label_kernel_path = config['dir_path'] + '/data/kernel_matrix/train_label_kernel_CNN_' + \
                              str(config['sample_size']) + ntl_version + '_global.pickle'
    test_label_kernel_path = config['dir_path'] + '/data/kernel_matrix/test_label_kernel_CNN_' + \
                             str(config['sample_size']) + ntl_version + '_global.pickle'
    pickle_out_train_label = open(train_label_kernel_path, 'wb')
    pickle_out_test_label = open(test_label_kernel_path, 'wb')
    pickle_in_train_label = open(train_label_kernel_path, 'rb')
    pickle_in_test_label = open(test_label_kernel_path, 'rb')
    pickle_in_train = open(train_kernel_path, 'rb')
    pickle_in_test = open(test_kernel_path, 'rb')
    H_train_train = pickle.load(pickle_in_train)
    pickle_in_train.close()
    H_test_train = pickle.load(pickle_in_test)
    pickle_in_test.close()
    # H_train_train = H_train_train[:1000, :1000]
    # H_test_train = H_test_train[:1000, :1000]

    print('set random seed', config['seed'])
    set_random_seed(config['seed'])
    print('sample data')
    save_sampled_data(config, config['dir_path'])
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
    X_train_norm = np.linalg.norm(total_inputs_train, axis=1).reshape(-1, 1)
    X_test_norm = np.linalg.norm(total_inputs_test, axis=1).reshape(-1, 1)
    norm_sum_train = np.square(X_train_norm).reshape(-1, 1) + np.square(X_train_norm).reshape(1, -1)
    norm_sum_test = np.square(X_test_norm).reshape(-1, 1) + np.square(X_train_norm).reshape(1, -1)
    norm_products_train = np.dot(X_train_norm, np.transpose(X_train_norm))
    norm_products_test = np.dot(X_test_norm, np.transpose(X_train_norm))
    inner_products_train = np.dot(total_inputs_train, np.transpose(total_inputs_train))
    inner_products_test = np.dot(total_inputs_test, np.transpose(total_inputs_train))
    cos_values_train = np.clip(inner_products_train / norm_products_train, -1.0, 1.0)
    cos_values_test = np.clip(inner_products_test / norm_products_test, -1.0, 1.0)
    # delta_train = np.arccos(cos_values_train)
    # delta_test = np.arccos(cos_values_test)
    euc_distance_train = norm_sum_train - 2 * inner_products_train
    euc_distance_test = norm_sum_test - 2 * inner_products_test
    # print('euc distance train', np.mean((euc_distance_train >= 0)))
    # print('euc distance test', np.mean((euc_distance_test >= 0)))
    # H_train_train = cos_values_train
    # H_test_train = cos_values_test
    # get train kernel values
    # config['train'] = True
    print(config['ntl_version'])
    if config['ntl_version'] == 'V1':
        train_label_kernels, test_label_kernels = \
            get_ntl_kernel_V1_pre(total_inputs_train, total_outputs_train, H_train_train, H_test_train)
    elif config['ntl_version'] == 'V2':
        train_label_kernels, test_label_kernels = \
            get_ntl_kernel_V2_pre(total_inputs_train, total_outputs_train, H_train_train, H_test_train)
    elif config['ntl_version'] == 'V3':
        train_label_kernels = np.dot(total_outputs_train, np.transpose(total_outputs_train))
        transform_matrix = np.linalg.solve(H_train_train, total_outputs_train)
        test_output_pred = np.dot(H_test_train, transform_matrix)
        test_label_kernels = np.dot(test_output_pred, np.transpose(total_outputs_train))
        print('test output pred', test_output_pred)
        test_output_pred = ((test_output_pred >= 0) * 2 - 1)
        print('test output pred', test_output_pred)
        print('test accuracy', np.mean(test_output_pred == total_outputs_test))
    elif config['ntl_version'] == 'V4':
        F_train_train, F_test_train = get_pca_features(total_inputs_train, total_inputs_test)
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
        test_label_kernels = test_label_kernels.reshape(-1, H_test_train.shape[1])
    elif config['ntl_version'] == 'V5':
        F_train_train, F_test_train = get_pca_features(total_inputs_train, total_inputs_test)
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
        train_label_kernels, test_label_kernels = SGD_linear(train_train_features, test_train_features,
                                                             train_label_kernels.reshape(-1, 1))
        train_label_kernels = train_label_kernels.reshape(-1, H_train_train.shape[1])
        test_label_kernels = test_label_kernels.reshape(-1, H_test_train.shape[1])
    elif config['ntl_version'] == 'V6':
        F_train_train, F_test_train = get_pca_features(total_inputs_train, total_inputs_test)
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
        train_label_kernels, test_label_kernels = miniGD_linear(train_train_features, test_train_features,
                                                                train_label_kernels.reshape(-1, 1))
        train_label_kernels = train_label_kernels.reshape(-1, H_train_train.shape[1])
        test_label_kernels = test_label_kernels.reshape(-1, H_test_train.shape[1])
    elif config['ntl_version'] == 'V7':
        train_label_products = (np.dot(total_outputs_train, np.transpose(total_outputs_train)) - 0.5) * 2
        sin_values_train = np.sqrt(1 - np.square(cos_values_train))
        sin_values_test = np.sqrt(1 - np.square(cos_values_test))
        theta_values_train = np.arccos(cos_values_train)
        theta_values_test = np.arccos(cos_values_test)
        sigma_square = np.mean(euc_distance_train)
        rbf_train = np.exp(- euc_distance_train / sigma_square)
        rbf_test = np.exp(- euc_distance_test / sigma_square)
        X_train_diff = total_inputs_train - np.mean(total_inputs_train, axis=1).reshape(-1, 1)
        X_test_diff = total_inputs_test - np.mean(total_inputs_test, axis=1).reshape(-1, 1)
        inner_products_train_diff = np.dot(X_train_diff, np.transpose(X_train_diff))
        inner_products_test_diff = np.dot(X_test_diff, np.transpose(X_train_diff))
        X_train_diff_norm = np.linalg.norm(X_train_diff, axis=1).reshape(-1, 1)
        X_test_diff_norm = np.linalg.norm(X_test_diff, axis=1).reshape(-1, 1)
        norm_products_train_diff = np.dot(X_train_diff_norm, np.transpose(X_train_diff_norm))
        norm_products_test_diff = np.dot(X_test_diff_norm, np.transpose(X_train_diff_norm))
        correlation_train = inner_products_train_diff / norm_products_train_diff
        correlation_test = inner_products_test_diff / norm_products_test_diff
        n = total_inputs_train.shape[0]
        b = total_inputs_test.shape[0]
        l1_norm_train = np.zeros((n, n))
        l1_norm_test = np.zeros((b, n))
        for i in range(n):
            for j in range(n):
                l1_norm_train[i, j] = np.linalg.norm(total_inputs_train[i] - total_inputs_train[j], ord=1)
        for i in range(b):
            for j in range(n):
                l1_norm_test[i, j] = np.linalg.norm(total_inputs_test[i] - total_inputs_train[j], ord=1)
        train_train_features = np.concatenate((H_train_train.reshape(-1, 1), inner_products_train.reshape(-1, 1)),
                                              axis=1)
        train_train_features = np.concatenate((train_train_features, cos_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, euc_distance_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, norm_products_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, sin_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, theta_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, rbf_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, correlation_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, l1_norm_train.reshape(-1, 1)), axis=1)

        test_train_features = np.concatenate((H_test_train.reshape(-1, 1), inner_products_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, cos_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, euc_distance_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, norm_products_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, sin_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, theta_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, rbf_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, correlation_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, l1_norm_test.reshape(-1, 1)), axis=1)

        train_label_kernels, test_label_kernels = get_pair_predictions(train_train_features, test_train_features,
                                                                       train_label_products.reshape(-1, 1))
        train_label_kernels = train_label_kernels.reshape(-1, H_train_train.shape[1])
        test_label_kernels = test_label_kernels.reshape(-1, H_test_train.shape[1])
    elif config['ntl_version'] == 'V8':
        train_label_products = (np.dot(total_outputs_train, np.transpose(total_outputs_train)) - 0.5) * 2
        sin_values_train = np.sqrt(1 - np.square(cos_values_train))
        sin_values_test = np.sqrt(1 - np.square(cos_values_test))
        theta_values_train = np.arccos(cos_values_train)
        theta_values_test = np.arccos(cos_values_test)
        sigma_square = np.mean(euc_distance_train)
        rbf_train = np.exp(- euc_distance_train / sigma_square)
        rbf_test = np.exp(- euc_distance_test / sigma_square)
        X_train_diff = total_inputs_train - np.mean(total_inputs_train, axis=1).reshape(-1, 1)
        X_test_diff = total_inputs_test - np.mean(total_inputs_test, axis=1).reshape(-1, 1)
        inner_products_train_diff = np.dot(X_train_diff, np.transpose(X_train_diff))
        inner_products_test_diff = np.dot(X_test_diff, np.transpose(X_train_diff))
        X_train_diff_norm = np.linalg.norm(X_train_diff, axis=1).reshape(-1, 1)
        X_test_diff_norm = np.linalg.norm(X_test_diff, axis=1).reshape(-1, 1)
        norm_products_train_diff = np.dot(X_train_diff_norm, np.transpose(X_train_diff_norm))
        norm_products_test_diff = np.dot(X_test_diff_norm, np.transpose(X_train_diff_norm))
        correlation_train = inner_products_train_diff / norm_products_train_diff
        correlation_test = inner_products_test_diff / norm_products_test_diff
        n = total_inputs_train.shape[0]
        b = total_inputs_test.shape[0]
        l1_norm_train = np.zeros((n, n))
        l1_norm_test = np.zeros((b, n))
        for i in range(n):
            for j in range(n):
                l1_norm_train[i, j] = np.linalg.norm(total_inputs_train[i] - total_inputs_train[j], ord=1)
        for i in range(b):
            for j in range(n):
                l1_norm_test[i, j] = np.linalg.norm(total_inputs_test[i] - total_inputs_train[j], ord=1)
        F_train_train, F_test_train = get_pca_features(total_inputs_train, total_inputs_test)
        train_train_features = np.concatenate((H_train_train.reshape(-1, 1), inner_products_train.reshape(-1, 1)),
                                              axis=1)
        train_train_features = np.concatenate((train_train_features, cos_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, euc_distance_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, norm_products_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, sin_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, theta_values_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, rbf_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, correlation_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, l1_norm_train.reshape(-1, 1)), axis=1)
        train_train_features = np.concatenate((train_train_features, F_train_train), axis=1)

        test_train_features = np.concatenate((H_test_train.reshape(-1, 1), inner_products_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, cos_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, euc_distance_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, norm_products_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, sin_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, theta_values_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, rbf_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, correlation_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, l1_norm_test.reshape(-1, 1)), axis=1)
        test_train_features = np.concatenate((test_train_features, F_test_train), axis=1)

        train_label_kernels, test_label_kernels = get_pair_predictions(train_train_features, test_train_features,
                                                                       train_label_products.reshape(-1, 1))
        train_label_kernels = train_label_kernels.reshape(-1, H_train_train.shape[1])
        test_label_kernels = test_label_kernels.reshape(-1, H_test_train.shape[1])

    pickle.dump(train_label_kernels, pickle_out_train_label)
    pickle_out_train_label.close()
    train_label_kernels = pickle.load(pickle_in_train_label)
    pickle_in_train_label.close()
    pickle.dump(test_label_kernels, pickle_out_test_label)
    pickle_out_test_label.close()
    test_label_kernels = pickle.load(pickle_in_test_label)
    pickle_in_test_label.close()
    train_label_products = ((np.dot(total_outputs_train, np.transpose(total_outputs_train)) - 0.5) * 2 >= 0)
    test_label_products = ((np.dot(total_outputs_test, np.transpose(total_outputs_train)) - 0.5) * 2 >= 0)

    train_label_kernels = (train_label_kernels >= 0)
    test_label_kernels = (test_label_kernels >= 0)
    # print('train label kernels', train_label_kernels)
    # print('test label kernels', test_label_kernels)
    # print('train label products', train_label_products)
    # print('test label products', test_label_products)
    print('train label kernel accuracy', np.mean(train_label_kernels == train_label_products))
    print('test label kernel accuracy', np.mean(test_label_kernels == test_label_products))


if __name__ == '__main__':
    run_ntl_experiments()

