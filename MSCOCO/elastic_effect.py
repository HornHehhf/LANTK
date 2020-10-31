import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy
import torchvision.models as models
import pickle
from scipy import stats

from utils import get_minibatches_idx
from models import MLPNet, MLPProb, MLPLinear, MLPLinearProb, CNN_MNIST, ResNet, MLPNet3Layer, MLPLinear3Layer, \
    MLPLinear1Layer, MLPNetSigmoid

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_model(config):
    if config['model'] == 'MLPProb':
        model = MLPProb(config['input_size'])
    elif config['model'] == 'MLPLinear':
        model = MLPLinear(config['input_size'])
    elif config['model'] == 'MLPLinearProb':
        model = MLPLinearProb(config['input_size'])
    elif config['model'] == 'CNN_MNIST':
        model = CNN_MNIST()
    elif config['model'] == 'ResNet':
        model = ResNet(config['input_size'])
    elif config['model'] == 'MLPNet3Layer':
        model = MLPNet3Layer(config['input_size'])
    elif config['model'] == 'MLPLinear3Layer':
        model = MLPLinear3Layer(config['input_size'])
    elif config['model'] == 'MLPLinear1Layer':
        model = MLPLinear1Layer(config['input_size'])
    elif config['model'] == 'MLPNetSigmoid':
        model = MLPNetSigmoid(config['input_size'])
    else:
        model = MLPNet(config['input_size'])
    if config['loss_function'] == 'BCE':
        loss_function = nn.BCELoss()
    else:
        loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    return model, loss_function, optimizer


def simple_train_batch(trainloader, model, loss_function, optimizer, config):
    model.train()
    for epoch in range(config['epoch_num']):
        # print(config['epoch_num'])
        # if epoch % (config['epoch_num'] // 10) == 0 or epoch == config['epoch_num'] - 1:
        print('current epoch: ', epoch)
        total_loss = 0
        minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=config['simple_train_batch_size'],
                                              shuffle=True)
        # model.train()
        # BCE, (100, 1, 1) doesn't matter
        # MSE, (100, 1, 1) matter
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(), Variable(targets.float().cuda()).squeeze()
            # inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.float().cuda()).squeeze(1)
            # print('inputs', inputs.size())
            # print('targets', targets.size())
            if config['model'] == 'CNN_MNIST':
                inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            # outputs = model(inputs)
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, targets)
            # print('outputs', outputs.size())
            # print('loss', loss)
            total_loss += loss
            loss.backward()
            optimizer.step()
        # if epoch % (config['epoch_num'] // 10) == 0 or epoch == config['epoch_num'] - 1:
        print('loss', total_loss)
    return model, total_loss


def simple_test_batch(testloader, model, config, hidden=False):
    model.eval()
    total = 0.0
    correct = 0.0
    pred_np = []
    hidden_vectors = []
    minibatches_idx = get_minibatches_idx(len(testloader), minibatch_size=config['simple_test_batch_size'],
                                          shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([list(testloader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(np.array([list(testloader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.cuda()).squeeze()
        outputs = model(inputs)
        if hidden:
            hiddens = get_hidden(model, inputs)
            hidden_vectors.extend(list(hiddens.cpu().data.numpy()))
        pred_np.extend(list(outputs.cpu().data.numpy()))
        predicted = (outputs >= 0.5).long().squeeze()
        total += targets.size(0)
        correct += predicted.eq(targets.long()).sum().item()
    test_accuracy = correct / total
    pred_np = np.array(pred_np)
    if hidden:
        return test_accuracy, pred_np, np.array(hidden_vectors)
    return test_accuracy, pred_np


def train_elastic_effect(trainloader, model, loss_function, config, index_list):
    static_model = copy.deepcopy(model)
    total_delta = {}
    total_update = {}
    train_accuracy, prev_prob_list = simple_test_batch(trainloader, model, config)
    no_update_index = []
    for epoch in range(config['epoch_num']):
        print('current epoch: ', epoch)
        update_number = 0
        max_step_lr = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx not in index_list:
                continue
            print('batch idx', batch_idx, end='\r')
            norm_const_condition = 0.0
            step_lr = config['prob_lr']
            while norm_const_condition == 0.0 and step_lr < 10000:
                if step_lr > max_step_lr:
                    max_step_lr = step_lr
                optimizer = optim.SGD(model.parameters(), lr=step_lr)
                model.load_state_dict(static_model.state_dict())
                model.train()
                inputs, targets = Variable(inputs.cuda()), Variable(targets.float().cuda())
                # print('input', inputs.size())
                # print('target', targets.size())
                # print('targets', targets)
                optimizer.zero_grad()
                outputs = model(inputs)
                # print('output', targets.size())
                loss = loss_function(outputs, targets)
                # print('loss', loss)
                loss.backward()
                optimizer.step()
                train_accuracy, prob_list = simple_test_batch(trainloader, model, config)
                if config['method_option'] == 'ripple':
                    delta_prob_list = np.abs(np.array(prob_list) - np.array(prev_prob_list))
                    norm_const = delta_prob_list[batch_idx]
                    norm_const_condition = norm_const
                else:
                    delta_prob_list = np.array(prob_list) - np.array(prev_prob_list)
                    # print('prev prob list', prev_prob_list)
                    # print('prob list', prob_list)
                    # print('delta prob list', delta_prob_list)
                    norm_const_condition = delta_prob_list[batch_idx]
                    delta_prob_list = delta_prob_list / 2 / step_lr
                    norm_const = - (prev_prob_list[batch_idx] - int(targets[0].data.cpu().numpy()))
                    # norm_const_condition = norm_const
                    if config['loss_function'] == 'BCE':
                        norm_const /= prev_prob_list[batch_idx]
                        norm_const /= (1 - prev_prob_list[batch_idx])
                step_lr *= 3
            if norm_const_condition != 0.0:
                delta_prob_list /= norm_const
                # print('normalized delta prob list', delta_prob_list)
                update_number += 1
                if batch_idx in total_delta:
                    total_delta[batch_idx] += delta_prob_list[index_list]
                    total_update[batch_idx] += 1
                else:
                    total_delta[batch_idx] = delta_prob_list[index_list]
                    total_update[batch_idx] = 1
            else:
                print('not update index', batch_idx)
                no_update_index.append(batch_idx)
            if batch_idx == len(trainloader) - 1:
                print('train accuracy', train_accuracy)
        print('update number', update_number)
        print('max step lr', max_step_lr)
        print('no update index', no_update_index)
    for key in total_delta.keys():
        total_delta[key] /= total_update[key]
    return total_delta


def get_similarity_matrix(total_delta, index_list, option='kernel', sym=True):
    similarity_matrix = np.zeros((len(index_list), len(index_list)))
    index_2_idx = {}
    for idx in range(len(index_list)):
        index_2_idx[index_list[idx]] = idx
    self_avg = 0
    other_avg = 0
    for idx in range(len(index_list)):
        index = index_list[idx]
        if index in total_delta.keys():
            similarity_matrix[idx] = total_delta[index].reshape(len(index_list))
            self_avg += similarity_matrix[idx][idx]
            other_avg += np.sum(similarity_matrix[idx])
            other_avg -= similarity_matrix[idx][idx]
    assert len(total_delta) == len(index_list)
    if sym:
        similarity_matrix = (similarity_matrix + np.transpose(similarity_matrix)) / 2
    print('similarity matrix', similarity_matrix)

    if option == 'normalized-kernel':
        for i in range(len(index_list)):
            if similarity_matrix[i][i] < 1e-6:
                print(similarity_matrix[i][i])
                print(similarity_matrix[i])
        normalized_similarity_matrix = np.zeros((len(index_list), len(index_list)))
        for i in range(len(index_list)):
            for j in range(len(index_list)):
                normalized_similarity_matrix[i][j] = similarity_matrix[i][j] / \
                                                     np.sqrt(similarity_matrix[i][i] * similarity_matrix[j][j])
        similarity_matrix = normalized_similarity_matrix
        print('normalized-kernel')
    return similarity_matrix


def get_hidden(model, x):
    out = x.view(x.size(0), -1)
    hidden = (model.fc1(out) >= 0).float()
    return hidden


def get_cosine_similarity(a, b):
    return np.sum(a * b) / np.sqrt(np.sum(a * a)) / np.sqrt(np.sum(b * b))


def get_activation_patterns(trainloader, model, config, index_list):
    total_delta = {}
    train_accuracy, prev_prob_list, hiddens = simple_test_batch(trainloader, model, config, hidden=True)
    hiddens = hiddens[index_list]
    # optimal_file = '/path/to/working/dir/data/optimal_hidden.pickle'
    # pickle.dump(hiddens, open(optimal_file, 'wb'))
    # print('store optimal hidden')
    print(hiddens.shape)
    print(np.sum(hiddens, axis=1))
    print(hiddens[0])
    for idx in range(len(index_list)):
        index = index_list[idx]
        cur_hidden_similarity = []
        for j in range(len(index_list)):
            cur_hidden_similarity.append(get_cosine_similarity(hiddens[idx], hiddens[j]))
        total_delta[index] = np.array(cur_hidden_similarity)
    print(total_delta)
    return total_delta

