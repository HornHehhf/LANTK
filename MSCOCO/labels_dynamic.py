import sys
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim

from utils import set_random_seed, shuffle_in_union, plot_delta_versus_distance
from data import save_data_mscoco, load_data_mscoco_from_pickle
from elastic_effect import build_model, simple_test_batch, train_elastic_effect, get_similarity_matrix, \
    simple_train_batch
from clustering import clustering_images, clustering_pca_images, clustering_images_resnet152

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def elastic_effect_analysis_model(data, index_list, config, model, loss_function):
    data_res, data_np = simple_test_batch(data, model, config)
    print('data accuracy', data_res)
    config['epoch_num'] = 1
    total_delta = train_elastic_effect(data, model, loss_function, config, index_list)
    similarity_matrix = get_similarity_matrix(total_delta, index_list, option='normalized-kernel', sym=False)
    data_res, data_np = simple_test_batch(data, model, config)
    print('data accuracy', data_res)
    return similarity_matrix


def get_labels_dynamic_results(train_data, index_list, config, model, loss_function, train_data_label, labels):
    similarity_matrix = elastic_effect_analysis_model(train_data, index_list, config, model, loss_function)
    print(train_data_label[index_list])
    print(similarity_matrix)
    same_class = []
    between_classes = []
    data_label = train_data_label[index_list]
    four_classes = [[], [], [], []]
    for idx in range(len(data_label)):
        for idy in range(len(data_label)):
            if data_label[idx] == data_label[idy]:
                same_class.append(similarity_matrix[idx][idy])
            else:
                between_classes.append(similarity_matrix[idx][idy])
            if labels[idx][0] != labels[idy][0] and labels[idx][1] != labels[idy][1]:
                four_classes[0].append(similarity_matrix[idx][idy])
            elif labels[idx][0] != labels[idy][0] and labels[idx][1] == labels[idy][1]:
                four_classes[1].append(similarity_matrix[idx][idy])
            elif labels[idx][0] == labels[idy][0] and labels[idx][1] != labels[idy][1]:
                four_classes[2].append(similarity_matrix[idx][idy])
            else:
                four_classes[3].append(similarity_matrix[idx][idy])

    average_same = np.mean(same_class)
    average_between = np.mean(between_classes)
    four_classes = np.mean(four_classes, axis=1)
    # print('same class', same_class)
    # print('between class', between_classes)
    print('average same', average_same)
    print('average between', average_between)
    print('four classes', four_classes)
    return average_same, average_between, four_classes


def run_elastic_effect_analysis_dynamic():
    dataset = sys.argv[1].split('=')[1]
    method_option = sys.argv[2].split('=')[1]
    model_option = sys.argv[3].split('=')[1]
    loss_option = sys.argv[4].split('=')[1]
    pos_one = sys.argv[5].split('=')[1]
    pos_two = sys.argv[6].split('=')[1]
    neg_one = sys.argv[7].split('=')[1]
    neg_two = sys.argv[8].split('=')[1]
    label_system = sys.argv[9].split('=')[1]
    print(label_system)
    dir_path = '/path/to/working/dir'
    config = {'batch_size': 1, 'epoch_num': 200, 'lr': 3e-4, 'test_batch_size': 1, 'sample_size': 200,
              'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'labels': {'pos': [pos_one, pos_two], 'neg': [neg_one, neg_two],
                         'all': [pos_one, pos_two, neg_one, neg_two]},
              'method_option': method_option, 'prob_lr': 1e-6, 'repeat_num': 1, 'label_system': label_system,
              'dataset': dataset, 'model': model_option, 'loss_function': loss_option, 'prob_num': 10}
    set_random_seed(666)
    if model_option == 'MLPLinear':
        config['lr'] = 3e-5
    config['input_size'] = 3 * 32 * 32
    print('setting: ', config['labels']['all'])
    print('save data')
    save_data_mscoco(config, dir_path)
    print('load data from pickle')
    data_one, data_two, labels = load_data_mscoco_from_pickle(config, dir_path)
    if config['label_system'] == 'one':
        train_data = np.array(data_one)
    else:
        train_data = np.array(data_two)
    labels = np.array(labels)
    train_data, labels = shuffle_in_union(train_data, labels)
    train_data_label = [train_data[x][1].numpy()[0] for x in range(len(train_data))]
    train_data_label = np.array(train_data_label)
    index_list = []
    for idx in range(len(train_data)):
        if train_data_label[idx] >= 0:
            index_list.append(idx)
    print('clustering images')
    clustering_images(train_data, index_list, train_data_label[index_list])
    clustering_pca_images(train_data, index_list, train_data_label[index_list])
    print('build model')
    model, loss_function, optimizer = build_model(config)
    loss_list = []
    acc_list = []
    same_list = []
    between_list = []
    four_classes_list = []
    epoch_num = config['epoch_num'] // config['prob_num']
    for i in range(config['prob_num'] + 1):
        if i == 0:
            train_res, train_np = simple_test_batch(train_data, model, config)
            print('train accuracy', train_res)
            static_model = copy.deepcopy(model)
            average_same, average_between, four_classes = get_labels_dynamic_results(train_data, index_list, config,
                                                                                     model, loss_function,
                                                                                     train_data_label, labels)
            loss_list.append('inf')
            acc_list.append(train_res)
            same_list.append(average_same)
            between_list.append(average_between)
            four_classes_list.append(four_classes)
            model = copy.deepcopy(static_model)
            if config['loss_function'] == 'BCE':
                loss_function = nn.BCELoss()
            else:
                loss_function = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=config['lr'])
        else:
            print('prob index', i)
            config['epoch_num'] = epoch_num
            model, total_loss = simple_train_batch(train_data, model, loss_function, optimizer, config)
            print('train loss', total_loss)
            train_res, train_np = simple_test_batch(train_data, model, config)
            print('train accuracy', train_res)
            static_model = copy.deepcopy(model)
            average_same, average_between, four_classes = get_labels_dynamic_results(train_data, index_list, config,
                                                                                     model, loss_function,
                                                                                     train_data_label, labels)
            loss_list.append(total_loss)
            acc_list.append(train_res)
            same_list.append(average_same)
            between_list.append(average_between)
            four_classes_list.append(four_classes)
            print(config)
            model = copy.deepcopy(static_model)
            if config['loss_function'] == 'BCE':
                loss_function = nn.BCELoss()
            else:
                loss_function = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    four_classes_list = np.array(four_classes_list)
    four_classes_sum = np.sum(four_classes_list, axis=1)
    normalized_four_classes = [four_classes_list[x] / four_classes_sum[x] for x in range(len(four_classes_sum))]
    four_classes_list = four_classes_list.transpose()
    normalized_four_classes_delta = [normalized_four_classes[x+1] - normalized_four_classes[x] for x in
                                     range(len(normalized_four_classes) - 1)]
    normalized_four_classes = np.array(normalized_four_classes).transpose()
    normalized_four_classes_delta = np.array(normalized_four_classes_delta).transpose()
    print(np.sum(normalized_four_classes, axis=1))
    print('loss list', loss_list)
    print('acc list', acc_list)
    print('same list', same_list)
    print('between list', between_list)
    print('four classes list', four_classes_list)
    normalized_same_list = [same_list[x] / (same_list[x] + between_list[x]) for x in range(len(same_list))]
    normalized_between_list = [between_list[x] / (same_list[x] + between_list[x]) for x in range(len(between_list))]
    normalized_same_delta = [normalized_same_list[x+1] - normalized_same_list[x] for x in
                             range(len(normalized_same_list) - 1)]
    normalized_between_delta = [normalized_between_list[x+1] - normalized_between_list[x] for x in
                                range(len(normalized_between_list) - 1)]
    print('normalized same list', normalized_same_list)
    print('normalized between list', normalized_between_list)
    print('normalized four classes', normalized_four_classes)

    print('normalized same delta', np.around(np.array(normalized_same_delta) * 10000, decimals=2))
    print('normalized between delta', np.around(np.array(normalized_between_delta) * 10000, decimals=2))
    print('normalized four classes delta', np.around(np.array(normalized_four_classes_delta) * 10000, decimals=2))


if __name__ == '__main__':
    run_elastic_effect_analysis_dynamic()
