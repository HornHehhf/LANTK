import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import cv2

from kernel_kmeans import KernelKMeans


def clustering_images(data, index_list, data_labels):
    images = []
    for index in index_list:
        point = data[index]
        inputs, targets = point
        image = inputs.cpu().data.numpy()[0].reshape(-1)
        images.append(image)
    images = np.array(images)
    km = KMeans(n_clusters=2, random_state=0)
    cluster_labels = km.fit_predict(images)
    cluster_labels += 1
    print(cluster_labels)
    print(data_labels)
    correct_num = np.sum(data_labels == cluster_labels)
    print('correct_num', correct_num)
    print('K-Means correct percentage', 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5))


def clustering_pca_images(data, index_list, data_labels, pca_n_components=100):
    images = []
    for index in index_list:
        point = data[index]
        inputs, targets = point
        image = inputs.cpu().data.numpy()[0].reshape(-1)
        images.append(image)
    images = np.array(images)
    pca = PCA(n_components=pca_n_components, svd_solver='full')
    images = pca.fit_transform(images)
    print('get pca done')
    km = KMeans(n_clusters=2, random_state=0)
    cluster_labels = km.fit_predict(images)
    cluster_labels += 1
    print(cluster_labels)
    print(data_labels)
    correct_num = np.sum(data_labels == cluster_labels)
    print('correct_num', correct_num)
    print('K-Means correct percentage', 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5))


def clustering_images_autoencoder(data, index_list, data_labels, autoencoder):
    images = []
    for index in index_list:
        point = data[index]
        inputs, targets = point
        image = autoencoder.get_features(inputs.cuda()).cpu().data.numpy()[0].reshape(-1)
        images.append(image)
    images = np.array(images)
    km = KMeans(n_clusters=2, random_state=0)
    cluster_labels = km.fit_predict(images)
    cluster_labels += 1
    print(cluster_labels)
    print(data_labels)
    correct_num = np.sum(data_labels == cluster_labels)
    print('correct_num', correct_num)
    print('K-Means correct percentage', 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5))


def clustering_images_resnet152(data, index_list, data_labels, pca_n_components=100):
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    model = models.resnet152(pretrained=True).to('cuda')
    layer = model._modules.get('avgpool')
    model.eval()
    images = []
    img_size = (224, 224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    for index in index_list:
        point = data[index]
        inputs, targets = point
        image = inputs.cpu().data.numpy()[0]
        image = image.transpose((1, 2, 0))
        # C * H * W
        image = cv2.resize(image, img_size)
        # resize
        image = (image + 1) * 0.5
        # [-1, 1] => [0, 1]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        t_img = Variable(normalize(to_tensor(image)).unsqueeze(0)).cuda().float()
        my_embedding = torch.zeros(2048)
        h = layer.register_forward_hook(copy_data)
        # print(model(t_img).shape)
        model(t_img)
        h.remove()
        resnet152_feature = my_embedding.data.cpu().numpy()
        image = np.array(resnet152_feature).flatten()
        images.append(image)
    images = np.array(images)
    print(data_labels)
    print('get resnet152 features done')
    km = KMeans(n_clusters=2, random_state=0)
    cluster_labels = km.fit_predict(images)
    cluster_labels += 1
    print(cluster_labels)
    correct_num = np.sum(data_labels == cluster_labels)
    print('K-Means correct percentage', 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5))
    pca = PCA(n_components=pca_n_components, svd_solver='full')
    images = pca.fit_transform(images)
    print('get pca done')
    km = KMeans(n_clusters=2, random_state=0)
    cluster_labels = km.fit_predict(images)
    cluster_labels += 1
    print(cluster_labels)
    print(data_labels)
    correct_num = np.sum(data_labels == cluster_labels)
    print('correct_num', correct_num)
    print('K-Means PCA correct percentage', 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5))


def similarity_clustering(similarity_matrix, data_labels):
    km = KMeans(n_clusters=2, random_state=0)
    cluster_labels = km.fit_predict(similarity_matrix)
    cluster_labels += 1
    correct_num = np.sum(data_labels == cluster_labels)
    print(cluster_labels)
    print('correct num', correct_num)
    print('K-Means correct percentage', 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5))
    k_means_result = 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5)
    sc = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='discretize')
    cluster_labels = sc.fit_predict(similarity_matrix)
    cluster_labels += 1
    correct_num = np.sum(data_labels == cluster_labels)
    print(cluster_labels)
    print('correct num', correct_num)
    print('Spectral Clustering correct percentage', 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5))
    return k_means_result


def kernel_clustering(similarity_matrix, data_labels):
    '''
    eigen_values, eigen_vectors = np.linalg.eig(similarity_matrix)
    sorted_indices = np.argsort(eigen_values)
    eigen_vectors = eigen_vectors[:, sorted_indices]
    eigen_values = eigen_values[sorted_indices]
    topk_evecs = eigen_vectors[:, -dim:]
    topk_evals = eigen_values[-dim:]
    print('positive eigenvalues', np.sum(eigen_values >= 0.0))
    diagonal_matrix = np.zeros((dim, dim))
    for index in range(10):
        diagonal_matrix[index][index] = np.sqrt(topk_evals[index])
    pca_features = np.dot(topk_evecs, diagonal_matrix)
    similarity_matrix = np.dot(pca_features, np.transpose(pca_features))
    print('similarity_matrix', similarity_matrix)
    '''
    km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1)
    cluster_labels = km.fit(similarity_matrix).predict(similarity_matrix)
    cluster_labels += 1
    print(cluster_labels)
    correct_num = np.sum(data_labels == cluster_labels)
    print('correct_num', correct_num)
    print('Kernel K-Means correct percentage', 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5))
    k_means_result = 0.5 + np.abs(correct_num / len(cluster_labels) - 0.5)
    return k_means_result
