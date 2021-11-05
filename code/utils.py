import numpy as np
import torch
import logging
import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math
import sklearn.metrics
import faiss
from typing import List, Optional, Any, Dict
from map import *

import faiss
import numpy as np
import sklearn.cluster
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn
import sklearn.cluster
import sklearn.metrics.cluster


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def assign_by_euclidian_at_k(X, T, k):
    """
        X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
        k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)
    chunk_size = 1000
    num_chunks = math.ceil(len(X)/chunk_size)
    distances = torch.tensor([])
    for i in tqdm(range(0, num_chunks)):
        chunk_indices = [chunk_size*i, min(len(X), chunk_size*(i+1))]
        chunk_X = X[chunk_indices[0]:chunk_indices[1], :]
        distance_mat = torch.from_numpy(sklearn.metrics.pairwise.pairwise_distances(X, chunk_X))
        distances = torch.cat((distances, distance_mat), dim=-1)
    assert distances.shape[0] == len(X)
    assert distances.shape[1] == len(X)

    distances = distances.numpy()
    # get nearest points
    indices = np.argsort(distances, axis = 1)[:, 1 : k + 1]

    return np.array([[T[i] for i in ii] for ii in indices])

def calc_recall_at_k(T, Y, k):
    """
        Check whether a sample's KNN contain any sample with the same class labels as itself
        T : [nb_samples] (target labels)
        Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    '''
        Predict on a batch
        :return: list with N lists, where N = |{image, label, index}|
    '''
    # print(list(model.parameters())[0].device)
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc="Batch-wise prediction"):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).cpu()
                for j in J:
                    #if i == 1: print(j)
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

@torch.no_grad()
def recall_at_ks(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = True) -> (Any, Dict[int, float]):
    """
    More efficient way to compute the recall between samples at each k. This function uses about 8GB of memory.

    """
    nmi = None
    offset = 0
    if gallery_features is None and gallery_labels is None:
        offset = 1
        gallery_features = query_features
        gallery_labels = query_labels
    elif gallery_features is None or gallery_labels is None:
        raise ValueError('gallery_features and gallery_labels needs to be both None or both Tensors.')

    if cosine:
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])

    if hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0

        max_k = max(ks)
        index_function = faiss.GpuIndexFlatIP if cosine else faiss.GpuIndexFlatL2
        index = index_function(res, g_f.shape[1], flat_config)
    else:
        max_k = max(ks)
        index_function = faiss.IndexFlatIP if cosine else faiss.IndexFlatL2
        index = index_function(g_f.shape[1])
    index.add(g_f)
    closest_indices = index.search(q_f, max_k + offset)[1]

    recalls = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        recalls[k] = (q_l[:, None] == g_l[indices]).any(1).mean()
    return {k: round(v * 100, 2) for k, v in recalls.items()}

def mapr(X, T):
    # MAP@R
    label_counts = get_label_match_counts(T, T) # get R
    # num_k = determine_k(
    #     num_reference_embeddings=len(T), embeddings_come_from_same_source=True
    # ) # equal to num_reference-1 (deduct itself)

    num_k = max([count[1] for count in label_counts])
    knn_indices = get_knn(
        X, X, num_k, True
    )

    knn_labels = T[knn_indices] # get KNN indicies
    map_R = mean_average_precision_at_r(knn_labels=knn_labels,
                                        gt_labels=T[:, None],
                                        embeddings_come_from_same_source=True,
                                        label_counts=label_counts,
                                        avg_of_avgs=False,
                                        label_comparison_fn=torch.eq)
    logging.info("MAP@R:{:.3f}".format(map_R * 100))

    return map_R


def mapr_inshop(X_query, T_query, X_gallery, T_gallery):
    # MAP@R
    label_counts = get_label_match_counts(T_query, T_gallery)  # get R
    # num_k = determine_k(
    #     num_reference_embeddings=len(T_gallery), embeddings_come_from_same_source=False
    # )  # equal to num_reference
    num_k = max([count[1] for count in label_counts])

    knn_indices = get_knn(
        X_gallery, X_query, num_k, True
    )
    knn_labels = T_gallery[knn_indices]  # get KNN indicies
    map_R = mean_average_precision_at_r(knn_labels=knn_labels,
                                        gt_labels=T_query[:, None],
                                        embeddings_come_from_same_source=False,
                                        label_counts=label_counts,
                                        avg_of_avgs=False,
                                        label_comparison_fn=torch.eq)
    logging.info("MAP@R:{:.3f}".format(map_R * 100))
    return map_R

def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
     # return sklearn.cluster.MiniBatchKMeans(nb_clusters, batch_size=32).fit(X).labels_
    X = X.detach().cpu().numpy()
    kmeans = faiss.Kmeans(d=X.shape[1], k=nb_clusters)
    kmeans.train(X.astype(np.float32))
    labels = kmeans.index.search(X.astype(np.float32), 1)[1]
    return np.squeeze(labels, 1)

def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys, average_method='geometric')


def calc_nmi(X, T, nb_classes):
    # calculate NMI with kmeans clustering
    nmi = calc_normalized_mutual_information(
        T,
        cluster_by_kmeans(
            X, nb_classes
        )
    )
    print(nmi)
    return nmi


def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, *_ = predict_batchwise(model, dataloader)

    nmi = calc_nmi(X, T, nb_classes)
    print(nmi)
    recall = recall_at_ks(X, T, ks=[1, 2, 4, 8])
    print("Recall@1,2,4,8 {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(recall[1], recall[2], recall[4], recall[8]))
    map_r = mapr(X, T)
    print(map_r)
    return recall

