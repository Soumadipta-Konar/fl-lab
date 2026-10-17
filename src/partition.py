import numpy as np
import torch
import torchvision.transforms.functional as TF

def iid_partition(y, num_clients=100, seed=42):
    rng = np.random.default_rng(seed)
    num_samples = len(y)
    indices = rng.permutation(num_samples)
    splits = np.array_split(indices, num_clients)
    return {i: splits[i] for i in range(num_clients)}

def pathological_partition(y, num_clients=100, num_shards=200, seed=42):
    rng = np.random.default_rng(seed)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
    sorted_indices = np.argsort(y_np)

    shard_size = len(y_np) // num_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    shard_order = rng.permutation(num_shards)

    shards_per_client = num_shards // num_clients
    partition = {}
    for i in range(num_clients):
        client_shards = shard_order[i * shards_per_client:(i + 1) * shards_per_client]
        partition[i] = np.concatenate([shards[s] for s in client_shards])
    return partition

def dirichlet_partition(y, num_clients=100, alpha=0.5, seed=42, min_size=10):
    rng = np.random.default_rng(seed)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
    num_classes = len(np.unique(y_np))

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        c_idx = np.where(y_np == c)[0]
        rng.shuffle(c_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions / proportions.sum())
        splits = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        c_splits = np.split(c_idx, splits)
        for i in range(num_clients):
            client_indices[i].append(c_splits[i])

    partition = {}
    for i in range(num_clients):
        arr = np.concatenate(client_indices[i])
        rng.shuffle(arr)
        partition[i] = arr
    return partition

def quantity_skew_partition(y, num_clients=100, beta=0.5, seed=42):
    rng = np.random.default_rng(seed)
    num_samples = len(y)
    indices = rng.permutation(num_samples)

    proportions = rng.dirichlet(np.repeat(beta, num_clients))
    proportions = proportions / proportions.sum()
    splits = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
    split_indices = np.split(indices, splits)

    return {i: split_indices[i] for i in range(num_clients)}

def feature_skew_partition(x, y, num_clients=100, seed=42):
    base_partition = iid_partition(y, num_clients=num_clients, seed=seed)
    transformed_x = x.clone()

    for i in range(num_clients):
        idx = base_partition[i]
        angle = (360.0 / num_clients) * i
        client_imgs = transformed_x[idx]
        transformed_x[idx] = TF.rotate(client_imgs, angle)

    return base_partition, transformed_x

def get_label_distribution(y, partition, num_classes=10):
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)
    num_clients = len(partition)
    dist = np.zeros((num_clients, num_classes), dtype=int)
    for i, idx in partition.items():
        labels, counts = np.unique(y_np[idx], return_counts=True)
        for lbl, cnt in zip(labels, counts):
            dist[i, lbl] = cnt
    return dist

def save_partition(partition, filepath):
    np.savez(filepath, **{str(k): v for k, v in partition.items()})

def load_partition(filepath):
    data = np.load(filepath)
    return {int(k): data[k] for k in data.files}
