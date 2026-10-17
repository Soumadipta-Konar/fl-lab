import copy
import torch
import torch.nn as nn
from .utils import evaluate

def client_update(model, x, y, epochs=1, batch_size=10, lr=0.01, device=None):
    if device is None:
        device = next(model.parameters()).device
    local_model = copy.deepcopy(model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_samples = len(y)
    bs = num_samples if (batch_size is None or batch_size >= num_samples) else batch_size

    for _ in range(epochs):
        perm = torch.randperm(num_samples)
        for i in range(0, num_samples, bs):
            b_idx = perm[i:i + bs]
            bx = x[b_idx].to(device)
            by = y[b_idx].to(device)
            optimizer.zero_grad()
            out = local_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    return local_model.state_dict()

def fedavg_round(global_model, client_data, selected_clients, epochs=1, batch_size=10, lr=0.01, device=None):
    if device is None:
        device = next(global_model.parameters()).device

    total_samples = sum(len(client_data[c][1]) for c in selected_clients)
    global_dict = global_model.state_dict()
    new_dict = copy.deepcopy(global_dict)
    for key in new_dict:
        new_dict[key] = torch.zeros_like(new_dict[key], dtype=torch.float32)

    client_updates = {}
    for c in selected_clients:
        cx, cy = client_data[c]
        c_weights = client_update(global_model, cx, cy, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
        client_updates[c] = c_weights
        weight = len(cy) / total_samples
        for key in new_dict:
            new_dict[key] += c_weights[key] * weight

    global_model.load_state_dict(new_dict)
    return client_updates

def run_fl(global_model, client_data, test_x, test_y, num_rounds=20, clients_per_round=10,
           epochs=1, batch_size=10, lr=0.01, target_acc=None, device=None):
    if device is None:
        device = next(global_model.parameters()).device

    num_clients = len(client_data)
    history = []

    test_loss, test_acc = evaluate(global_model, test_x, test_y, device=device)
    history.append({
        'round': 0,
        'test_loss': test_loss,
        'test_acc': test_acc
    })

    for rnd in range(1, num_rounds + 1):
        perm = torch.randperm(num_clients)
        selected = perm[:clients_per_round].tolist()
        fedavg_round(global_model, client_data, selected, epochs=epochs, batch_size=batch_size, lr=lr, device=device)

        test_loss, test_acc = evaluate(global_model, test_x, test_y, device=device)
        history.append({
            'round': rnd,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

        if target_acc is not None and test_acc >= target_acc:
            break

    return history
