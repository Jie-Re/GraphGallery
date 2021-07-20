import matplotlib.pyplot as plt

import graphgallery as gg
import graphgallery.functional as gf
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gg.set_backend("th")


def forward(A, X, target, w1, w2):
    A = gf.normalize_adj_tensor(A)
    h1 = F.relu(A @ X @ w1)
    h2 = A @ h1 @ w2
    out = h2[target]
    return out.view(1, -1)


def defense(graph, budget, eps, target):
    """
    Defender Model
    @param graph: clean graph or attacked graph
    @param budget: maximum number of modification
    @param eps: calibration constant
    @param target: target node
    @return: graph after defensive perturbation
    """
    trainer = gg.gallery.nodeclas.GCN(device='cpu', seed=100).make_data(graph).build()
    trainer.fit(splits.train_nodes, splits.val_nodes)
    w1, w2 = trainer.model.parameters()
    w1 = w1.t()
    w2 = w2.t()

    A = gf.astensor(graph.A)
    X = gf.astensor(graph.x)
    A = nn.Parameter(A.to_dense())

    t = gf.astensor(target)
    C = graph.num_classes

    loss_fn = nn.CrossEntropyLoss()

    edges = []
    for _ in range(int(budget)):
        out = forward(A, X, t, w1, w2)
        coeff = F.softmax(out / eps, dim=-1).view(-1)

        loss = 0
        for c in torch.arange(C):
            loss += loss_fn(out, torch.tensor([c])) * coeff[c]

        adj_grad = torch.autograd.grad(loss.sum(), A, create_graph=False)[0]

        # 计算梯度最大值对应的边
        N = adj_grad.size(0)
        # gradient ascent
        # if A_ij is 1 then the edge should be removed
        # if A_ij is 0 then the edge should be added
        # if adj_grad_ij > 0 then the edge would be added
        # if adj_grad_ij < 0 then the edge would be removed
        adj_grad *= 1 - 2 * A
        adj_grad = adj_grad[t]
        I = adj_grad.argmax()

        # row = I.floor_divide(N)
        row = t
        col = I.fmod(N)
        A[row, col] = A[col, row] = 1 - A[row, col]

        edges.append([row.item(), col.item()])

    defense_g = graph.from_flips(edge_flips=edges)
    return defense_g


"""
Data Preprocessing
"""
data = gg.datasets.NPZDataset('cora', root='~/GraphData/datasets', transform='standardize')
graph = data.graph
splits = data.split_nodes(random_state=15)


"""
Attacker Model
"""
# GCN for attacker
trainer_a = gg.gallery.nodeclas.GCN(device='cpu', seed=42).make_data(graph).build()
trainer_a.fit(splits.train_nodes, splits.val_nodes)
w1_a, w2_a = trainer_a.model.parameters()
w1_a = w1_a.t()
w2_a = w2_a.t()
# attacker model
W_a = w1_a @ w2_a
W_a = gf.tensoras(W_a)
attacker = gg.attack.targeted.Nettack(graph).process(W_a)
# attacker = gg.attack.targeted.FGSM(graph).process(trainer)


"""
Generate attacked_g, clean_defended_g, attacked_defended_g
"""
# set target, budget, eps
# target = np.random.choice(splits.test_nodes, 1)[0]
target = 1000
budget = 1
# eps = 1
eps_range = list(np.arange(0.01, 1, 0.01))

# true label
target_label = graph.node_label[target]

# attacked_g
attacker.set_max_perturbations()
attacker.reset()
attacker.attack(target,
                direct_attack=True,
                structure_attack=True,
                feature_attack=False)
attack_g = attacker.g
print(f'attacker: {attacker.num_budgets} edges has been modified.')

# eps_loop
clean_change = []
attacked_change = []
plt.figure(1)
plot_flag = 1
for eps in eps_range:
    # clean_defended_g
    clean_defended_g = defense(graph, budget, eps, target)

    # attacked_defended_g
    attacked_defended_g = defense(attack_g, budget, eps, target)


    """
    Prediction
    """
    # clean graph
    trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(graph).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=0,
                      epochs=100)
    clean_predict = trainer.predict(target, transform="softmax")
    clean_label = np.argmax(clean_predict)

    # attacked graph
    trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(attack_g).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=0,
                      epochs=100)
    attacked_predict = trainer.predict(target, transform="softmax")
    attacked_label = np.argmax(attacked_predict)

    # clean defended graph
    trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(clean_defended_g).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=0,
                      epochs=100)
    clean_defended_predict = trainer.predict(target, transform="softmax")
    clean_defended_label = np.argmax(clean_defended_predict)

    # attacked defended graph
    trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(attacked_defended_g).build()
    his = trainer.fit(splits.train_nodes,
                      splits.val_nodes,
                      verbose=0,
                      epochs=100)
    attacked_defended_predict = trainer.predict(target, transform="softmax")
    attacked_defended_label = np.argmax(attacked_defended_predict)

    # change list
    clean_change.append(clean_predict[clean_label] - clean_defended_predict[clean_defended_label])
    attacked_change.append(attacked_predict[attacked_label] - attacked_defended_predict[attacked_defended_label])

    # subplot
    if eps * 10 == plot_flag:
        x1 = range(7)
        plt.subplot(3, 3, plot_flag)
        plt.xlabel('categories')
        plt.ylabel('probability')
        plt.plot(x1, clean_predict, color='green', linewidth=1.0, linestyle='-', label='clean')
        plt.plot(x1, clean_defended_predict, color='blue', linewidth=1.0, linestyle='-', label='clean_defended')
        plt.plot(x1, attacked_predict, color='red', linewidth=1.0, linestyle='-', label='attacked')
        plt.plot(x1, attacked_defended_predict, color='orange', linewidth=1.0, linestyle='-', label='attacked_defended')
        plt.legend(['clean', 'clean_defended', 'attacked', 'attacked_defended'])
        plt.title(f'eps={eps}, target={target}')
        plot_flag += 1
plt.show()

"""
Plot
"""
plt.figure(2)
x2 = eps_range
plt.xlabel('eps')
plt.ylabel('change')
plt.plot(x2, clean_change, color='green', linewidth=1.0, linestyle='-', label='change of clean graph')
plt.plot(x2, attacked_change, color='red', linewidth=1.0, linestyle='-', label='change of attacked graph')
plt.legend(['change of clean graph', 'change of attacked graph'])
plt.show()
