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


def defense(graph, budget, eps, target, w1, w2):
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
trainer = gg.gallery.nodeclas.GCN(device='cpu', seed=42).make_data(graph).build()
trainer.fit(splits.train_nodes, splits.val_nodes)
w1, w2 = trainer.model.parameters()
w1 = w1.t()
w2 = w2.t()
# attacker model
W = w1 @ w2
W = gf.tensoras(W)
attacker = gg.attack.targeted.Nettack(graph).process(W)
# attacker = gg.attack.targeted.FGSM(graph).process(trainer)


"""
Defender Model
"""
# GCN for defender
trainer = gg.gallery.nodeclas.GCN(device='cpu', seed=100).make_data(graph).build()
trainer.fit(splits.train_nodes, splits.val_nodes)
w1, w2 = trainer.model.parameters()
w1 = w1.t()
w2 = w2.t()


"""
Generate attacked_g, clean_defended_g, attacked_defended_g
"""
# set target, budget, eps
# target = np.random.choice(splits.test_nodes, 1)[0]
target = 1000
budget = 1
eps = 0.5

# attacked_g
attacker.set_max_perturbations()
attacker.reset()
attacker.attack(target,
                direct_attack=True,
                structure_attack=True,
                feature_attack=False)
attack_g = attacker.g
print(f'{attacker.num_budgets} edges has been modified.')

# clean_defended_g
clean_defended_g = defense(graph, budget, eps, target, w1, w2)

# attacked_defended_g
attacked_defended_g = defense(attack_g, budget, eps, target, w1, w2)


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

# attacked graph
trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(attack_g).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=0,
                  epochs=100)
attacked_predict = trainer.predict(target, transform="softmax")

# clean defended graph

# attacked defended graph
