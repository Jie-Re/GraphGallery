import graphgallery as gg
import graphgallery.functional as gf
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt

gg.set_backend("th")


def softmax_cross_entropy_with_logits(labels, logits, dim=-1):
    return (-labels * F.log_softmax(logits, dim=dim)).sum(dim=dim)


def kld_with_logits(logit_q, logit_p):
    q = F.softmax(logit_q, dim=-1)
    cross_entropy = softmax_cross_entropy_with_logits(logits=logit_p, labels=q)
    entropy = softmax_cross_entropy_with_logits(logits=logit_q, labels=q)
    return (cross_entropy - entropy).mean()


def forward(A, X, target):
    A = gf.normalize_adj_tensor(A)
    h1 = F.relu(A @ X @ w1)
    h2 = A @ h1 @ w2
    out = h2[target]
    return out.view(1, -1)


def defense(graph, budget=1, eps=1):
    A = gf.astensor(graph.A)
    X = gf.astensor(graph.x)
    A = nn.Parameter(A.to_dense())

    t = gf.astensor(target)
    C = graph.num_classes

    loss_fn = nn.CrossEntropyLoss()

    edges = []
    for _ in range(int(budget)):
        out = forward(A, X, t)
        #         norm = torch.ones_like(out) / out.size(-1)
        #         loss = -(kld_with_logits(norm, out) + kld_with_logits(out, norm))

        #         best_wrong = (out-torch.eye(7)[out.argmax()]*1000).argmax()
        #         loss = loss_fn(out, torch.tensor([out.argmax()]))
        coeff = F.softmax(out / eps, dim=-1).view(-1)
        #         out = out.repeat(C, 1)
        #         loss = loss_fn(out, torch.arange(C)) * coeff

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


data = gg.datasets.NPZDataset('cora', root='~/GraphData/datasets', transform='standardize')
graph = data.graph
splits = data.split_nodes(random_state=15)

# GCN for attacker
trainer = gg.gallery.nodeclas.GCN(device='cpu', seed=42).make_data(graph).build()
trainer.fit(splits.train_nodes, splits.val_nodes)

w1, w2 = trainer.model.parameters()
w1 = w1.t()
w2 = w2.t()

################### Attacker model ############################
W = w1 @ w2
W = gf.tensoras(W)
attacker = gg.attack.targeted.Nettack(graph).process(W)
# attacker = gg.attack.targeted.FGSM(graph).process(trainer)

# GCN for defender
trainer = gg.gallery.nodeclas.GCN(device='cpu', seed=100).make_data(graph).build()
trainer.fit(splits.train_nodes, splits.val_nodes)
w1, w2 = trainer.model.parameters()
w1 = w1.t()
w2 = w2.t()

# set target
# target = np.random.choice(splits.test_nodes, 1)[0]
target = 1000

attacker.set_max_perturbations()
attacker.reset()
attacker.attack(target,
                direct_attack=True,
                structure_attack=True,
                feature_attack=False)
attack_g = attacker.g
print(f'{attacker.num_budgets} edges has been modified.')

################### Victim model ############################
# Before attack
trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(graph).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=0,
                  epochs=100)
clean_predict = trainer.predict(target, transform="softmax")

# After attack
trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(attack_g).build()
his = trainer.fit(splits.train_nodes,
                  splits.val_nodes,
                  verbose=0,
                  epochs=100)
attacked_predict = trainer.predict(target, transform="softmax")
target_label = graph.node_label[target]

#TODO:
clean_changes = []
attacked_changes = []
for i in range(9):
    eps_range = list(np.arange(i, i+1.1, 0.1))
    ################### Defense model ############################
    budget = 1
    # eps = 1
    # eps_range = list(np.arange(3, 4, 0.01))
    clean_change = []
    attacked_change = []

    for eps in eps_range:
        # Defense for clean
        trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(defense(graph, budget, eps)).build()
        his = trainer.fit(splits.train_nodes,
                          splits.val_nodes,
                          verbose=0,
                          epochs=100)
        clean_defended_predict = trainer.predict(target, transform="softmax")

        # Defense for attacked
        trainer = gg.gallery.nodeclas.GCN(seed=1234567).make_data(defense(attack_g, budget, eps)).build()
        his = trainer.fit(splits.train_nodes,
                          splits.val_nodes,
                          verbose=0,
                          epochs=100)
        attacked_defended_predict = trainer.predict(target, transform="softmax")

        ################### Results ############################
        #     print("eps ", eps)
        #     print("target node ", target)
        #     print('true label', target_label)

        #     print('clean prediction', np.round(clean_predict, 3))
        #     print('defended clean prediction', np.round(clean_defended_predict, 3))
        #     print('attacked prediction', np.round(attacked_predict, 3))
        #     print('defended attacked prediction', np.round(attacked_defended_predict, 3))

        #     print("clean predict label ", np.argmax(clean_predict))
        #     print("defended clean predict label ", np.argmax(clean_defended_predict))
        #     print("attacked predict label ", np.argmax(attacked_predict))
        #     print("defended attacked predict label ", np.argmax(attacked_defended_predict))

        clean_predict_label = np.argmax(clean_predict)
        attacked_predict_label = np.argmax(attacked_predict)

        # print( f"The probability change is (before-after) {clean_predict[clean_predict_label] - clean_defended_predict[
        # clean_predict_label]}")
        # print( f"The probability change is (before-after) {attacked_predict[attacked_predict_label] -
        # attacked_defended_predict[attacked_predict_label]}")

        clean_change.append(clean_predict[clean_predict_label] - clean_defended_predict[clean_predict_label])
        attacked_change.append(attacked_predict[attacked_predict_label] - attacked_defended_predict[attacked_predict_label])
        if np.argmax(clean_predict) != np.argmax(clean_defended_predict):
            print(
                f"predict label for clean graph has been changed from {np.argmax(clean_predict)} to {np.argmax(clean_defended_predict)}")
        if np.argmax(attacked_predict) != np.argmax(attacked_defended_predict):
            print(
                f"predict label for attacked graph has been changed from {np.argmax(attacked_predict)} to {np.argmax(attacked_defended_predict)}")

        # x = range(7)
        # plt.xlabel('categories')
        # plt.ylabel('probability')
        # plt.plot(x, clean_predict, color='green', linewidth=3.0, linestyle='-', label='clean')
        # plt.plot(x, clean_defended_predict, color='blue', linewidth=3.0, linestyle='-', label='clean_defended')
        # plt.plot(x, attacked_predict, color='red', linewidth=3.0, linestyle='-', label='attacked')
        # plt.plot(x, attacked_defended_predict, color='orange', linewidth=3.0, linestyle='-', label='attacked_defended')
        # plt.legend(['clean', 'clean_defended', 'attacked', 'attacked_defended'])
        # plt.title('eps=1, target=1000')
    clean_changes.append(clean_change)
    attacked_changes.append(attacked_change)

################### Plot #########################
x = eps_range
plt.xlabel('eps')
plt.ylabel('change')
plt.subplot(3, 3, i+1)
plt.plot(x, clean_change, color='green', linewidth=1.0, linestyle='-', label='change of clean graph')
plt.plot(x, attacked_change, color='red', linewidth=1.0, linestyle='-', label='change of attacked graph')
plt.legend(['change of clean graph', 'change of attacked graph'])
plt.show()
