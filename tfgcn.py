from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os

import torch
import torch.nn as nn

import numpy as np

from utils.utils import *
from models.gcn import GCN
from models.mlp import MLP
from models.fgcn import FastGCN, SamplerFastGCN

from config import CONFIG

cfg = CONFIG()

if len(sys.argv) != 2:
    sys.exit("Use: python train.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")
cfg.dataset = dataset

# Set random seed
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    cfg.dataset)

features = sp.identity(features.shape[0])
train_index = np.where(train_mask)[0]
train_adj = adj[train_index, :][:, train_index]
y_train = y_train[train_index]
num_train = train_adj.shape[0]
features = nontuple_preprocess_features(features).todense()
train_features = features[train_index]
train_adj = nontuple_preprocess_adj(train_adj)
adj = nontuple_preprocess_adj(adj)
train_adj2 = [adj, adj[train_index, :]]
train_adj2 = [sparse_mx_to_torch_sparse_tensor(cur_adj)
              for cur_adj in train_adj2]

layer_sizes = [256, 256]
input_dim = features.shape[1]
train_nums = train_adj.shape[0]
nclass = y_train.shape[1]

features = torch.FloatTensor(features)
train_features = torch.FloatTensor(train_features)
train_labels = torch.LongTensor(y_train).max(1)[1]

val_adj = [adj, adj[np.where(val_mask)[0], :]]
val_feats = features
val_labels = y_val[np.where(val_mask)[0]]
val_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj)
           for cur_adj in val_adj]
val_labels = torch.LongTensor(val_labels).max(1)[1]

test_adj = [adj, adj[np.where(test_mask)[0], :]]
test_feats = features
test_labels = y_test[np.where(test_mask)[0]]
test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj)
            for cur_adj in test_adj]
test_labels = torch.LongTensor(test_labels).max(1)[1]

model_func = FastGCN
sampler = SamplerFastGCN(None, features, adj,
                         input_dim=input_dim,
                         layer_sizes=layer_sizes)
model = model_func(nfeat=features.shape[1],
                   nhid=16,
                   dropout=0.0,
                   sampler=sampler,
                   nclass=nclass)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)


def train(train_ind, train_labels, batch_size):
    model.train()
    for batch_inds, batch_labels in get_batches(train_ind, train_labels, batch_size):
        sampled_feats, sampled_adjs, var_loss = model.sampling(batch_inds)
        optimizer.zero_grad()
        logits = model(sampled_feats, sampled_adjs)
        loss = criterion(logits, batch_labels)
        acc = accuracy(logits, batch_labels)
        loss.backward()
        optimizer.step()
    return loss, acc


# Define model evaluation function
def evaluate(features, adj, labels):
    t_test = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        loss = criterion(logits, labels)
        pred = torch.max(logits, 1)[1]
        acc = accuracy(logits, labels)
    return loss, acc, pred.numpy(), (time.time() - t_test)


def test(features, adj, labels):
    model.train()
    logits = model(features, adj)
    loss = criterion(logits, labels)
    acc = accuracy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc


train_losss = []
train_accs = []
val_losses = []
val_losss = []
val_accs = []
# Train model
for epoch in range(cfg.epochs):
    t = time.time()
    # Forward pass
    # loss, acc = test(features, train_adj2, train_labels)
    loss, acc = train(np.arange(train_nums),
                      train_labels,
                      int(len(np.arange(train_nums))/100))
    train_losss.append(loss.item())
    train_accs.append(acc.item())
    # Validation
    val_loss, val_acc, pred, duration = evaluate(val_feats, val_adj, val_labels)
    val_losses.append(val_loss)
    val_losss.append(val_loss.item())
    val_accs.append(val_acc.item())

    print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}" \
              .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

    if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping + 1):-1]):
        print_log("Early stopping...")
        break

print_log("Optimization Finished!")

# plot
plot_loss_acc(train_losss, val_losss, train_accs, val_accs, '../FastGCN mr')
# Testing
test_loss, test_acc, pred, test_duration = evaluate(test_feats, test_adj, test_labels)
print_log(
    "Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

labels = torch.from_numpy(y_test[np.where(test_mask)[0]]).numpy()
test_pred = pred
test_labels = []
for i in range(len(labels)):
    test_labels.append(np.argmax(labels[i]))

print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(test_labels, test_pred, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# doc and word embeddings
tmp = model.layer1.embedding.numpy()
word_embeddings = tmp[train_size: adj.shape[0] - test_size]
train_doc_embeddings = tmp[:train_size]  # include val docs
test_doc_embeddings = tmp[adj.shape[0] - test_size:]

print_log('Embeddings:')
print_log('\rWord_embeddings:' + str(len(word_embeddings)))
print_log('\rTrain_doc_embeddings:' + str(len(train_doc_embeddings)))
print_log('\rTest_doc_embeddings:' + str(len(test_doc_embeddings)))
print_log('\rWord_embeddings:')
print(word_embeddings)

with open('./data/corpus/' + dataset + '_vocab.txt', 'r') as f:
    words = f.readlines()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
with open('./data/' + dataset + '_word_vectors.txt', 'w') as f:
    f.write(word_embeddings_str)

doc_vectors = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
with open('./data/' + dataset + '_doc_vectors.txt', 'w') as f:
    f.write(doc_embeddings_str)
