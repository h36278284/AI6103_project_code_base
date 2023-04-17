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
from models.fast_gcn import FastGCN, SamplerFastGCN

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
seed = random.randint(1, 200)
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)


# Settings
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    cfg.dataset)

features = sp.identity(features.shape[0])  # featureless
# Some preprocessing
features = preprocess_features(features)
if cfg.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = FastGCN
else:
    raise ValueError('Invalid argument for model: ' + str(cfg.model))

# Define placeholders
t_features = torch.from_numpy(features)
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

t_support = []
for i in range(len(support)):
    t_support.append(torch.Tensor(support[i]))

sampler = SamplerFastGCN(None, t_features, nontuple_preprocess_adj(adj),
                         input_dim=features.shape[0])
model = model_func(input_dim=features.shape[0], support=t_support, sampler=sampler, num_classes=y_train.shape[1])
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


def train(train_ind, train_labels, batch_size):
    model.train()
    for batch_inds, batch_labels in get_batches(train_ind,
                                                train_labels,
                                                batch_size):
        sampled_feats, sampled_adjs, var_loss = model.sampling(
            batch_inds)
        optimizer.zero_grad()
        logits = model(sampled_feats)
        loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
        acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[
            1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()
        loss.backward()
        optimizer.step()
    # just return the train loss of the last train epoch
    return loss, acc


# Define model evaluation function
def evaluate(features, labels, mask):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        t_mask = torch.from_numpy(np.array(mask * 1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()

    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)


train_losss = []
train_accs = []
val_losses = []
val_losss = []
val_accs = []

# Train model
for epoch in range(cfg.epochs):

    t = time.time()

    # Forward pass
    loss, acc = train(np.arange(y_train.shape[0]),
                      y_train,
                      int(len(np.arange(y_train.shape[0]))/50))
    train_losss.append(loss.item())
    train_accs.append(acc)
    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
    val_losses.append(val_loss)
    val_losss.append(val_loss.item())
    val_accs.append(val_acc)
    print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}" \
              .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

    if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping + 1):-1]):
        print_log("Early stopping...")
        break

print_log("Optimization Finished!")

# plot
plot_loss_acc(train_losss, val_losss, train_accs, val_accs, '../TextFGCN with Sampling mr')
# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
print_log(
    "Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
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
