import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import re
# 1.Data Preparation
def load_data(path="../Dataset/VNESEcorpus.txt"):
    with open(path, 'r') as file:
        data = file.read()
    return data

def build_corpus(data, end_sentence='<END>', start_sentence='<START>'):
    corpus = []
    sentences = data.split(".")
    for sentence in sentences:
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        corpus.append(pattern.findall(sentence.lower()))
    return corpus

def build_vocab(corpus):
    vocab = []
    reverse_vocab = {}
    for sentence in corpus:
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
    vocab.sort()

    return vocab

data = "We are about to study the idea of a computational process.\
    Computational processes are abstract beings that inhabit computers.\
    As they evolve, processes manipulate other abstract things called data.\
    The evolution of a process is directed by a pattern of rules\
    called a program. People create programs to direct processes. In effect,\
    we conjure the spirits of the computer with our spells <3"

tokenized_corpus = build_corpus(data)
vocabulary = build_vocab(tokenized_corpus)
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

print(tokenized_corpus)
print(word2idx)
vocabulary_size = len(vocabulary)

window_size = 2

def sample_training(corpus, word2idx, window_size=2):
    training_samples = []
    for sentence in corpus:
        len_sentence = len(sentence)
        for t in range(len_sentence):
            for m in range(-window_size, window_size+1, 1):
                if (m != 0) and (t+m >= 0) and (t+m < len_sentence):
                    pair = [word2idx[sentence[t]], word2idx[sentence[t+m]]]
                    training_samples.append(pair)
    return training_samples

def to_one_hot(index_of_word, vocab_size):
    tmp = np.zeros(vocab_size)
    tmp[index_of_word] = 1
    return tmp

traning_samples = sample_training(tokenized_corpus, word2idx)
print(traning_samples)

class W2V(nn.Module):
    def __init__(self, vocabulary_size, embedding_size=100):
        super(W2V, self).__init__()

        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.l1 = nn.Linear(embedding_size, vocabulary_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        out = self.l1(lookup_embeds)
        out = F.log_softmax(out)
        return out

loss_func = nn.CrossEntropyLoss()
model = W2V(vocabulary_size, embedding_size=10)
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    total_loss = 0
    for pair in traning_samples:
        center_word = torch.LongTensor(to_one_hot(pair[0], vocabulary_size))
        model.zero_grad()
        log_probs = model(center_word)
        loss = loss_func(log_probs, Variable(torch.LongTensor(to_one_hot(pair[1], vocabulary_size))))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    if epoch % 10 == 0:
        print('Epoch {}'.format(epoch), total_loss)

torch.save(model, 'w2v_model.pt')