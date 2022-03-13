import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataclasses import dataclass
from collections import defaultdict
import time
from tqdm import tqdm


@dataclass
class SequenceData():
    def __init__(self, filename, kmer):
        self.filename: str = filename
        self.sequences: list[Sequence] = []
        self.kmer = kmer
        self.totlen = 0
        with open(filename, 'r') as f:
            f.readline()  # skip first line

            for line in f.readlines():
                self.sequences.append(Sequence(line.strip(), self.kmer))
                self.totlen += len(line.strip())

        self.create_vocab()
        self.create_prob_dist()

    def __len__(self):
        return self.totlen

    def create_vocab(self):
        self.kmer_freq = defaultdict(lambda: 0)
        for seq in self.sequences:
            for (idx, kmer) in enumerate(iter(seq)):
                self.kmer_freq[kmer] += 1
        self.vocab = sorted(self.kmer_freq)
        self.kmer_to_index = {km: i for (i, km) in enumerate(self.vocab)}
        self.index_to_kmer = {i: km for (i, km) in enumerate(self.vocab)}
        self.idxs = np.arange(len(self.vocab))

    def create_prob_dist(self):
        self.prob_dist = np.array(list(self.kmer_freq.values())) / \
            np.sum(list(self.kmer_freq.values()))

        self.cumsum = np.cumsum(self.prob_dist)

    def __iter__(self):
        self.sequence_index = 0
        return self

    def __next__(self):
        if self.sequence_index < len(self.sequences):
            self.sequence_index += 1
            return self.sequences[self.sequence_index - 1]
        else:
            raise StopIteration


@dataclass
class Sequence():
    def __init__(self, seq, kmer=1):
        self.seq = seq
        self.kmer = kmer
        self.len = len(seq)
        self.amino_acids = {aa: 1 for (idx, aa) in enumerate(seq)}
        self.kmers = [self.seq[i: i+self.kmer]
                      for i in range(len(self.seq) - self.kmer)]

    def __iter__(self):
        self.kmer_index = 0
        return self

    def __next__(self):
        if self.kmer_index < len(self.kmers):
            self.kmer_index += 1
            return self.kmers[self.kmer_index - 1]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.kmers)

    def get(self, i):
        if i < len(self.kmers) and i >= 0:
            return self.kmers[i]
        else:
            raise IndexError


class PosNegSampler(torch.utils.data.IterableDataset):
    def __init__(self, sequenceData, block_size, window_size, neg_samples):
        super(PosNegSampler, self).__init__()
        self.sequenceData = sequenceData  # SequenceData object
        self.window_size = window_size  # number of kmer's in window
        self.neg_samples = neg_samples  # number of negative samples per positive sample
        self.block_size = block_size  # number of pos/neg pairs per block

    def __len__(self):
        self.size = 2 * self.window_size * \
            len(self.sequenceData) - self.sequenceData.kmer + 1
        self.size *= self.neg_samples + 1
        self.size /= self.block_size
        self.size = int(self.size)
        return self.size

    def __iter__(self):
        """
        returns one iterable block of target context pairs with pos/neg labeling
        """
        for (i, (T, C)) in enumerate(self.pair_generator()):
            posT = np.array(T)
            posC = np.array(C)
            posL = np.ones(len(T))
            yield (posT, posC, posL)
            for j in range(self.neg_samples):
                negT = np.array(T)
                negC = self.getNegWords(len(T))
                negL = np.zeros(len(T))
                yield (negT, negC, negL)

    def getNegWords(self, numOfNegWords):
        return np.searchsorted(
            self.sequenceData.cumsum, np.random.random(numOfNegWords))

    def pair_generator(self):
        T = []
        C = []
        for seq_idx, seq in enumerate(iter(self.sequenceData)):
            for kmer_idx, kmer in enumerate(iter(seq)):

                # create window
                start_idx = max(0, kmer_idx - self.window_size)
                end_idx = min(len(seq), kmer_idx + self.window_size + 1)
                for window_idx in range(start_idx, end_idx):
                    if kmer_idx != window_idx:
                        T.append(self.sequenceData.kmer_to_index[kmer])
                        C.append(
                            self.sequenceData.kmer_to_index[seq.get(window_idx)])

                if len(T) >= self.block_size:
                    yield (T, C)
                    T, C = [], []

        # any remaining pairs must be returned
        # in the case where there are not enough pairs to fill up a block
        # just return them if we reached the end
        yield (T, C)


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(NeuralNetwork, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        logits = self.model_sequence(x)
        return logits


class ProteinEmbedding(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super(ProteinEmbedding, self).__init__()
        self.embedding_dim = embed_dim
        self.T = nn.Embedding(vocab_size, embed_dim)
        self.C = nn.Embedding(vocab_size, embed_dim)

    def forward(self, t_kmer, c_kmer, label):
        t_embed = self.T(t_kmer)
        c_embed = self.C(c_kmer)
        out = torch.sum(t_embed * c_embed, dim=1)
        return out

    def save_embeddings(self, filename, idx_to_word, mode):
        if mode == 'avg':
            # average the T and C matrices
            W = (net.T.weight.cpu().data.numpy() +
                 net.C.weight.cpu().data.numpy())/2.
        elif mode == 'target':
            # W is T matrices
            W = net.T.weight.cpu().data.numpy()
        elif mode == 'context':
            # W is C matrices
            W = net.C.weight.cpu().data.numpy()
        else:
            sys.exit(f'{mode} not supported')

        with open(filename, "w") as f:
            f.write("%d %d\n" % (len(idx_to_word), self.embedding_dim))
            for wid, w in idx_to_word.items():
                e = ' '.join(map(lambda x: str(x), W[wid]))
                f.write("%s %s\n" % (w, e))


device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# data arguments
kmer = 3
block_size = 256*256
window_size = 10
neg_samples = 2
embedding_dim = 24
embeddings_outfile = '..\data\embeddings.txt'

# training arguments
batch_size = 1
learning_rate = 0.01
epochs = 5

# read and create data
S = SequenceData('..\data\sample_uniprot.txt', 3)
PNS = PosNegSampler(S, block_size, window_size, neg_samples)
V = len(PNS.sequenceData.vocab)
print(f"vocab size: {V}")

# load data into dataloader object
train_data = DataLoader(PNS, batch_size=batch_size)


# create embedding model
net = ProteinEmbedding(embed_dim=embedding_dim, vocab_size=V)
net.to(device)

# adam optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

start_t = time.time()
for epoch in range(epochs):
    for batch_idx, (targets, contexts, labels) in enumerate(tqdm(train_data, total=len(PNS))):
        targets = targets.flatten().to(device)
        contexts = contexts.flatten().to(device)
        labels = labels.flatten().to(device)

        net.zero_grad()
        predictions = net(targets, contexts, labels)
        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()
end_t = time.time()

print("training time: ", end_t - start_t)

for mode in ['avg']:
    outfile = 'prot_embeddings_m%s' % (mode)
    net.save_embeddings(outfile, S.index_to_kmer, mode)

"""
# classification model parameters
in_dim = 128
hidden_dim = 768
out_dim = 1
model = NeuralNetwork(in_dim, hidden_dim, out_dim).to(device)
"""
