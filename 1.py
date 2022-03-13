import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from collections import defaultdict


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
        pass

    def __next__(self):
        pass


device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

kmer = 3
block_size = 256*256
window_size = 10
neg_samples = 2

S = SequenceData('./small_uniprot.txt', 3)
PNS = PosNegSampler(S, block_size, window_size, neg_samples)

print(len(PNS))
# model parameters
in_dim = 128
hidden_dim = 768
out_dim = 1

model = NeuralNetwork(in_dim, hidden_dim, out_dim).to(device)
