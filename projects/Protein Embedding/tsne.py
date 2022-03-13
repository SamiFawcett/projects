from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# t-SNE t-Distributed Stocastic Neighbor Embedding
tSne = TSNE(learning_rate=50)

X = []
embeddings_dir = '..\..\data\embeddings'
with open(embeddings_dir + '\prot_embeddings_mavg.vec') as f:
    vocab_size, embedding_dim = f.readline().split(' ')
    for line in f.readlines():
        line = line.strip()
        codon = line[0:3]
        embedding = line[4:]
        embedding_coord_str = embedding.split(' ')
        embedding_coord_float = []
        for coord in embedding_coord_str:
            embedding_coord_float.append(float(coord))
        X.append(embedding_coord_float)

tsne_features = tSne.fit_transform(X)
x = tsne_features[:, 0]
y = tsne_features[:, 1]
df = pd.DataFrame(dtype=np.float32)
df['x'] = x
df['y'] = y

sns.scatterplot(x="x", y="y", data=df)

plt.show()
