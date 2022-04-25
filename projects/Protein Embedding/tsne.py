from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# t-SNE t-Distributed Stocastic Neighbor Embedding
tSne = TSNE(n_components=2, perplexity=1, learning_rate="auto", n_iter=270,
            n_iter_without_progress=150, random_state=0, init='pca')

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

X = np.array(X)
tsne_features = tSne.fit_transform(X)
x = tsne_features[:, 0]
y = tsne_features[:, 1]
#z = tsne_features[:, 2]

df = pd.DataFrame(dtype=np.float32)
df['x'] = x
df['y'] = y
#df['z'] = z

sns.scatterplot(x="x", y="y", data=df)
#ax = plt.axes(projection='3d')
#ax.scatter3D(x, y, z)
plt.show()
