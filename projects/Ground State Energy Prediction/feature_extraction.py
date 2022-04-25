import pandas as pd
import os
import numpy as np
import pandas as pd
import seaborn as sns
import json
from scipy.spatial.distance import pdist, squareform
import glob
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


"""## Function: Reading in the json (input) file, extracting features and saving results as csv """


def json_to_csv(json_file_str):
    # Read JSON data (only one file for example)
    with open(json_file_str) as f:
        data = json.load(f)

    # %% [markdown]
    # Now, we will construct some basic features based on Coulomb matrices defined as:
    #
    # $$
    # C_{IJ} = \frac{Z_I  Z_J}{\vert R_I - R_J \vert}, \quad  ({\rm I \neq J}) \qquad
    # C_{IJ} = Z_I^{2.4}, \quad (I=J)
    # $$
    #
    # After these Columb matrices are constructed, we will unroll then into feature vectors for each molecule. Since these matrices are symmetric, the feature vectors will contain only the upper triangular part.
    #
    # First we need the atomic numbers (Z) so, below is a small dictionary that stores the atomic numbers and masses for the elements in this data set

    # %% [code] {"jupyter":{"outputs_hidden":true}}
    periodic_table = {'H': [1, 1.0079],
                      'C': [6, 12.0107],
                      'N': [7, 14.0067],
                      'O': [8, 15.9994],
                      'S': [16, 32.065],
                      'F': [9, 18.9984],
                      'Si': [14, 28.0855],
                      'P': [15, 30.9738],
                      'Cl': [17, 35.453],
                      'Br': [35, 79.904],
                      'I': [53, 126.9045]}

    # %% [markdown]
    # Let's also keep the maximum number of atoms to 50. This can be changed later. Having a maximum of 50 atoms mean that the Coulomb matrices will be 50x50, and the feature vectors will be (50x51)/2 dimensional. For molecules that has less than 50 atoms, the Coulomb matrices are padded with zeros.
    #
    # Using the formula above, let's construct them below:

    # %% [code] {"jupyter":{"outputs_hidden":true}}
    # Maximum number of atoms and the number of molecules
    natMax = 50
    nMolecules = len(data)

    # Initiate arrays to store data
    data_CM = np.zeros((nMolecules, natMax*(natMax+1)//2), dtype=float)
    data_ids = np.zeros(nMolecules, dtype=int)
    data_multipoles = np.zeros((nMolecules, 14), dtype=float)
    data_mmff94 = np.zeros(nMolecules, dtype=float)

    # Loop over molecules and save
    ind = 0
    for molecule in data:

        # Check size, do not store molecules which has more atoms than natMax
        natoms = len(molecule['atoms'])
        if natoms > natMax:
            continue

        # Read energy, shape multipoles and Id
        data_mmff94[ind] = molecule['En']
        data_multipoles[ind, :] = molecule['shapeM']
        data_ids[ind] = molecule['id']

        # Initiate full CM padded with zeroes
        full_CM = np.zeros((natMax, natMax))
        full_Z = np.zeros(natMax)

        # Atoms: types and positions
        pos = []
        Z = []
        for i, at in enumerate(molecule['atoms']):
            Z.append(periodic_table[at['type']][0])
            pos.append(at['xyz'])

        pos = np.array(pos, dtype=float)
        Z = np.array(Z, dtype=float)

        # Construct Coulomb Matrices
        tiny = 1e-20    # A small constant to avoid division by 0
        dm = pdist(pos)  # Pairwise distances

        # Coulomb matrix
        coulomb_matrix = np.outer(Z, Z) / (squareform(dm) + tiny)
        full_CM[0:natoms, 0:natoms] = coulomb_matrix
        full_Z[0:natoms] = Z

        # Coulomb vector (upper triangular part)
        iu = np.triu_indices(natMax, k=1)  # No diagonal k=1
        coulomb_vector = full_CM[iu]

        # Sort elements by decreasing order
        shuffle = np.argsort(-coulomb_vector)
        coulomb_vector = coulomb_vector[shuffle]  # Unroll into vrctor

        # Construct feature vector
        coulomb_matrix = squareform(coulomb_vector)
        assert np.trace(coulomb_matrix) == 0, "Wrong Coulomb Matrix!"

        # Add diagonal terms
        # Add the diagonal terms
        coulomb_matrix += 0.5*np.power(full_Z, 2.4)*np.eye(natMax)
        # Upper diagonal
        iu = np.triu_indices(natMax)
        # Unroll into vector
        feature_vector = coulomb_matrix[iu]
        assert feature_vector.shape[0] == natMax * \
            (natMax+1)//2, "Wrong feature dimensions"

        # Save data
        data_CM[ind] = feature_vector

        # Iterate
        ind += 1

    # %% [markdown]
    # Now, we can convert into a data frame and save into file

    # %% [code] {"jupyter":{"outputs_hidden":true}}
    # Now save as pandas frame
    # Stack CM and multipole features
    dat = np.column_stack((data_CM, data_multipoles))
    df = pd.DataFrame(dat)

    # Column names
    numfeats = np.shape(dat)[1]
    cols = [x for x in range(1, numfeats+1, 1)]
    col_names = list(map(lambda x: 'f'+str(x), cols))
    df.columns = col_names

    # Add Energy and id
    df.insert(0, 'pubchem_id', data_ids)
    df['En'] = data_mmff94

    # /content/drive/MyDrive/CAPSTONE_TEAM_OVERFIT/NoteBooks/data/jsons/pubChem_p_00000001_00025000.json
    json_specific = json_file_str[74:93]

    # Save
    path = "data/csvs/molecules_" + \
        json_specific + ".csv"
    print(path)
    df.to_csv(path)


"""## Generating features in csv format"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# json_folder = '/content/drive/MyDrive/CAPSTONE_TEAM_OVERFIT/NoteBooks/data/jsons'
#
#
# for filename in os.listdir(json_folder):
#   json_to_csv(json_folder + "/" + filename)

"""## Concatenating the csv file into one single master file for ML model training"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # setting the path for joining multiple files
# files = os.path.join("/content/drive/MyDrive/Group 2 - Overfit/NoteBooks/data/csvs","molecules_p_*.csv")
#
# # list of merged files returned
# files = glob.glob(files)
# files.pop(len(files)-1)
# test = files[len(files)-1]
#
# print(test)
# # joining files with concat and read_csv
# df = pd.concat(map(pd.read_csv, files), ignore_index=True)
# #df.to_csv("/content/drive/MyDrive/Group 2 - Overfit/NoteBooks/data/csvs/molecules_p_00000001_00225000_train_molecules.csv", index=False)

master_file = pd.read_csv(
    "./data/csvs/molecules_p_00000001_00025000.csv")

print(master_file.shape)
master_file = master_file[master_file['pubchem_id'] != 0]
print(master_file.shape)

master_file['En'].plot.box()

sns.distplot(master_file['En'], kde=True, color="g")
plt.xlabel('Atomization Energy')
plt.ylabel('Frequency')
plt.title('Atomization Energy Distribution')

count_NA = master_file.isna().sum().sum()
print('There are %i null entries detected' % count_NA)
if count_NA != 0:
    print('Cleaning data ----')
    master_file.dropna()
    count_NA = master_file.isna().sum().sum()
    print('There are now $i null entries' % count_NA)

label = master_file.iloc[:, -1].to_numpy()  # label is our last column
features_df = master_file.drop(['Unnamed: 0', 'pubchem_id', 'En'], axis=1)
features = features_df.to_numpy()

# currently having 90% for training 10% for testing
x_train, x_test, y_train, y_test = train_test_split(
    features, label, test_size=0.01, shuffle=True)

decision_tree = DecisionTreeClassifier()
# score = np.zeros(5)
kf = KFold(n_splits=5, shuffle=True)
print(kf.get_n_splits(master_file))
kf

X = features_df.to_numpy()
Y = label
# Testing purposes, printing shapes
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)

"""# PCA"""

# currently having 80% for training 20% for testing
x_train, x_test, y_train, y_test = train_test_split(
    features, label, test_size=0.01, shuffle=True)

x_train_scaled = StandardScaler().fit_transform(x_train)

x_train_scaled

# pca = PCA(n_components=2)
pca = PCA(0.96)
pc = pca.fit_transform(x_train_scaled)
np.array(pc)

pca_df = pd.DataFrame(data=pc, columns=[
                      'pc 1', 'pc 2', 'pc 3', 'pc 4', 'pc 5', 'pc 6', 'pc 7', 'pc 8', 'pc 9', 'pc 10'])

pca_final_df = pd.concat([pca_df, master_file['En']], axis=1)
pca_final_df

plt.scatter(pca_df['pc 1'], pca_df['pc 2'])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

plt.scatter(pca_df['pc 1'].iloc[:500], pca_df['pc 2'].iloc[:500])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Add the target column to the dataframe
pca_final_df = pd.concat([pca_df, master_file['En']], axis=1)

plt.scatter(pca_final_df['pc 1'], pca_final_df['En'])
plt.xlabel("Principal Component 1")
plt.ylabel("Label")
plt.show()

plt.scatter(pca_final_df['pc 1'].iloc[:500], pca_final_df['En'].iloc[:500])
plt.xlabel("Principal Component 1")
plt.ylabel("Label")
plt.show()

variance = [0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
components = []
map = {}
for var in variance:
    pca2 = PCA(var)
    pc2 = pca2.fit(x_train_scaled)
    num_components = pca2.n_components_
    map[var] = num_components
    components.append(num_components)
map

plt.scatter(components, variance)
plt.xlabel("# of Components")
plt.ylabel("Variance")
plt.show()

x_train_test = pc2.transform(x_train_scaled)
x_train_test.shape
