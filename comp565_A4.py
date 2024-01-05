# STUDENT NAME: Alina Tan
import pickle
import numpy as np
import pandas as pd

import scanpy as sc
import anndata
import random

import torch
from etm import ETM
from torch import optim
from torch.nn import functional as F

import os
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from seaborn import heatmap, lineplot, clustermap

random.seed(10)

# mouse pancreas single-cell dataset
# read in data and cell type labels
with open('data/MP.pickle', 'rb') as f:
    df = pickle.load(f)

with open('data/MP_genes.pickle', 'rb') as f:
    genes = pickle.load(f)

df.set_index('Unnamed: 0', inplace=True)  # set first column (cell ID as the index column)
sample_id = pickle.load(open('data/cell_IDs.pkl', 'rb'))
df = df.loc[list(sample_id), :]

X = df[genes].values  # extract the N x M cells-by-genes matrix

sample_info = pd.read_csv('data/sample_info.csv')

mp_anndata = anndata.AnnData(X=X)

mp_anndata.obs['Celltype'] = sample_info['assigned_cluster'].values

N = X.shape[0]  # number of single-cell samples
K = 16  # number of topics
M = X.shape[1]  # number of genes


def evaluate_ari(cell_embed, adata):
    """
        This function is used to evaluate ARI using the lower-dimensional embedding
        cell_embed of the single-cell data
        :param cell_embed: a NxK single-cell embedding generated from NMF or scETM
        :param adata: single-cell AnnData data object (default to to mp_anndata)
        :return: ARI score of the clustering top_geness produced by Louvain
    """
    adata.obsm['cell_embed'] = cell_embed
    sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=30)
    sc.tl.louvain(adata, resolution=0.15)
    ari = adjusted_rand_score(adata.obs['Celltype'], adata.obs['louvain'])
    return ari
#random.seed(10)

######## Q1 NMF sum of squared error ########
W_init = np.random.random((M, K))
H_init = np.random.random((K, N))


# Complete this function: non-negative matrix factorization
def nmf_sse(X, W, H, adata=mp_anndata, niter=100):
    """
        NMF with sum of squared error loss as the objective
        :param X: M x N input matrix
        :param W: M x K basis matrix
        :param H: K x N coefficient matrix
        :param adata: annotated X matrix with cluster labels for evaluating ARI (default to mouse pancreas data)
        :param niter: number of iterations to run
        :return:
            1. updated W and H that minimize sum of squared error ||X - WH||^2_F s.t. W,H>=0
            2. niter-by-3 ndarray with iteration index, SSE, and ARI as the 3 columns
    """
    perf = np.ndarray(shape=(niter, 3), dtype='float')
    
    # WRITE YOUR CODE HERE
    for i in range(niter):
        # Update H and W by the fradient descent
        H = H * ((np.matmul(W.T, X) / np.matmul(np.matmul(W.T, W), H)))
        W = W * ((np.matmul(X, H.T) / np.matmul(np.matmul(W, H), H.T)))
        
        # Calculate MSE and ARI
        mse = np.sum(np.square(X - (np.matmul(W, H))))/(N*M)
        ari = evaluate_ari(H.T, mp_anndata)
        perf[i] = [i, mse, ari]  # i is the index iteration index
    
    
    return W, H, perf


W_nmf_sse, H_nmf_sse, nmf_sse_perf = nmf_sse(X.T, W_init, H_init, niter=100)


######## Q2: write a function to monitor ARI and objective function ########
def monitor_perf(perf, objective, path=""):
    """
    :param perf: niter-by-3 ndarray with iteration index, objective function, and ARI as the 3 columns
    :param objective: 'SSE', 'Poisson', or 'NELBO'
    :param path: path to save the figure if not display to the screen
    :behaviour: display or save a 2-by-1 plot showing the progress of optimizing objective and ARI as
        a function of iterations
    """

    # WRITE YOUR CODE HERE
    fig, axs = plt.subplots(2)
    fig.suptitle(str(objective)+' and ARI of NMF at each iteration.')
    iter_x = perf[:, 0]
    MSE_y= perf[:, 1]
    ARI_y=perf[:, 2]
    axs[0].plot(iter_x, MSE_y)
    axs[0].set_ylabel(objective)
    axs[1].plot(iter_x, ARI_y)
    axs[1].set_ylabel('ARI')
    axs[1].set_xlabel('Iteration')
    plt.savefig(path)

monitor_perf(nmf_sse_perf, "MSE", 'figures/nmf_sse.eps')


######## Q3 NMF Poisson likelihood ########
# NMF with Poisson likelihood
# Complete this function
def nmf_psn(X, W, H, adata=mp_anndata, niter=100):
    """
        NMF with log Poisson likelihood as the objective
        :param X.T: M x N input matrix
        :param W: M x K basis matrix
        :param H: K x N coefficient matrix
        :param niter: number of iterations to run
        :return:
            1. updated W and H that minimize sum of squared error ||X - WH||^2_F s.t. W,H>=0
            2. niter-by-3 ndarray with iteration index, SSE, and ARI as the 3 columns
    """
    perf = np.ndarray(shape=(niter, 3), dtype='float')

    # WRITE YOUR CODE HERE 

    W =np.where(W > 0, W, 1e-16)
    H =np.where(H > 0, H, 1e-16)
    #update w and h to minimize see(p26 lec13)
    for i in range(niter):
        ones = np.ones((M,N))
        w_term2= (np.matmul((X/np.matmul(W,H)),H.T)) / (np.matmul(ones,H.T))
        W = W*w_term2
        
        h_term2 = (np.matmul(W.T, X/np.matmul(W,H))) / (np.matmul(W.T,ones))
        H = H*h_term2

        # calculate average log poisson likelihood and ari
        W =np.where(W > 0, W, 1e-16)
        H =np.where(H > 0, H, 1e-16)


        poisson_likelihood = np.sum(X * np.log(np.matmul(W, H)) - np.matmul(W, H)) / (N * M)
        ari = evaluate_ari(H.T, mp_anndata)
        #print("iter ", i, poisson_likelihood, ari)
        perf[i] = [i, poisson_likelihood, ari]
 
    return W, H, perf


W_nmf_psn, H_nmf_psn, nmf_psn_perf = nmf_psn(X.T, W_init, H_init, niter=100)

monitor_perf(nmf_psn_perf, "Poisson", 'figures/nmf_psn.eps')

# compare NMF-SSE and NMF-Poisson
fig, ax = plt.subplots()
nmf_sse_perf_df = pd.DataFrame(data=nmf_sse_perf, columns=['Iter', "SSE", 'ARI'])
nmf_psn_perf_df = pd.DataFrame(data=nmf_psn_perf, columns=['Iter', "Poisson", 'ARI'])
ax.plot(nmf_sse_perf_df["Iter"], nmf_sse_perf_df["ARI"], color='blue', label='NMF-SSE')
ax.plot(nmf_psn_perf_df["Iter"], nmf_psn_perf_df["ARI"], color='red', label='NMF-PSN')
ax.legend()
plt.xlabel("Iteration");
plt.ylabel("ARI")
plt.title("NMF-SSE vs NMF-Poisson in terms of ARI")
plt.savefig("figures/nmf_sse_vs_psn.eps")


######## Q4-Q8 VAE single-cell embedded topic model ########
X_tensor = torch.from_numpy(np.array(X, dtype="float32"))
sums = X_tensor.sum(1).unsqueeze(1)
X_tensor_normalized = X_tensor / sums

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ETM(num_topics=K,
            vocab_size=len(genes),
            t_hidden_size=256,
            rho_size=256,
            theta_act='relu',
            embeddings=None,
            train_embeddings=True,
            enc_drop=0.5).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1.2e-6)


# train the VAE for one epoch
def train_scETM_helper(model, X_tensor, X_tensor_normalized):
    # initialize the model and loss
    model.train()
    optimizer.zero_grad()
    model.zero_grad()

    # forward and backward pass
    nll, kl_theta = model(X_tensor, X_tensor_normalized)
    loss = nll + kl_theta
    loss.backward()  # backprop gradients w.r.t. negative ELBO

    # clip gradients to 2.0 if it gets too large
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    # update model to minimize negative ELBO
    optimizer.step()

    return torch.sum(loss).item()


# get sample encoding theta from the trained encoder network
def get_theta(model, input_x):
    model.eval()
    with torch.no_grad():
        q_theta = model.q_theta(input_x)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta


######## Q4 complete this function ########
def train_scETM(model, X_tensor, X_tensor_normalized, adata=mp_anndata, niter=1000):
    """
        :param model: the scETM model object
        :param X_tensor: NxM raw read count matrix X
        :param X_tensor_normalized: NxM normalized read count matrix X
        :param adata: annotated single-cell data object with ground-truth cell type information for evaluation
        :param niter: maximum number of epochs
        :return:
            1. model: trained scETM model object
            2. perf: niter-by-3 ndarray with iteration index, SSE, and ARI as the 3 columns
    """
    perf = np.ndarray(shape=(niter, 3), dtype='float')

    # WRITE YOUR CODE HERE
    for i in range(niter):
    
        nelbo = train_scETM_helper(model, X_tensor, X_tensor_normalized)
        theta_topic_mixture = get_theta(model, X_tensor_normalized)
        
        # Calculate ARI
        with torch.no_grad():
            ARI = evaluate_ari(theta_topic_mixture, mp_anndata)
        perf[i] = [i, nelbo, ARI]

    return model, perf


model, scetm_perf = train_scETM(model, X_tensor, X_tensor_normalized)

monitor_perf(scetm_perf, "NELBO", 'figures/scETM_train.eps')

######## Q5 Compare NMF-Poisson and scETM ########

# WRITE YOUR CODE HERE
W_nmf_psn, H_nmf_psn, nmf_psn_perf = nmf_psn(X.T, W_init, H_init, niter=1000)
monitor_perf(nmf_psn_perf, "Poisson", 'figures/nmf_psn_1000.eps')


fig, ax = plt.subplots()
nmf_psn_perf_df = pd.DataFrame(data=nmf_psn_perf, columns=['Iter', "Poisson", 'ARI'])
scetm_perf_df = pd.DataFrame(data=scetm_perf, columns=['Iter', "NELBO", 'ARI'])

ax.plot(nmf_psn_perf_df["Iter"], nmf_psn_perf_df["ARI"], color='red', label='NMF-PSN')
ax.plot(scetm_perf_df["Iter"], scetm_perf_df["ARI"], color='black', label='scETM')

ax.legend(loc='upper right')
plt.xlabel("Iteration");
plt.ylabel("ARI")
plt.title("NMF-Poisson versus scETM in terms of ARI")
plt.savefig("figures/nmf_psn_vs_scETM.eps")


######## Q6 plot t-SNE for NMF-Poisson and scETM ########

# WRITE YOUR CODE HERE

# Generate t-SNE plots for cell embeddings from scETM and NMF-Poisson

mp_anndata.obsm['cell_embed'] = H_nmf_psn.T
sc.tl.tsne(mp_anndata, use_rep='cell_embed')
sc.pl.tsne(mp_anndata, color='Celltype', save="_psn.pdf", title="NMF-Poisson on MP")

mp_anndata.obsm['cell_embed'] = get_theta(model, X_tensor_normalized)
sc.tl.tsne(mp_anndata, use_rep='cell_embed')
sc.pl.tsne(mp_anndata, color='Celltype', save="_scETM.pdf", title="scETM on MP")



######## Q7 plot cells by topics ########

# WRITE YOUR CODE HERE

# Convert Celltype column to object type
mp_anndata.obs["Celltype"] = mp_anndata.obs["Celltype"].astype('object')

# Create color map dictionary and row colors
cell_types = mp_anndata.obs["Celltype"].unique()
#colors = ["royalblue", "orange", "green", "red", "blueviolet", "brown", "hotpink", "darkkhaki", "turquoise", "lightsteelblue", "peachpuff", "lightgreen", "pink"]
colors =["blueviolet","hotpink", "darkkhaki","red","brown","peachpuff","lightgreen","lightsteelblue","royalblue","turquoise","orange","pink","green"]
cell_type_colors = dict(zip(cell_types, colors))
row_colors = mp_anndata.obs["Celltype"].map(cell_type_colors).to_numpy()

# Create clustermap
cluster_map = clustermap(mp_anndata.obsm['cell_embed'], row_colors=row_colors, cmap="Reds")

# Add legend
handles = [Patch(facecolor=cell_type_colors[name]) for name in cell_types]
plt.legend(handles, cell_types, title='Cell Types', bbox_to_anchor=(1,1), bbox_transform=plt.gcf().transFigure, loc='upper right')

plt.savefig('figures/cells_heatmap_scetm.png')



######## Q8 plot genes-by-topics ########

# WRITE YOUR CODE HERE
gene_topic = ETM.get_beta(model) #torch.Size([16, 4194])
gene_topic = gene_topic.detach().numpy()

gene_topic_df = pd.DataFrame(gene_topic.T) #(4194,16)
gene_topic_df.insert(0, 'genes', genes)

# Initialize a dataframe
top_genes= pd.DataFrame()
top_genes = pd.concat([gene_topic_df.nlargest(5, i) for i in range(K)], ignore_index=True)


top_genes = top_genes.fillna(0)
top_genes = top_genes.set_index('genes')

fig, ax = plt.subplots(figsize=(10, 30))
ax = heatmap(top_genes, vmax=0.2, cmap="Reds", linecolor="white")


plt.savefig('figures/topics_heatmap_scetm.png')


