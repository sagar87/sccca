import numpy as np
import pandas as pd
import torch
from models import scCCACore, scPCACore
from scipy.sparse import issparse
from tp import DEFAULT


def scPCA(
    adata,
    num_factors=10,
    batch_params=True,
    layers_key="counts",
    batch_key=None,
    device=torch.device("cuda"),
    svi_kwargs=DEFAULT,
):

    if layers_key is None:
        X = adata.X
    else:
        X = adata.layers[layers_key].toarray()

    if issparse(X):
        X = torch.tensor(X.toarray(), device=device)

    if batch_key is None:
        batch = None
    else:
        batch = pd.get_dummies(adata.obs[batch_key]).values.astype(np.float32)

    return scPCACore(X, batch, num_factors, batch_params=batch_params, device=device, svi_kwargs=DEFAULT)


def scCCA(
    adata,
    num_factors=10,
    batch_params=True,
    layers_key="counts",
    protein_obsm_key="protein_expression",
    batch_key=None,
    device=torch.device("cuda"),
    svi_kwargs=DEFAULT,
):

    if layers_key is None:
        X = adata.X
    else:
        X = adata.layers[layers_key].toarray()

    if issparse(X):
        X = torch.tensor(X.toarray(), device=device)

    if batch_key is None:
        batch = None
    else:
        batch = pd.get_dummies(adata.obs[batch_key]).values.astype(np.float32)

    if isinstance(adata.obsm[protein_obsm_key], pd.DataFrame):
        Y = adata.obsm[protein_obsm_key].values
    else:
        Y = adata.obsm[protein_obsm_key]

    return scPCACore(X, Y, batch, num_factors, batch_params=batch_params, device=device, svi_kwargs=DEFAULT)
