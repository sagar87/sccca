import torch
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from .sccca.utils import  batch_mean

@pytest.fixture
def anndata():
    counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
    adata = ad.AnnData(counts)
    return adata

def test_batch_mean():
    batch = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])
    X = torch.tensor([[3], [3], [3], [5], [5]])
    res = batch_mean(batch, X)
    assert res[0] == 3.

