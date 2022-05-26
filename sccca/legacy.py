import sys

import numpy as np
import pyro
import pyro.distributions as dist
import torch

sys.path.insert(0, "/home/voehring/voehring/opt/pyromaniac/pyromaniac/")
from handler import SVIModel
from tp import DEFAULT
from utils import batch_mean, batch_var, log_library


class scPCACore(SVIModel):
    """
    The scPCA model.

    Parameter:
    ----------
    X: a single cell num_cells x num_genes RNA count matrix
    batch: a batch num_cells x num_batches (design/dummie) matrix or None
    num_factors: number of factors to extract
    """

    def __init__(
        self,
        X: np.ndarray,
        batch: np.ndarray = None,
        num_factors: int = 10,
        batch_params: bool = False,
        device: torch.device = torch.device("cuda"),
        svi_kwargs: dict = DEFAULT,
    ):
        super().__init__(**svi_kwargs)
        self.device = device
        self.X = torch.tensor(X, device=self.device)
        self.num_cells, self.num_genes = self.X.shape

        if batch is None:
            self.batch = torch.ones((self.num_cells, 1), device=self.device)
        else:
            self.batch = torch.tensor(batch, device=self.device)

        self.num_batches = self.batch.shape[1]
        self.num_factors = num_factors
        self.rna_log_library = log_library(self.X)
        self.rna_log_mean = batch_mean(self.rna_log_library, self.batch)  # self.rna_library.mean()
        self.rna_log_var = batch_var(self.rna_log_library, self.batch)
        self.batch_params = batch_params

    @property
    def _latent_variables(self):
        return [
            "z",
            "β_rna",
            "α_rna",
            "W_fac",
            "W_add",
            "W",
        ]

    def model(self, *args, **kwargs):
        gene_plate = pyro.plate("genes", self.num_genes)
        batch_plate = pyro.plate("batches", self.num_batches)
        cell_plate = pyro.plate("cells", self.num_cells)
        factor_plate = pyro.plate("factor", self.num_factors)

        # factor matrices
        with factor_plate:
            W_fac = pyro.sample(
                "W_fac",
                dist.Normal(
                    torch.zeros(self.num_genes, device=self.device), torch.ones(self.num_genes, device=self.device)
                ).to_event(1),
            )

        if self.num_batches > 1:
            if self.batch_params:
                W_add = pyro.param("W_add", torch.zeros((self.num_batches, self.num_genes), device=self.device))
            else:
                with batch_plate:
                    W_add = pyro.sample(
                        "W_add",
                        dist.Normal(
                            torch.zeros(self.num_genes, device=self.device),
                            torch.ones(self.num_genes, device=self.device),
                        ).to_event(1),
                    )

            W = pyro.deterministic("W", W_fac.unsqueeze(0) + W_add.unsqueeze(1))  # * torch.exp(W_add.unsqueeze(1))
        else:
            W = W_fac

        # hypyer prior gene wise
        β_rna = pyro.sample(
            "β_rna", dist.Gamma(torch.tensor(9.0, device=self.device), torch.tensor(3.0, device=self.device))
        )

        with gene_plate:
            σ_rna = pyro.sample("σ_rna", dist.Exponential(β_rna))
            α_rna = pyro.deterministic("α_rna", (1 / σ_rna).reshape(1, -1))

        with cell_plate:
            z = pyro.sample(
                "z",
                dist.Normal(
                    torch.zeros(self.num_factors, device=self.device),
                    0.1 * torch.ones(self.num_factors, device=self.device),
                ).to_event(1),
            )
            if self.num_batches > 1:
                μ_rna = pyro.deterministic(
                    "μ_rna",
                    torch.exp(
                        self.rna_log_library + (self.batch.T.unsqueeze(-1) * (torch.einsum("cf,bfp->bcp", z, W))).sum(0)
                    ),
                )
            else:
                μ_rna = pyro.deterministic("μ_rna", torch.exp(self.rna_log_library + torch.einsum("cf,fp->cp", z, W)))

            pyro.sample("rna", dist.GammaPoisson(α_rna, α_rna / μ_rna).to_event(1), obs=self.X)

    def guide(self, *args, **kwargs):

        W_fac_loc = pyro.param("W_fac_loc", torch.zeros((self.num_factors, self.num_genes), device=self.device))
        W_fac_scale = pyro.param(
            "W_fac_scale",
            0.1 * torch.ones((self.num_factors, self.num_genes), device=self.device),
            constraint=dist.constraints.greater_than(0.01),
        )

        with pyro.plate("factor", self.num_factors):
            pyro.sample("W_fac", dist.Normal(W_fac_loc, W_fac_scale).to_event(1))

        if self.num_batches > 1:
            if not self.batch_params:
                W_add_loc = pyro.param("W_add_loc", torch.zeros((self.num_batches, self.num_genes), device=self.device))
                W_add_scale = pyro.param(
                    "W_add_scale",
                    0.1 * torch.ones((self.num_batches, self.num_genes), device=self.device),
                    constraint=dist.constraints.greater_than(0.01),
                )

                with pyro.plate("batches", self.num_batches):
                    pyro.sample("W_add", dist.Normal(W_add_loc, W_add_scale).to_event(1))

        β_rna_loc = pyro.param("β_rna_loc", torch.tensor(0.0, device=self.device))
        β_rna_scale = pyro.param(
            "β_rna_rate", torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive
        )
        pyro.sample(
            "β_rna", dist.TransformedDistribution(dist.Normal(β_rna_loc, β_rna_scale), dist.transforms.ExpTransform())
        )

        σ_rna_loc = pyro.param("σ_rna_loc", torch.zeros(self.num_genes, device=self.device))
        σ_rna_scale = pyro.param(
            "σ_rna_scale", 0.1 * torch.ones(self.num_genes, device=self.device), constraint=dist.constraints.positive
        )

        with pyro.plate("genes", self.num_genes):
            pyro.sample(
                "σ_rna",
                dist.TransformedDistribution(dist.Normal(σ_rna_loc, σ_rna_scale), dist.transforms.ExpTransform()),
            )

        z_loc = pyro.param("z_loc", torch.zeros((self.num_cells, self.num_factors), device=self.device))
        z_scale = pyro.param(
            "z_scale",
            0.1 * torch.ones((self.num_cells, self.num_factors), device=self.device),
            constraint=dist.constraints.greater_than(0.01),
        )
        with pyro.plate("cells", self.num_cells):
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


class scPCA(SVIModel):
    def __init__(self, adata, num_factors=10, layers="counts", device=torch.device("cuda")):
        super().__init__(num_epochs=5000, num_samples=100, optimizer_kwargs={"lr": 0.01})
        self.adata = adata
        self.device = device

        if layers is not None:
            self.X = torch.tensor(adata.layers[layers].toarray(), device=device)

        self.num_factors = num_factors
        self.num_cells, self.num_genes = self.X.shape
        self.rna_library = torch.log(self.X.sum(1))
        self.log_mean = self.rna_library.mean()
        self.log_var = self.rna_library.var()

    @property
    def _latent_variables(self):
        return ["z", "s", "α"]

    def model(self, *args, **kwargs):
        # model W matrix as parameter
        W = pyro.param("W", torch.zeros((self.num_factors, self.num_genes), device=self.device))

        # gene wise dispersion parameter
        β0 = pyro.sample(
            "β",
            dist.Gamma(
                torch.tensor(9.0, device=self.device),
                torch.tensor(3.0, device=self.device),
            ),
        )

        with pyro.plate("genes", self.num_genes):
            σ = pyro.sample("σ", dist.Exponential(β0))
            α = pyro.deterministic("α", (1 / σ).reshape(1, -1))

        with pyro.plate("cells", self.num_cells):
            z = pyro.sample(
                "z",
                dist.Normal(
                    torch.zeros(self.num_factors, device=self.device),
                    torch.ones(self.num_factors, device=self.device),
                ).to_event(1),
            )
            s = pyro.sample("s", dist.LogNormal(self.log_mean, self.log_var))
            μ = pyro.deterministic("μ", s.reshape(-1, 1) * torch.exp(z @ W))
            pyro.sample("obs", dist.GammaPoisson(α, α / μ).to_event(1), obs=self.X)

    def guide(self, *args, **kwargs):
        β_loc = pyro.param("β_loc", torch.tensor(0.0, device=self.device))
        β_scale = pyro.param(
            "β_rate",
            torch.tensor(1.0, device=self.device),
            constraint=dist.constraints.positive,
        )
        pyro.sample(
            "β",
            dist.TransformedDistribution(dist.Normal(β_loc, β_scale), dist.transforms.ExpTransform()),
        )

        # σ_rate = pyro.param("σ_rate", torch.ones(num_genes, device=device), constraint=dist.constraints.positive)
        σ_loc = pyro.param("σ_loc", torch.zeros(self.num_genes, device=self.device))
        σ_scale = pyro.param(
            "σ_scale",
            0.1 * torch.ones(self.num_genes, device=self.device),
            constraint=dist.constraints.positive,
        )

        with pyro.plate("genes", self.num_genes):
            # pyro.sample("σ", dist.Exponential(σ_rate))
            pyro.sample(
                "σ",
                dist.TransformedDistribution(dist.Normal(σ_loc, σ_scale), dist.transforms.ExpTransform()),
            )

        z_loc = pyro.param(
            "z_loc",
            torch.zeros((self.num_cells, self.num_factors), device=self.device),
        )
        z_scale = pyro.param(
            "z_scale",
            0.1 * torch.ones((self.num_cells, self.num_factors), device=self.device),
            constraint=dist.constraints.positive,
        )
        s_loc = pyro.param(
            "s_loc",
            self.log_mean * torch.ones(self.num_cells, device=self.device),
        )
        s_scale = pyro.param(
            "s_scale",
            self.log_var * torch.ones(self.num_cells, device=self.device),
            constraint=dist.constraints.positive,
        )

        with pyro.plate("cells", self.num_cells):
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            pyro.sample("s", dist.LogNormal(s_loc, s_scale))

    def deterministic(self):
        self.mu = np.expand_dims(self.posterior["s"], -1) * np.exp(
            np.einsum("fg,scf->scg", self.params["W"], self.posterior["z"])
        )


class scCCAReparam(SVIModel):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch: np.ndarray = None,
        num_factors: int = 10,
        batch_params: bool = False,
        device=torch.device("cuda"),
        svi_kwargs=DEFAULT,
    ):
        super().__init__(**svi_kwargs)
        self.device = device
        self.X = torch.tensor(X, device=device)
        self.Y = torch.tensor(Y, device=device)

        self.num_factors = num_factors
        self.num_cells, self.num_genes = self.X.shape

        if batch is None:
            self.batch = torch.ones((self.num_cells, 1), device=self.device)
        else:
            self.batch = torch.tensor(batch, device=self.device)

        _, self.num_proteins = self.Y.shape
        self.num_batches = self.batch.shape[1]
        self.rna_log_library = torch.log(self.X.sum(1, keepdims=True))
        self.prot_log_library = torch.log(self.Y.sum(1, keepdims=True))
        self.rna_log_mean = batch_mean(self.rna_log_library, self.batch)  # self.rna_library.mean()
        self.rna_log_var = batch_var(self.rna_log_library, self.batch)
        self.prot_log_mean = batch_mean(self.prot_log_library, self.batch)
        self.prot_log_var = batch_var(self.prot_log_library, self.batch)

        self.batch_params = batch_params

    @property
    def _latent_variables(self):
        return [
            "z",
            "β_rna",
            "β_prot",
            "α_rna",
            "α_prot",
            "W_fac",
            "V_fac",
            "W_add",
            "V_add",
            "W",
            "V"
            # "μ_rna",
            # "μ_prot"
        ]

    def model(self, *args, **kwargs):

        gene_plate = pyro.plate("genes", self.num_genes)
        protein_plate = pyro.plate("proteins", self.num_proteins)
        batch_plate = pyro.plate("batches", self.num_batches)
        cell_plate = pyro.plate("cells", self.num_cells)
        # factor_plate = pyro.plate("factor", self.num_factors)

        # factor matrices
        with pyro.plate("factor", self.num_factors):
            W_fac = pyro.sample(
                "W_fac",
                dist.Normal(
                    torch.zeros(self.num_genes, device=self.device), torch.ones(self.num_genes, device=self.device)
                ).to_event(1),
            )
            V_fac = pyro.sample(
                "V_fac",
                dist.Normal(
                    torch.zeros(self.num_proteins, device=self.device),
                    torch.ones(self.num_proteins, device=self.device),
                ).to_event(1),
            )

        if self.num_batches > 1:
            if self.batch_params:
                W_add = pyro.param("W_add", torch.zeros((self.num_batches, self.num_genes), device=self.device))
                V_add = pyro.param("V_add", torch.zeros((self.num_batches, self.num_proteins), device=self.device))
            else:
                with batch_plate:
                    W_add = pyro.sample(
                        "W_add",
                        dist.Normal(
                            torch.zeros(self.num_genes, device=self.device),
                            torch.ones(self.num_genes, device=self.device),
                        ).to_event(1),
                    )
                    V_add = pyro.sample(
                        "V_add",
                        dist.Normal(
                            torch.zeros(self.num_proteins, device=self.device),
                            torch.ones(self.num_proteins, device=self.device),
                        ).to_event(1),
                    )

            # W_add = pyro.param("W_add", torch.zeros((self.num_batches, 1, self.num_genes), device=self.device))
            # print(W_add.unsqueeze(1).shape, W_fac.shape)
            # W_mul = pyro.param("W_mul", torch.randn((1, 1, self.num_genes), device=self.device))
            # W_bat = pyro.param("W_bat", torch.randn((self.num_batches, 1, 1), device=self.device))

            W = pyro.deterministic("W", W_fac.unsqueeze(0) + W_add.unsqueeze(1))  # * torch.exp(W_add.unsqueeze(1))
            V = pyro.deterministic("V", V_fac.unsqueeze(0) + V_add.unsqueeze(1))  # * torch.exp(V_add.unsqueeze(1))
            # W = W_raw # / torch.linalg.norm(W_raw, dim=-1, keepdims=True)
        else:
            W = W_fac
            V = V_fac

        # hypyer prior gene wise
        β_rna = pyro.sample(
            "β_rna", dist.Gamma(torch.tensor(9.0, device=self.device), torch.tensor(3.0, device=self.device))
        )
        β_prot = pyro.sample(
            "β_prot",
            dist.Gamma(
                torch.tensor(9.0, device=self.device),
                torch.tensor(3.0, device=self.device),
            ),
        )

        with gene_plate:
            σ_rna = pyro.sample("σ_rna", dist.Exponential(β_rna))
            α_rna = pyro.deterministic("α_rna", (1 / σ_rna).reshape(1, -1))

        with protein_plate:
            σ_prot = pyro.sample("σ_prot", dist.Exponential(β_prot))
            α_prot = pyro.deterministic("α_prot", (1 / σ_prot).reshape(1, -1))

        with cell_plate:
            z = pyro.sample(
                "z",
                dist.Normal(
                    torch.zeros(self.num_factors, device=self.device),
                    0.1 * torch.ones(self.num_factors, device=self.device),
                ).to_event(1),
            )
            if self.num_batches > 1:
                μ_rna = pyro.deterministic(
                    "μ_rna",
                    torch.exp(
                        self.rna_log_library + (self.batch.T.unsqueeze(-1) * (torch.einsum("cf,bfp->bcp", z, W))).sum(0)
                    ),
                )
            else:
                μ_rna = pyro.deterministic("μ_rna", torch.exp(self.rna_log_library + torch.einsum("cf,fp->cp", z, W)))

            pyro.sample("rna", dist.GammaPoisson(α_rna, α_rna / μ_rna).to_event(1), obs=self.X)

            if self.num_batches > 1:
                μ_prot = pyro.deterministic(
                    "μ_prot",
                    torch.exp(
                        self.prot_log_library
                        + (self.batch.T.unsqueeze(-1) * (torch.einsum("cf,bfp->bcp", z, V))).sum(0)
                    ),
                )
            else:
                μ_prot = pyro.deterministic("μ_prot", torch.exp(self.rna_log_library + torch.einsum("cf,fp->cp", z, V)))

            pyro.sample("prot", dist.GammaPoisson(α_prot, α_prot / μ_prot).to_event(1), obs=self.Y)

    def guide(self, *args, **kwargs):

        W_fac_loc = pyro.param("W_fac_loc", torch.zeros((self.num_factors, self.num_genes), device=self.device))
        W_fac_scale = pyro.param(
            "W_fac_scale",
            0.1 * torch.ones((self.num_factors, self.num_genes), device=self.device),
            constraint=dist.constraints.greater_than(0.01),
        )
        V_fac_loc = pyro.param("V_fac_loc", torch.zeros((self.num_factors, self.num_proteins), device=self.device))
        V_fac_scale = pyro.param(
            "V_fac_scale",
            0.1 * torch.ones((self.num_factors, self.num_proteins), device=self.device),
            constraint=dist.constraints.greater_than(0.01),
        )

        with pyro.plate("factor", self.num_factors):
            pyro.sample("W_fac", dist.Normal(W_fac_loc, W_fac_scale).to_event(1))
            pyro.sample("V_fac", dist.Normal(V_fac_loc, V_fac_scale).to_event(1))

        if self.num_batches > 1:
            if not self.batch_params:
                W_add_loc = pyro.param("W_add_loc", torch.zeros((self.num_batches, self.num_genes), device=self.device))
                W_add_scale = pyro.param(
                    "W_add_scale",
                    0.1 * torch.ones((self.num_batches, self.num_genes), device=self.device),
                    constraint=dist.constraints.greater_than(0.01),
                )
                V_add_loc = pyro.param(
                    "V_add_loc", torch.zeros((self.num_batches, self.num_proteins), device=self.device)
                )
                V_add_scale = pyro.param(
                    "V_add_scale",
                    0.1 * torch.ones((self.num_batches, self.num_proteins), device=self.device),
                    constraint=dist.constraints.greater_than(0.01),
                )

                with pyro.plate("batches", self.num_batches):
                    pyro.sample("W_add", dist.Normal(W_add_loc, W_add_scale).to_event(1))
                    pyro.sample("V_add", dist.Normal(V_add_loc, V_add_scale).to_event(1))

        β_rna_loc = pyro.param("β_rna_loc", torch.tensor(0.0, device=self.device))
        β_rna_scale = pyro.param(
            "β_rna_rate", torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive
        )
        pyro.sample(
            "β_rna", dist.TransformedDistribution(dist.Normal(β_rna_loc, β_rna_scale), dist.transforms.ExpTransform())
        )

        β_prot_loc = pyro.param("β_prot_loc", torch.tensor(0.0, device=self.device))
        β_prot_scale = pyro.param(
            "β_prot_rate", torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive
        )
        pyro.sample(
            "β_prot",
            dist.TransformedDistribution(dist.Normal(β_prot_loc, β_prot_scale), dist.transforms.ExpTransform()),
        )

        σ_rna_loc = pyro.param("σ_rna_loc", torch.zeros(self.num_genes, device=self.device))
        σ_rna_scale = pyro.param(
            "σ_rna_scale", 0.1 * torch.ones(self.num_genes, device=self.device), constraint=dist.constraints.positive
        )

        with pyro.plate("genes", self.num_genes):
            pyro.sample(
                "σ_rna",
                dist.TransformedDistribution(dist.Normal(σ_rna_loc, σ_rna_scale), dist.transforms.ExpTransform()),
            )

        σ_prot_loc = pyro.param("σ_prot_loc", torch.zeros(self.num_proteins, device=self.device))
        σ_prot_scale = pyro.param(
            "σ_prot_scale",
            0.1 * torch.ones(self.num_proteins, device=self.device),
            constraint=dist.constraints.positive,
        )

        with pyro.plate("proteins", self.num_proteins):
            pyro.sample(
                "σ_prot",
                dist.TransformedDistribution(dist.Normal(σ_prot_loc, σ_prot_scale), dist.transforms.ExpTransform()),
            )

        z_loc = pyro.param("z_loc", torch.zeros((self.num_cells, self.num_factors), device=self.device))
        z_scale = pyro.param(
            "z_scale",
            0.1 * torch.ones((self.num_cells, self.num_factors), device=self.device),
            constraint=dist.constraints.greater_than(0.01),
        )

        # s_loc = pyro.param("s_loc", log_mean * torch.ones(num_cells, device=device),)
        # s_scale = pyro.param("s_scale", log_var * torch.ones(num_cells, device=device), constraint=dist.constraints.positive)
        with pyro.plate("cells", self.num_cells):
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


# class scCCA(SVIModel):
#     def __init__(
#         self,
#         adata,
#         num_factors=10,
#         layers_key="counts",
#         protein_obsm_key="protein_expression",
#         batch_key=None,
#         device=torch.device("cuda"),
#         svi_kwargs=DEFAULT_SVI_SETTINGS,
#     ):
#         super().__init__(**svi_kwargs)
#         self.adata = adata
#         self.device = device

#         if layers_key is not None:
#             self.X = torch.tensor(adata.layers[layers_key].toarray(), device=device)
#         else:
#             self.X = torch.tensor(adata.X, device=device)

#         self.Y = torch.tensor(adata.obsm[protein_obsm_key].values, device=device)

#         self.num_factors = num_factors
#         self.num_cells, self.num_genes = self.X.shape
#         _, self.num_proteins = self.Y.shape

#         if batch_key is None:
#             self.batch = torch.ones((self.num_cells, 1), device=device)
#         else:
#             dummies = pd.get_dummies(adata.obs[batch_key]).values.astype(np.float32)
#             self.batch = torch.tensor(dummies, device=device)

#         self.num_batches = self.batch.shape[1]
#         # compute library
#         self.rna_log_library = torch.log(self.X.sum(1, keepdims=True))
#         self.prot_log_library = torch.log(self.Y.sum(1, keepdims=True))
#         # self.library_sizes = torch.diag(self.batch.T @ self.batch)

#         # compute hyper parameters
#         self.rna_log_mean = batch_mean(
#             self.rna_log_library, self.batch
#         )  # self.rna_library.mean()
#         self.rna_log_var = batch_var(
#             self.rna_log_library, self.batch
#         )  # 1 / (self.library_sizes - 1) * torch.pow(self.rna_library - self.batch.T @ self.rna_library, 2).sum()

#         self.prot_log_mean = batch_mean(self.prot_log_library, self.batch)
#         self.prot_log_var = batch_var(self.prot_log_library, self.batch)

#     @property
#     def _latent_variables(self):
#         return ["z", "s", "α"]

#     def model(self, *args, **kwargs):
#         # factor matrices
#         W_fac = pyro.param(
#             "W_fac",
#             torch.randn((1, self.num_factors, self.num_genes), device=self.device),
#         )
#         W_add = pyro.param(
#             "W_add",
#             torch.zeros((self.num_batches, 1, self.num_genes), device=self.device),
#         )

#         W_mul = pyro.param(
#             "W_mul", torch.ones((1, 1, self.num_genes), device=self.device)
#         )
#         W_bat = pyro.param(
#             "W_bat", torch.ones((self.num_batches, 1, 1), device=self.device)
#         )
#         W = (W_fac + W_add) * (W_mul * W_bat)

#         # print(W.shape)

#         V_fac = pyro.param(
#             "V_fac",
#             torch.randn((1, self.num_factors, self.num_proteins), device=self.device),
#         )
#         V_add = pyro.param(
#             "V_add",
#             torch.zeros((self.num_batches, 1, self.num_proteins), device=self.device),
#         )

#         V_mul = pyro.param(
#             "V_mul", torch.ones((1, 1, self.num_proteins), device=self.device)
#         )
#         V_bat = pyro.param(
#             "V_bat", torch.ones((self.num_batches, 1, 1), device=self.device)
#         )
#         V = (V_fac + V_add) * (V_mul * V_bat)

#         β_rna = pyro.sample(
#             "β_rna",
#             dist.Gamma(
#                 torch.tensor(9.0, device=self.device),
#                 torch.tensor(3.0, device=self.device),
#             ),
#         )
#         β_prot = pyro.sample(
#             "β_prot",
#             dist.Gamma(
#                 torch.tensor(9.0, device=self.device),
#                 torch.tensor(3.0, device=self.device),
#             ),
#         )

#         with pyro.plate("genes", self.num_genes):
#             σ_rna = pyro.sample("σ_rna", dist.Exponential(β_rna))
#             α_rna = pyro.deterministic("α_rna", (1 / σ_rna).reshape(1, -1))

#         with pyro.plate("proteins", self.num_proteins):
#             σ_prot = pyro.sample("σ_prot", dist.Exponential(β_prot))
#             α_prot = pyro.deterministic("α_prot", (1 / σ_prot).reshape(1, -1))
#         # print(α.device)

#         with pyro.plate("cells", self.num_cells):
#             z = pyro.sample(
#                 "z",
#                 dist.Normal(
#                     torch.zeros(self.num_factors, device=self.device),
#                     torch.ones(self.num_factors, device=self.device),
#                 ).to_event(1),
#             )

#             l_rna = pyro.sample(
#                 "l_rna",
#                 dist.LogNormal(
#                     (self.batch @ self.rna_log_mean).squeeze(),
#                     (self.batch @ self.rna_log_var).squeeze(),
#                 ),
#             )
#             μ_rna = pyro.deterministic(
#                 "μ_rna",
#                 l_rna.reshape(-1, 1)
#                 * torch.exp(
#                     (
#                         self.batch.T.unsqueeze(-1) * torch.einsum("cf,bfp->bcp", z, W)
#                     ).sum(0)
#                 ),  # torch.exp(z @ W)
#             )

#             pyro.sample(
#                 "rna", dist.GammaPoisson(α_rna, α_rna / μ_rna).to_event(1), obs=self.X
#             )

#             l_prot = pyro.sample(
#                 "l_prot",
#                 dist.LogNormal(
#                     (self.batch @ self.prot_log_mean).squeeze(),
#                     (self.batch @ self.prot_log_var).squeeze(),
#                 ),
#             )
#             μ_prot = pyro.deterministic(
#                 "μ_prot",
#                 l_prot.reshape(-1, 1)
#                 * torch.exp(
#                     (
#                         self.batch.T.unsqueeze(-1) * torch.einsum("cf,bfp->bcp", z, V)
#                     ).sum(0)
#                 ),
#             )
#             # print(μ_protein)
#             pyro.sample(
#                 "prot",
#                 dist.GammaPoisson(α_prot, α_prot / μ_prot).to_event(1),
#                 obs=self.Y,
#             )

#     def guide(self, *args, **kwargs):
#         β_rna_loc = pyro.param("β_rna_loc", torch.tensor(0.0, device=self.device))
#         β_rna_scale = pyro.param(
#             "β_rna_rate",
#             torch.tensor(1.0, device=self.device),
#             constraint=dist.constraints.positive,
#         )
#         pyro.sample(
#             "β_rna",
#             dist.TransformedDistribution(
#                 dist.Normal(β_rna_loc, β_rna_scale), dist.transforms.ExpTransform()
#             ),
#         )

#         β_prot_loc = pyro.param("β_prot_loc", torch.tensor(0.0, device=self.device))
#         β_prot_scale = pyro.param(
#             "β_prot_rate",
#             torch.tensor(1.0, device=self.device),
#             constraint=dist.constraints.positive,
#         )
#         pyro.sample(
#             "β_prot",
#             dist.TransformedDistribution(
#                 dist.Normal(β_prot_loc, β_prot_scale), dist.transforms.ExpTransform()
#             ),
#         )

#         σ_rna_loc = pyro.param(
#             "σ_rna_loc", torch.zeros(self.num_genes, device=self.device)
#         )
#         σ_rna_scale = pyro.param(
#             "σ_rna_scale",
#             0.1 * torch.ones(self.num_genes, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         with pyro.plate("genes", self.num_genes):
#             # pyro.sample("σ", dist.Exponential(σ_rate))
#             pyro.sample(
#                 "σ_rna",
#                 dist.TransformedDistribution(
#                     dist.Normal(σ_rna_loc, σ_rna_scale), dist.transforms.ExpTransform()
#                 ),
#             )

#         σ_prot_loc = pyro.param(
#             "σ_prot_loc", torch.zeros(self.num_proteins, device=self.device)
#         )
#         σ_prot_scale = pyro.param(
#             "σ_prot_scale",
#             0.1 * torch.ones(self.num_proteins, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         with pyro.plate("proteins", self.num_proteins):
#             pyro.sample(
#                 "σ_prot",
#                 dist.TransformedDistribution(
#                     dist.Normal(σ_prot_loc, σ_prot_scale),
#                     dist.transforms.ExpTransform(),
#                 ),
#             )

#         z_loc = pyro.param(
#             "z_loc",
#             torch.zeros((self.num_cells, self.num_factors), device=self.device),
#         )
#         z_scale = pyro.param(
#             "z_scale",
#             0.1 * torch.ones((self.num_cells, self.num_factors), device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         # l_rna_loc = pyro.param("l_rna_loc", torch.zeros(self.num_cells, device=self.device))
#         # l_rna_scale = pyro.param("l_rna_scale", .1 * torch.ones(self.num_cells, device=self.device), constraint=dist.constraints.positive)
#         l_rna_loc = pyro.param(
#             "l_rna_loc",
#             (self.batch @ self.rna_log_mean).squeeze(),
#             constraint=dist.constraints.positive,
#         )
#         l_rna_scale = pyro.param(
#             "l_rna_scale",
#             (self.batch @ self.rna_log_var).squeeze(),
#             constraint=dist.constraints.positive,
#         )

#         warn_if_nan(l_rna_loc)
#         warn_if_nan(l_rna_scale)

#         # l_prot_loc = pyro.param("l_prot_loc", torch.zeros(self.num_cells, device=self.device))
#         # l_prot_scale = pyro.param("l_prot_scale", .1 * torch.ones(self.num_cells, device=self.device), constraint=dist.constraints.positive)

#         l_prot_loc = pyro.param(
#             "l_prot_loc",
#             (self.batch @ self.prot_log_mean).squeeze(),
#             constraint=dist.constraints.positive,
#         )
#         l_prot_scale = pyro.param(
#             "l_prot_scale",
#             (self.batch @ self.prot_log_var).squeeze(),
#             constraint=dist.constraints.positive,
#         )

#         with pyro.plate("cells", self.num_cells):
#             pyro.sample("l_rna", dist.LogNormal(l_rna_loc, l_rna_scale))
#             pyro.sample("l_prot", dist.LogNormal(l_prot_loc, l_prot_scale))
#             # pyro.sample("l_rna", dist.TransformedDistribution(dist.Normal(l_rna_loc, l_rna_scale), dist.transforms.ExpTransform()))
#             # pyro.sample("l_prot", dist.TransformedDistribution(dist.Normal(l_prot_loc, l_prot_scale), dist.transforms.ExpTransform()))
#             pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


# class scCCAReparam(SVIModel):
#     def __init__(
#         self,
#         adata,
#         num_factors=10,
#         layers_key="counts",
#         protein_obsm_key="protein_expression",
#         batch_key=None,
#         device=torch.device("cuda"),
#         svi_kwargs=DEFAULT_SVI_SETTINGS,
#     ):
#         super().__init__(**svi_kwargs)
#         self.adata = adata
#         self.device = device

#         if layers_key is not None:
#             self.X = torch.tensor(adata.layers[layers_key].toarray(), device=device)
#         else:
#             self.X = torch.tensor(adata.X, device=device)

#         self.Y = torch.tensor(adata.obsm[protein_obsm_key].values, device=device)

#         self.num_factors = num_factors
#         self.num_cells, self.num_genes = self.X.shape
#         _, self.num_proteins = self.Y.shape

#         if batch_key is None:
#             self.batch = torch.ones((self.num_cells, 1), device=device)
#         else:
#             dummies = pd.get_dummies(adata.obs[batch_key]).values.astype(np.float32)
#             self.batch = torch.tensor(dummies, device=device)

#         self.num_batches = self.batch.shape[1]
#         # compute library
#         self.rna_log_library = torch.log(self.X.sum(1, keepdims=True))
#         self.prot_log_library = torch.log(self.Y.sum(1, keepdims=True))
#         # self.library_sizes = torch.diag(self.batch.T @ self.batch)

#         # compute hyper parameters
#         self.rna_log_mean = batch_mean(
#             self.rna_log_library, self.batch
#         )  # self.rna_library.mean()
#         self.rna_log_var = batch_var(
#             self.rna_log_library, self.batch
#         )  # 1 / (self.library_sizes - 1) * torch.pow(self.rna_library - self.batch.T @ self.rna_library, 2).sum()

#         self.prot_log_mean = batch_mean(self.prot_log_library, self.batch)
#         self.prot_log_var = batch_var(self.prot_log_library, self.batch)

#     @property
#     def _latent_variables(self):
#         return [
#             "z",
#             "β_rna",
#             "β_prot",
#             "α_rna",
#             "α_prot",
#             "γ_rna",
#             "γ_prot",
#             "δ_rna",
#             "δ_prot",
#             "l_rna",
#             "l_prot",
#         ]

#     def model(self, *args, **kwargs):
#         # factor matrices
#         W_fac = pyro.param(
#             "W_fac",
#             torch.randn((1, self.num_factors, self.num_genes), device=self.device),
#         )
#         if self.num_batches > 1:
#             W_add = pyro.param(
#                 "W_add",
#                 torch.zeros((self.num_batches, 1, self.num_genes), device=self.device),
#             )

#             W_mul = pyro.param(
#                 "W_mul", torch.ones((1, 1, self.num_genes), device=self.device)
#             )
#             W_bat = pyro.param(
#                 "W_bat", torch.ones((self.num_batches, 1, 1), device=self.device)
#             )

#             W = (W_fac + W_add) * (W_mul * W_bat)
#         else:
#             W = W_fac

#         V_fac = pyro.param(
#             "V_fac",
#             torch.randn((1, self.num_factors, self.num_proteins), device=self.device),
#         )

#         if self.num_batches > 1:
#             V_add = pyro.param(
#                 "V_add",
#                 torch.zeros(
#                     (self.num_batches, 1, self.num_proteins), device=self.device
#                 ),
#             )

#             V_mul = pyro.param(
#                 "V_mul", torch.ones((1, 1, self.num_proteins), device=self.device)
#             )
#             V_bat = pyro.param(
#                 "V_bat", torch.ones((self.num_batches, 1, 1), device=self.device)
#             )
#             V = (V_fac + V_add) * (V_mul * V_bat)
#         else:
#             V = V_fac

#         # hyper priors (num_batches)
#         with pyro.plate("batches", self.num_batches):
#             β_rna = pyro.sample(
#                 "β_rna",
#                 dist.Gamma(
#                     torch.tensor(9.0, device=self.device),
#                     torch.tensor(3.0, device=self.device),
#                 ),
#             )
#             β_prot = pyro.sample(
#                 "β_prot",
#                 dist.Gamma(
#                     torch.tensor(9.0, device=self.device),
#                     torch.tensor(3.0, device=self.device),
#                 ),
#             )
#             γ_rna = pyro.sample(
#                 "γ_rna",
#                 dist.Normal(
#                     torch.tensor(0.0, device=self.device),
#                     torch.tensor(1.0, device=self.device),
#                 ),
#             )
#             γ_prot = pyro.sample(
#                 "γ_prot",
#                 dist.Normal(
#                     torch.tensor(0.0, device=self.device),
#                     torch.tensor(1.0, device=self.device),
#                 ),
#             )
#             δ_rna = pyro.sample(
#                 "δ_rna",
#                 dist.Gamma(
#                     torch.tensor(1.0, device=self.device),
#                     torch.tensor(1.0, device=self.device),
#                 ),
#             )
#             δ_prot = pyro.sample(
#                 "δ_prot",
#                 dist.Gamma(
#                     torch.tensor(1.0, device=self.device),
#                     torch.tensor(1.0, device=self.device),
#                 ),
#             )

#         with pyro.plate("genes", self.num_genes):
#             σ_rna = pyro.sample("σ_rna", dist.Exponential(β_rna))
#             α_rna = pyro.deterministic("α_rna", (1 / σ_rna).reshape(1, -1))

#         with pyro.plate("proteins", self.num_proteins):
#             σ_prot = pyro.sample("σ_prot", dist.Exponential(β_prot))
#             α_prot = pyro.deterministic("α_prot", (1 / σ_prot).reshape(1, -1))

#         with pyro.plate("cells", self.num_cells):
#             z = pyro.sample(
#                 "z",
#                 dist.Normal(
#                     torch.zeros(self.num_factors, device=self.device),
#                     torch.ones(self.num_factors, device=self.device),
#                 ).to_event(1),
#             )

#             l_rna_offset = pyro.sample(
#                 "l_rna_offset",
#                 dist.Normal(
#                     torch.tensor(0.0, device=self.device),
#                     torch.tensor(1.0, device=self.device),
#                 ),
#             )
#             l_rna = pyro.deterministic(
#                 "l_rna",
#                 (self.batch @ γ_rna)
#                 + l_rna_offset * (self.batch @ torch.pow(δ_rna, -1 / 2)),
#             )
#             # print(l_rna.shape)
#             μ_rna = pyro.deterministic(
#                 "μ_rna",
#                 torch.exp(l_rna.reshape(-1, 1))
#                 * torch.exp(
#                     (
#                         self.batch.T.unsqueeze(-1) * torch.einsum("cf,bfp->bcp", z, W)
#                     ).sum(0)
#                 ),  # torch.exp(z @ W)
#             )

#             pyro.sample(
#                 "rna", dist.GammaPoisson(α_rna, α_rna / μ_rna).to_event(1), obs=self.X
#             )

#             l_prot_offset = pyro.sample(
#                 "l_prot_offset",
#                 dist.Normal(
#                     torch.tensor(0.0, device=self.device),
#                     torch.tensor(1.0, device=self.device),
#                 ),
#             )
#             l_prot = pyro.deterministic(
#                 "l_prot",
#                 (self.batch @ γ_prot)
#                 + l_prot_offset * (self.batch @ torch.pow(δ_prot, -1 / 2)),
#             )

#             μ_prot = pyro.deterministic(
#                 "μ_prot",
#                 torch.exp(l_prot.reshape(-1, 1))
#                 * torch.exp(
#                     (
#                         self.batch.T.unsqueeze(-1) * torch.einsum("cf,bfp->bcp", z, V)
#                     ).sum(0)
#                 ),
#             )
#             # print(μ_protein)
#             pyro.sample(
#                 "prot",
#                 dist.GammaPoisson(α_prot, α_prot / μ_prot).to_event(1),
#                 obs=self.Y,
#             )

#     def guide(self, *args, **kwargs):
#         β_rna_loc = pyro.param(
#             "β_rna_loc", torch.zeros(self.num_batches, device=self.device)
#         )
#         β_rna_scale = pyro.param(
#             "β_rna_rate",
#             torch.ones(self.num_batches, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         β_prot_loc = pyro.param(
#             "β_prot_loc", torch.zeros(self.num_batches, device=self.device)
#         )
#         β_prot_scale = pyro.param(
#             "β_prot_rate",
#             torch.ones(self.num_batches, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         γ_rna_loc = pyro.param(
#             "γ_rna_loc", torch.zeros(self.num_batches, device=self.device),
#         )
#         γ_rna_scale = pyro.param(
#             "γ_rna_scale",
#             0.01 * torch.ones(self.num_batches, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         δ_rna_loc = pyro.param(
#             "δ_rna_loc", torch.zeros(self.num_batches, device=self.device)
#         )
#         δ_rna_scale = pyro.param(
#             "δ_rna_scale",
#             torch.ones(self.num_batches, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         γ_prot_loc = pyro.param(
#             "γ_prot_loc", torch.zeros(self.num_batches, device=self.device),
#         )
#         γ_prot_scale = pyro.param(
#             "γ_prot_scale",
#             0.1 * torch.ones(self.num_batches, device=self.device),
#             constraint=dist.constraints.positive,
#         )
#         δ_prot_loc = pyro.param(
#             "δ_prot_loc", torch.zeros(self.num_batches, device=self.device)
#         )
#         δ_prot_scale = pyro.param(
#             "δ_prot_scale",
#             torch.ones(self.num_batches, device=self.device),
#             constraint=dist.constraints.positive,
#         )
#         with pyro.plate("batches", self.num_batches):
#             pyro.sample(
#                 "β_rna",
#                 dist.TransformedDistribution(
#                     dist.Normal(β_rna_loc, β_rna_scale), dist.transforms.ExpTransform()
#                 ),
#             )
#             pyro.sample(
#                 "β_prot",
#                 dist.TransformedDistribution(
#                     dist.Normal(β_prot_loc, β_prot_scale),
#                     dist.transforms.ExpTransform(),
#                 ),
#             )

#             pyro.sample("γ_rna", dist.Normal(γ_rna_loc, γ_rna_scale))
#             pyro.sample("γ_prot", dist.Normal(γ_prot_loc, γ_prot_scale))

#             pyro.sample(
#                 "δ_rna",
#                 dist.TransformedDistribution(
#                     dist.Normal(δ_rna_loc, δ_rna_scale), dist.transforms.ExpTransform()
#                 ),
#             )
#             pyro.sample(
#                 "δ_prot",
#                 dist.TransformedDistribution(
#                     dist.Normal(δ_prot_loc, δ_prot_scale),
#                     dist.transforms.ExpTransform(),
#                 ),
#             )

#         σ_rna_loc = pyro.param(
#             "σ_rna_loc", torch.zeros(self.num_genes, device=self.device)
#         )
#         σ_rna_scale = pyro.param(
#             "σ_rna_scale",
#             0.1 * torch.ones(self.num_genes, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         with pyro.plate("genes", self.num_genes):
#             pyro.sample(
#                 "σ_rna",
#                 dist.TransformedDistribution(
#                     dist.Normal(σ_rna_loc, σ_rna_scale), dist.transforms.ExpTransform()
#                 ),
#             )

#         σ_prot_loc = pyro.param(
#             "σ_prot_loc", torch.zeros(self.num_proteins, device=self.device)
#         )
#         σ_prot_scale = pyro.param(
#             "σ_prot_scale",
#             0.1 * torch.ones(self.num_proteins, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         with pyro.plate("proteins", self.num_proteins):
#             pyro.sample(
#                 "σ_prot",
#                 dist.TransformedDistribution(
#                     dist.Normal(σ_prot_loc, σ_prot_scale),
#                     dist.transforms.ExpTransform(),
#                 ),
#             )

#         z_loc = pyro.param(
#             "z_loc",
#             torch.zeros((self.num_cells, self.num_factors), device=self.device),
#         )
#         z_scale = pyro.param(
#             "z_scale",
#             0.1 * torch.ones((self.num_cells, self.num_factors), device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         l_rna_loc = pyro.param(
#             "l_rna_offset_loc",
#             # self.rna_log_mean,
#             torch.zeros(self.num_cells, device=self.device),
#             # constraint=dist.constraints.interval(-6, 6)
#         )
#         l_rna_scale = pyro.param(
#             "l_rna_offset_scale",
#             0.1 * torch.ones(self.num_cells, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         l_prot_loc = pyro.param(
#             "l_prot_offset_loc", torch.zeros(self.num_cells, device=self.device),
#         )
#         l_prot_scale = pyro.param(
#             "l_prot_offset_scale",
#             0.1 * torch.ones(self.num_cells, device=self.device),
#             constraint=dist.constraints.positive,
#         )

#         with pyro.plate("cells", self.num_cells):
#             pyro.sample("l_rna_offset", dist.Normal(l_rna_loc, l_rna_scale))
#             pyro.sample("l_prot_offset", dist.Normal(l_prot_loc, l_prot_scale))
#             pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

#     def deterministic(self):
#         self.results = {}

#         if self.num_batches > 1:

#             self.results["V"] = (self.params["V_fac"] + self.params["V_add"]) * (
#                 self.params["V_bat"] * self.params["V_mul"]
#             )

#             self.results["W"] = (self.params["W_fac"] + self.params["W_add"]) * (
#                 self.params["W_bat"] * self.params["W_mul"]
#             )
#         else:
#             self.results["V"] = self.params["V_fac"]
#             self.results["W"] = self.params["W_fac"]

#         self.results["μ_prot"] = np.expand_dims(
#             np.exp(self.posterior["l_prot"]), -1
#         ) * np.exp(np.einsum("bfg,bcf->bcg", self.results["V"], self.posterior["z"]))

#         self.results["μ_rna"] = np.expand_dims(
#             np.exp(self.posterior["l_rna"]), -1
#         ) * np.exp(np.einsum("bfg,bcf->bcg", self.results["W"], self.posterior["z"]))

#         self.results["rna_denoised"] = np.expand_dims(
#             np.exp(self.posterior["l_rna"]), -1
#         ) * np.exp(np.einsum("bfg,bcf->bcg", self.params["W_fac"], self.posterior["z"]))

#         self.results["prot_denoised"] = np.expand_dims(
#             np.exp(self.posterior["l_prot"]), -1
#         ) * np.exp(np.einsum("bfg,bcf->bcg", self.params["V_fac"], self.posterior["z"]))

#         self.adata.obsm["X_sccca"] = self.posterior.mean("z")
#         # self.adata.varm['V_sccca'] = self.params['V_fac']
#         # self.adata.varm['W_sccca'] = self.params['W_fac']
#         # np.einsum("bfg,bcf->bcg", V, m0.posterior["z"])
#         # pass
#         # self.mu = np.expand_dims(self.posterior['s'], -1) * np.exp(np.einsum('fg,scf->scg', self.params['W'], self.posterior['z']))
