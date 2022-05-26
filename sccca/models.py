import sys

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from tp import DEFAULT

sys.path.insert(0, "/home/voehring/voehring/opt/pyromaniac/pyromaniac/")
from handler import SVIModel


def log_library(X):
    """
    Computes the log library.
    """
    return torch.log(X.sum(1, keepdims=True))


def batch_mean(library, batch):
    """
    Computes the batch mean.
    @params:
    rna_library: a num_cells x 1 vector with (log) rna size
    batch: a num_cells x num_batches dummies matrix
    """
    return 1 / torch.diag(batch.T @ batch) * (batch.T @ library).squeeze()


def batch_var(library, batch, ddof=1):
    m = library.squeeze() - batch @ batch_mean(library, batch)
    return 1 / (torch.diag(batch.T @ batch) - ddof) * (batch * (m**2).unsqueeze(-1)).sum(0)


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


class scCCACore(SVIModel):
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
