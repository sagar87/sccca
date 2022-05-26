import pyro
import torch

# DEFAULT_SVI_SETTINGS = dict(
#     num_epochs=5000,
#     num_samples=100,
#     optimizer_kwargs={"lr": 0.01},
#     scheduler_kwargs={"factor": 0.99},
#     loss_kwargs={"num_particles": 1},
# )

DEFAULT = dict(
    num_epochs=5000,
    num_samples=100,
    optimizer=pyro.optim.ClippedAdam,
    optimizer_kwargs={"lr": 0.001, "betas": (0.95, 0.999)},
    scheduler=None,
    loss_kwargs={"num_particles": 1},
)
AGGRESSIVE_DEC = dict(
    num_epochs=4000,
    num_samples=100,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001, "betas": (0.95, 0.999)},
    scheduler_kwargs={"factor": 0.1, "min_lr": 1e-6},
    loss_kwargs={"num_particles": 1},
)
