import pyro
import pyro.distributions                                       as dist
from pyro.distributions                                         import MultivariateNormal, Beta, Categorical, InverseGamma, Gamma
from pyro.distributions                                         import Exponential, Uniform
from pyro.infer                                                 import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate

import torch
import torch.nn                                                 as nn
from torch.distributions                                        import constraints

import numpy                                                    as np
from tqdm                                                       import tqdm
from datetime                                                   import datetime
import os
import pathlib

from MAP                                                        import train_map_model, PretrainNF
from distributions                                              import UniformStiefel
from utils                                                      import angle_to_matrix, mix_weights

"""
Implementation of the mean field variational inference algorithm
to the posterior at https://arxiv.org/abs/2205.15717.
"""


def train(num_iterations):

        pyro.clear_param_store()
        losses = []

        for j in tqdm(range(num_iterations)):
            loss = svi.step(data)
            losses.append(loss)
            if train_param["cosine"]:
                optimizer.step()
            if j % 1000 == 0 or j == 10:
                print(pyro.param("tau")[0])
                print(loss)

        return losses


class MeanFieldVI(nn.Module):

    def __init__(self, ndim, T, flows, map_init):
        """
        We keep alpha, kappa etc. fixed here to 1
        """

        super().__init__()

        stiefel_dim = ndim * (ndim - 1) // 2

        self.ndim = ndim
        self.stiefel_dim = stiefel_dim
        self.T = T
        self.flows = flows
        self.map_init = map_init

    @config_enumerate
    def model(self, data):

        N = len(data)

        beta = pyro.sample("beta", Beta(1, torch.ones([self.T-1])).to_event())
        beta_weights = mix_weights(beta)

        with pyro.plate("b_lamb_plate", self.ndim):
            b = pyro.sample("b", Exponential(1))
            lamb = pyro.sample("lamb", InverseGamma(1, b))
        diag_lamb = torch.diag_embed(lamb).unsqueeze(-3)

        with pyro.plate("mu_theta_plate", self.T):
            mu = pyro.sample("mu", MultivariateNormal(torch.zeros([self.ndim]), torch.eye(self.ndim)))
            theta = pyro.sample('theta', UniformStiefel(self.T, self.ndim))

        with pyro.plate("data_plate", N):
            z = pyro.sample("z", Categorical(beta_weights), infer={'enumerate': 'parallel'})
            if beta.dim() < 2:  # as first SVI step has num_particles=1
                mu_z = mu[z]
                O_z =  angle_to_matrix(theta, self.ndim)[z]
                obs = pyro.sample("obs", MultivariateNormal(mu_z, O_z @ diag_lamb @ O_z.transpose(-1, -2)), obs=data)  
            else:
                mu_z = mu[torch.arange(mu.shape[0]).unsqueeze(-1), z]  
                orth_mats = angle_to_matrix(theta, self.ndim)
                cov = orth_mats @ diag_lamb @ orth_mats.transpose(-1, -2)
                cov_z = cov[torch.arange(cov.shape[0]).unsqueeze(-1), z]
                obs = pyro.sample("obs", MultivariateNormal(mu_z, cov_z), obs=data)

    @config_enumerate
    def guide(self, data):
        """
        Note that we can also alter the 
        initialisation for VI parameters.
        """

        N = len(data)

        # register all nf params with pyro
        for i in range(len(self.flows)):
            pyro.module(str(i), self.flows[i].mprqat.autoregressive_net)

        kappa_1 = pyro.param("kappa_1", lambda: 1 + self.map_init["beta"] * 1000, constraint=constraints.positive)
        kappa_2 = pyro.param("kappa_2", lambda: 1 + (1-self.map_init["beta"]) * 1000, constraint=constraints.positive)
        q_beta = pyro.sample("beta", Beta(kappa_1, kappa_2).to_event())

        gamma_1 = pyro.param("gamma_1", lambda: (1+N) * torch.ones([self.ndim]), constraint=constraints.positive)
        gamma_2 = pyro.param("gamma_2", lambda: N * self.map_init["lamb"], constraint=constraints.positive)
        delta_1 = pyro.param("delta_1", lambda: N * torch.ones([self.ndim]), constraint=constraints.positive)
        delta_2 = pyro.param("delta_2", lambda: N / self.map_init["b"], constraint=constraints.positive)
        with pyro.plate("b_lamb_plate", self.ndim):
            q_lamb = pyro.sample("lamb", InverseGamma(gamma_1, gamma_2))
            q_b = pyro.sample("b", Gamma(delta_1, delta_2))

        tau = pyro.param('tau', lambda: self.map_init["mu"])
        sigma = pyro.param('sigma', lambda: 0.001 * torch.ones([self.T]), constraint=constraints.positive).view(-1, 1, 1)
        with pyro.plate("mu_theta_plate", self.T):
            q_mu = pyro.sample("mu", MultivariateNormal(tau, sigma * torch.eye(self.ndim))) 
            q_theta_base = Uniform(torch.ones([self.T, self.stiefel_dim]) * -np.pi,
                                torch.ones([self.T, self.stiefel_dim]) * np.pi).to_event(1)
            q_theta_dist = dist.TransformedDistribution(q_theta_base, list(self.flows))
            q_theta = pyro.sample("theta", q_theta_dist)

        phi = pyro.param('phi', lambda: self.map_init["labels"], constraint=constraints.simplex)
        with pyro.plate("data_plate", N):
            q_z = pyro.sample("z", Categorical(phi), infer={'enumerate': 'parallel'})

        return kappa_1, kappa_2, gamma_1, gamma_2, delta_1, delta_2, tau, sigma, q_theta_dist, phi


if __name__ == "__main__":

    """
    See commented code for saving variational 
    parameters etc.
    """

    map_params = {
        "steps": 10000,
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "lamb_init": (10*torch.ones(2), torch.tensor([.01, .1]))
    }

    pretrain_params = {
        "steps": 500,
        "lr": 5e-4,
        "betas": (0.95, 0.999),
        "num_particles": 2 ** 4
    }

    # num_trans is the number of transformation in the
    # normalising flow
    flow_params = {
        "num_trans": 5,
        "num_blocks": 1,
        "num_hidden_channels": 256,
        "num_bins": 10,
        "activation": torch.nn.ReLU,
        "dropout_probability": 0.0,
        "permute_mask": True,
        "init_identity": True
    }

    # T is the max number of clusters in the
    # Dirichlet process mixture.
    train_param = {
        "steps": 3000,
        "lr": 5e-4,
        "T": 20,
        "betas": (0.95, 0.999),
        "num_particles": 2 ** 4,
        "cosine": True
    }

    file_dir = pathlib.Path(__file__).parent

    # makes folder to save our variational parameters
    # NOTE: uncomment to save variational parameters
    """
    experiments_dir = datetime.now().strftime("%d:%m:%Y_%H:%M:%S")
    os.mkdir(file_dir / experiments_dir)
    """

    # set up training data - we use this file as an example
    data = np.genfromtxt(file_dir / "spiral_2D_delta_01_data.csv", delimiter="", encoding=None)
    data = torch.from_numpy(data).float()
    ndim = data.shape[-1]

    # run MAP training for initialisation
    map_losses, map_init = train_map_model(train_param["T"], data, map_params)

    # NOTE: uncomment to save MAP values
    """
    torch.save(map_init, file_dir / experiments_dir / "map_init.pt")
    """

    # pre-train non-joint NF
    pre_nfs = PretrainNF(ndim, train_param["T"], flow_params, map_init)
    optimizer = pyro.optim.ClippedAdam({"lr": pretrain_params["lr"], "betas": pretrain_params["betas"]})
    elbo = Trace_ELBO(num_particles=pretrain_params["num_particles"], vectorize_particles=True) 
    svi = SVI(pre_nfs.model, pre_nfs.guide, optimizer, loss=elbo)

    pretrain_losses = []
    for _ in tqdm(range(pretrain_params["steps"])):
        loss = svi.step(data)
        pretrain_losses.append(loss)

    # NOTE: uncomment to save pre-trained normalising flow
    """
    torch.save(pre_nfs.flows.state_dict(), file_dir / experiments_dir / "pre_flows.pt")
    """

    # VI training
    nfs = MeanFieldVI(ndim=ndim, 
                      T=train_param["T"], 
                      flows=pre_nfs.flows,
                      map_init=map_init)

    if train_param["cosine"]:
        base_optimizer = torch.optim.Adam
        params_scheduler = {"optimizer": base_optimizer, 
                            "optim_args": {"lr": train_param["lr"], 
                                           "betas":train_param["betas"]}, 
                            "T_max": train_param["steps"]}
        optimizer = pyro.optim.CosineAnnealingLR(params_scheduler)
    else:
        optimizer = pyro.optim.Adam({"lr": train_param["lr"], "betas":train_param["betas"]})

    elbo = TraceEnum_ELBO(num_particles=train_param["num_particles"], vectorize_particles=True) 
    svi = SVI(nfs.model, nfs.guide, optimizer, loss=elbo)
    losses = train(train_param["steps"])

    # outputs final variational parameters
    kappa_1, kappa_2, gamma_1, gamma_2, delta_1, delta_2, tau, sigma, q_theta_dist, phi = nfs.guide(data)

    # saves final variational parameters etc.
    # NOTE: uncomment to save 
    """
    torch.save(data, file_dir / experiments_dir / "data.pt")
    torch.save(flow_params, file_dir / experiments_dir / "flow_params.pt")

    torch.save(losses, file_dir / experiments_dir / "losses.pt")
    torch.save(map_losses, file_dir / experiments_dir / "map_losses.pt")
    torch.save(pretrain_losses, file_dir / experiments_dir / "pretrain_losses.pt")

    torch.save(kappa_1.detach(), file_dir / experiments_dir / "kappa_1.pt")
    torch.save(kappa_2.detach(), file_dir / experiments_dir / "kappa_2.pt")
    torch.save(gamma_1.detach(), file_dir / experiments_dir / "gamma_1.pt")
    torch.save(gamma_2.detach(), file_dir / experiments_dir / "gamma_2.pt")
    torch.save(delta_1.detach(), file_dir / experiments_dir / "delta_1.pt")
    torch.save(delta_2.detach(), file_dir / experiments_dir / "delta_2.pt")
    torch.save(tau.detach(), file_dir / experiments_dir / "tau.pt")
    torch.save(sigma.detach(), file_dir / experiments_dir / "sigma.pt")
    torch.save(nfs.flows.state_dict(), file_dir / experiments_dir / "flows.pt")
    torch.save(phi.detach(), file_dir / experiments_dir / "phi.pt")
    """
