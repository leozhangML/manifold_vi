import pyro
from pyro.distributions                                         import MultivariateNormal, Beta, Categorical, InverseGamma, Gamma
from pyro.distributions                                         import Exponential, Uniform, Dirichlet, TransformedDistribution
from pyro.infer                                                 import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.infer.autoguide.guides                                import AutoDelta
from pyro                                                       import poutine

import torch
import torch.nn                                                 as nn
from torch.distributions                                        import constraints

import numpy                                                    as np
from tqdm                                                       import tqdm

from stiefel                                                    import Circular
from distributions                                              import UniformStiefel
from utils                                                      import mix_weights, angle_to_matrix


@config_enumerate
def map_model(T, data):
    """
    Pyro model for the MAP estimates model
    """
    
    N, ndim = data.shape

    beta = pyro.sample("beta", Beta(1, torch.ones([T-1])).to_event())
    beta_weights = mix_weights(beta)

    with pyro.plate("b_lamb_plate", ndim):
        b = pyro.sample("b", Exponential(1))
        lamb = pyro.sample("lamb", InverseGamma(1, b))
    diag_lamb = torch.diag_embed(lamb).unsqueeze(-3)

    with pyro.plate("mu_theta_plate", T):
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros([ndim]), torch.eye(ndim)))
        theta = pyro.sample('theta', UniformStiefel(T, ndim))

    with pyro.plate("data_plate", N):
        z = pyro.sample("z", Categorical(beta_weights), infer={'enumerate': 'parallel'})
        mu_z = mu[z]
        orth_mats = angle_to_matrix(theta, ndim)
        cov = orth_mats @ diag_lamb @ orth_mats.transpose(-1, -2)
        cov_z = cov[z]
        obs = pyro.sample("obs", MultivariateNormal(mu_z, cov_z), obs=data) 

def train_map_model(T, data, map_params):
    """
    Runs training for the MAP estimates
    """

    ndim = data.shape[-1]
    stiefel_dim = ndim * (ndim - 1) // 2

    def init_loc_fn(site):
        if site["name"] == "mu":
            rand_ind = torch.multinomial(torch.ones([len(data)]), T)
            return data[rand_ind]
        if site["name"] == "theta":
            init_theta_dist = Uniform(torch.ones([T, stiefel_dim]) * -np.pi,
                                      torch.ones([T, stiefel_dim]) * np.pi)
            return init_theta_dist.sample()
        if site["name"] == "lamb":
            init_lamb_dist = InverseGamma(map_params["lamb_init"][0], 
                                          map_params["lamb_init"][1])
            return init_lamb_dist.sample()

    pyro.clear_param_store()
    optim = pyro.optim.ClippedAdam({"lr": map_params["lr"], "betas": map_params["betas"]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    global_guide = AutoDelta(poutine.block(map_model, hide=["z"]), init_loc_fn = init_loc_fn)
    svi = SVI(map_model, global_guide, optim, loss=elbo)

    num_iterations = map_params["steps"]
    losses = []
    for _ in tqdm(range(num_iterations)):
        loss = svi.step(T, data)
        losses.append(loss)

    beta = pyro.param("AutoDelta.beta").detach()
    b = pyro.param("AutoDelta.b").detach()
    lamb = pyro.param("AutoDelta.lamb").detach()
    mu = pyro.param("AutoDelta.mu").detach()
    theta = pyro.param("AutoDelta.theta").detach()

    z_marginals = TraceEnum_ELBO().compute_marginals(map_model, global_guide, *[T, data])
    labels = torch.exp(z_marginals["z"].logits).detach()

    pyro.clear_param_store()
    map_init = {"beta": beta, "b": b, "lamb": lamb, "mu": mu, "theta": theta, "labels": labels}

    return losses, map_init


class PretrainNF(nn.Module):
    """
    Pretrains non-joint NF from the MAP estimates
    """

    def __init__(self, ndim, T, flow_params, map_init):
    
        super().__init__()

        # set up flows
        stiefel_dim = ndim * (ndim - 1) // 2
        num_trans = flow_params["num_trans"]

        flows = [Circular(num_input_channels=stiefel_dim,
                          context_features=T,
                          **flow_params) for _ in range(num_trans)]

        self.ndim = ndim
        self.stiefel_dim = stiefel_dim
        self.T = T
        self.flows = nn.ModuleList(flows)
        self.num_trans = num_trans
        self.map_init = map_init

    @config_enumerate
    def model(self, data):

        N = len(data)

        with pyro.plate("theta_plate", self.T):
            theta = pyro.sample('theta', UniformStiefel(self.T, self.ndim))

        mu = torch.broadcast_to(self.map_init["mu"], [*theta.shape[:-1], self.ndim])
        diag_lamb = torch.diag_embed(self.map_init["lamb"]).unsqueeze(-3)
        orth_mats = angle_to_matrix(theta, self.ndim)
        cov = orth_mats @ diag_lamb @ orth_mats.transpose(-1, -2)
        z = torch.argmax(self.map_init["labels"], dim=-1)  # need the correct labels for databatch

        with pyro.plate("obs_plate", N):
            if theta.dim() < 3:
                mu_z = mu[z]
                cov_z = cov[z]
                obs = pyro.sample("obs", MultivariateNormal(mu_z, cov_z), obs=data)  
            else:
                mu_z = mu[torch.arange(mu.shape[0]).unsqueeze(-1), z]  # [num_particles, N, ndim]
                cov_z = cov[torch.arange(cov.shape[0]).unsqueeze(-1), z]
                obs = pyro.sample("obs", MultivariateNormal(mu_z, cov_z), obs=data)

    @config_enumerate
    def guide(self, data):

        # register all nf params with pyro
        for i in range(len(self.flows)):
            pyro.module(str(i), self.flows[i].mprqat.autoregressive_net)

        with pyro.plate("theta_plate", self.T):
            q_theta_base = Uniform(torch.ones([self.T, self.stiefel_dim]) * -np.pi,
                                torch.ones([self.T, self.stiefel_dim]) * np.pi).to_event(1)
            q_theta_dist = TransformedDistribution(q_theta_base, list(self.flows))
            q_theta = pyro.sample("theta", q_theta_dist)

        return q_theta_dist
