import pyro
import pyro.distributions                                       as dist
from pyro.distributions                                         import MultivariateNormal, Beta, Categorical, InverseGamma, Gamma, Delta
from pyro.distributions                                         import Exponential, TransformedDistribution, Uniform, Dirichlet, VonMises
from pyro.infer                                                 import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from torch.distributions                                        import constraints

import torch
import numpy                                                    as np
import matplotlib.pyplot                                        as plt


from utils                                                      import angle_to_matrix, mix_weights


class PostPredSigma():
    """
    With sigma
    """

    def __init__(self, kappa_1, kappa_2, gamma, tau, sigma, flows, phi):

        ndim = len(gamma)
        stiefel_dim = ndim * (ndim - 1) // 2
        N, T = phi.shape

        # posterior distributions
        lamb_dist = InverseGamma(1+N/2, gamma.detach()).to_event()  # [ndim]
        beta_dist = Beta(1+kappa_1.detach(), 1+kappa_2.detach()).to_event()  # [T-1]
        theta_base = Uniform(torch.ones([T, stiefel_dim]) * -np.pi, 
                            torch.ones([T, stiefel_dim]) * np.pi).to_event(1)
        theta_dist = dist.TransformedDistribution(theta_base, flows)  # [T, stiefel_dim]
        mu_dist = MultivariateNormal(tau.detach(), sigma.detach() * torch.eye(ndim)).to_event()  # [T, ndim]
        z_dist = Categorical(phi.detach()).to_event()  # [N]

        self.ndim = ndim
        self.stiefel_dim = stiefel_dim
        self.T = T

        self.lamb_dist = lamb_dist
        self.beta_dist = beta_dist
        self.mu_dist = mu_dist
        self.theta_dist = theta_dist
        self.z_dist = z_dist

        self.flows = flows

    def sample(self, num):
        """
        For each sample, we sample from the posterior
        and sample obs from the mixture given by the betas etc.
        """

        lamb = self.lamb_dist.sample(torch.Size([num]))
        beta = self.beta_dist.sample(torch.Size([num]))  # [num, T-1]
        mu = self.mu_dist.sample(torch.Size([num]))  # [num, T, ndim]
        theta = self.theta_dist.sample(torch.Size([num])) # [num, T, stiefel_dim]

        beta_weights = mix_weights(beta)  # [num, T]
        diag_lamb = torch.diag_embed(lamb)

        labels = Categorical(beta_weights).sample()  # [num]

        mu_z = mu[torch.arange(num), labels]  # [num, ndim]
        O_z = angle_to_matrix(theta[torch.arange(num), labels], self.ndim)  # [num, ndim, ndim]
        cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)

        obs_dist = MultivariateNormal(mu_z, cov_z)  # [num, ndim]
        obs = obs_dist.sample()

        return obs, lamb, beta, mu, theta, beta_weights, labels, mu_z, cov_z
    

class PostPred():

    def __init__(self, kappa_1, kappa_2, gamma_1, gamma_2, tau, sigma, flows, phi):

        ndim = len(gamma_1)
        stiefel_dim = ndim * (ndim - 1) // 2
        N, T = phi.shape

        # posterior distributions
        lamb_dist = InverseGamma(gamma_1.detach(), gamma_2.detach()).to_event()  # [ndim]
        beta_dist = Beta(kappa_1.detach(), kappa_2.detach()).to_event()  # [T-1]
        theta_base = Uniform(torch.ones([T, stiefel_dim]) * -np.pi, 
                            torch.ones([T, stiefel_dim]) * np.pi).to_event(1)
        theta_dist = dist.TransformedDistribution(theta_base, flows)  # [T, stiefel_dim]
        mu_dist = MultivariateNormal(tau.detach(), sigma.detach() * torch.eye(ndim)).to_event()  # [T, ndim]
        z_dist = Categorical(phi.detach()).to_event()  # [N]

        self.ndim = ndim
        self.stiefel_dim = stiefel_dim
        self.T = T

        self.lamb_dist = lamb_dist
        self.beta_dist = beta_dist
        self.mu_dist = mu_dist
        self.theta_dist = theta_dist
        self.z_dist = z_dist

        self.flows = flows

    def sample(self, num):
        """
        For each sample, we sample from the posterior
        and sample obs from the mixture given by the betas etc.
        """

        lamb = self.lamb_dist.sample(torch.Size([num]))
        beta = self.beta_dist.sample(torch.Size([num]))  # [num, T-1]
        mu = self.mu_dist.sample(torch.Size([num]))  # [num, T, ndim]
        theta = self.theta_dist.sample(torch.Size([num])) # [num, T, stiefel_dim]

        beta_weights = mix_weights(beta)  # [num, T]
        diag_lamb = torch.diag_embed(lamb)

        labels = Categorical(beta_weights).sample()  # [num]

        mu_z = mu[torch.arange(num), labels]  # [num, ndim]
        O_z = angle_to_matrix(theta[torch.arange(num), labels], self.ndim)  # [num, ndim, ndim]
        cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)

        obs_dist = MultivariateNormal(mu_z, cov_z)  # [num, ndim]
        obs = obs_dist.sample()

        return obs


def show_mix(data, postpred, tau, k=None, n=100):
    """
    For a given sample of lambda, O, mu from from the hyperprior
    and the DP, we plot each observation cluster
    """
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for index in range(6):

        lamb = postpred.lamb_dist.sample()

        mu = postpred.mu_dist.sample()  # [T, ndim]

        theta = postpred.theta_dist.sample()  # [T, stiefel_dim]

        diag_lamb = torch.diag_embed(lamb)

        O_z = angle_to_matrix(theta, postpred.ndim)  # [T, ndim, ndim]
        cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)
        obs = MultivariateNormal(mu, cov_z).sample([n])

        if k is None:
            obs = obs.view(-1, postpred.ndim)  # [n, T, ndim]
            alpha_obs = 0.1
        else:
            obs = obs[:, k]
            alpha_obs = 1

        i = index // 3
        j = index % 3

        axes[i, j].scatter(data[:, 0], data[:, 1], alpha=0.2)
        axes[i, j].scatter(obs[:, 0], obs[:, 1], alpha=alpha_obs)
        axes[i, j].scatter(mu[:, 0], mu[:, 1], marker="x")
        axes[i, j].scatter(tau.detach()[:, 0], tau.detach()[:, 1], marker=",")

def show_mix_map3D(data, postpred, k=None, n=100):
    """
    For run_mean_field_map.py
    """

    #fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    lamb = postpred.lamb_dist.sample()

    mu = postpred.mu_dist.sample()  # [T, ndim]

    theta = postpred.theta_dist.sample()  # [T, stiefel_dim]

    diag_lamb = torch.diag_embed(lamb)

    O_z = angle_to_matrix(theta, postpred.ndim)  # [T, ndim, ndim]
    cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)
    obs = MultivariateNormal(mu, cov_z).sample([n])

    if k is None:
        obs = obs.view(-1, postpred.ndim)
        alpha_obs = 0.1
    else:
        obs = obs[:, k]
        alpha_obs = 0.8

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.1)
    ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2])
    ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], marker="x")