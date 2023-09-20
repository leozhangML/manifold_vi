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


class PostPredMap():
    """
    For run_mean_field_map.py
    """


    def __init__(self, mu, theta, beta, lamb):

        ndim = len(lamb)
        stiefel_dim = ndim * (ndim - 1) // 2
        N, T = mu.shape

        beta_weights = mix_weights(beta)
        z_dist = Categorical(beta_weights) 

        self.ndim = ndim
        self.stiefel_dim = stiefel_dim
        self.T = T

        self.mu = mu
        self.theta = theta
        self.beta = beta
        self.lamb = lamb

        self.z_dist = z_dist

    def sample(self, num):
        """
        For each sample, we sample from the posterior
        and sample obs from the mixture given by the betas etc.
        """

        diag_lamb = torch.diag_embed(self.lamb)
        z = self.z_dist.sample([num])

        mu_z = self.mu[z]
        O_z = angle_to_matrix(self.theta[z], self.ndim)
        cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)

        obs_dist = MultivariateNormal(mu_z, cov_z)
        obs = obs_dist.sample()

        return obs


def show_mix_map(data, postpredmap, k=None, n=100):
    """
    For run_mean_field_map.py
    """

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for index in range(6):

        lamb = postpredmap.lamb

        mu = postpredmap.mu  # [T, ndim]

        theta = postpredmap.theta  # [T, stiefel_dim]

        diag_lamb = torch.diag_embed(lamb)

        O_z = angle_to_matrix(theta, postpredmap.ndim)  # [T, ndim, ndim]
        cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)
        obs = MultivariateNormal(mu, cov_z).sample([n])

        if k is None:
            obs = obs.view(-1, postpredmap.ndim)  # [n, T, ndim]
            alpha_obs = 0.1
        else:
            obs = obs[:, k]
            alpha_obs = 1

        i = index // 3
        j = index % 3

        axes[i, j].scatter(data[:, 0], data[:, 1], alpha=0.2)
        axes[i, j].scatter(obs[:, 0], obs[:, 1], alpha=alpha_obs)
        axes[i, j].scatter(mu[:, 0], mu[:, 1], marker="x")


class PostPredVM():
    """
    With sigma
    """

    def __init__(self, kappa_1, kappa_2, gamma_1, gamma_2, tau, sigma, omega_1, omega_2, phi, map_init=None):

        ndim = len(gamma_1)
        stiefel_dim = ndim * (ndim - 1) // 2
        N, T = phi.shape

        # posterior distributions
        if map_init is None:
            lamb_dist = InverseGamma(gamma_1.detach(), gamma_2.detach()).to_event()  # [ndim]
        else:
            lamb = map_init["lamb"]
        beta_dist = Beta(kappa_1.detach(), kappa_2.detach()).to_event()  # [T-1]
        theta_dist = VonMises(omega_1, omega_2)
        mu_dist = MultivariateNormal(tau.detach(), sigma.detach() * torch.eye(ndim)).to_event()  # [T, ndim]
        z_dist = Categorical(phi.detach()).to_event()  # [N]

        self.ndim = ndim
        self.stiefel_dim = stiefel_dim
        self.T = T

        if map_init is None:
            self.lamb_dist = lamb_dist
        else:
            self.lamb = lamb
        self.beta_dist = beta_dist
        self.mu_dist = mu_dist
        self.theta_dist = theta_dist
        self.z_dist = z_dist

        self.map_init = map_init

    def sample(self, num):
        """
        For each sample, we sample from the posterior
        and sample obs from the mixture given by the betas etc.
        """

        if self.map_init is None:
            lamb = self.lamb_dist.sample(torch.Size([num]))
        else:
            lamb = torch.broadcast_to(self.lamb, [num, self.ndim])

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


class PostPredPre():
 
    def __init__(self, mu, beta, lamb, pre_flows):

        ndim = len(lamb)
        stiefel_dim = ndim * (ndim - 1) // 2
        T = len(mu)

        beta_weights = mix_weights(beta)
        z_dist = Categorical(beta_weights) 

        self.ndim = ndim
        self.stiefel_dim = stiefel_dim
        self.T = T

        base = Uniform(torch.ones([self.T, self.stiefel_dim]) * -np.pi,
                       torch.ones([self.T, self.stiefel_dim]) * np.pi).to_event(1)
        theta_dist = dist.TransformedDistribution(base, list(pre_flows))

        self.mu = mu
        self.theta_dist = theta_dist
        self.beta = beta
        self.lamb = lamb

        self.z_dist = z_dist

    def sample(self, num):
        """
        For each sample, we sample from the posterior
        and sample obs from the mixture given by the betas etc.
        """

        obs_list = []
        diag_lamb = torch.diag_embed(self.lamb)

        for i in range(num):
            z = self.z_dist.sample()
            theta = self.theta_dist.sample()
            mu_z = self.mu[z]
            O_z = angle_to_matrix(theta[z].view(1, -1), self.ndim).view(self.ndim, self.ndim)
            cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)
            obs_dist = MultivariateNormal(mu_z, cov_z)
            obs = obs_dist.sample()
            obs_list.append(obs)

        return torch.stack(obs_list, dim=0)


def show_mix_pre(data, prepred, k=None, n=100):
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for index in range(6):

        lamb = prepred.lamb

        mu = prepred.mu  # [T, ndim]

        theta = prepred.theta_dist.sample()  # [T, stiefel_dim]

        diag_lamb = torch.diag_embed(lamb)

        O_z = angle_to_matrix(theta, prepred.ndim)  # [T, ndim, ndim]
        cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)
        obs = MultivariateNormal(mu, cov_z).sample([n])

        if k is None:
            obs = obs.view(-1, prepred.ndim)  # [n, T, ndim]
            alpha_obs = 0.1
        else:
            obs = obs[:, k]
            alpha_obs = 1

        i = index // 3
        j = index % 3

        axes[i, j].scatter(data[:, 0], data[:, 1], alpha=0.2)
        axes[i, j].scatter(obs[:, 0], obs[:, 1], alpha=alpha_obs)
        axes[i, j].scatter(mu[:, 0], mu[:, 1], marker="x")


class PostPredPreJoint():
 
    def __init__(self, beta, lamb, pre_flows):

        ndim = len(lamb)
        stiefel_dim = ndim * (ndim - 1) // 2
        T = pre_flows[0].context_features

        beta_weights = mix_weights(beta)
        z_dist = Categorical(beta_weights) 

        self.ndim = ndim
        self.stiefel_dim = stiefel_dim
        self.T = T

        mu_theta_base = BaseGaussianUniform(T, ndim)
        mu_theta_dist = TransformedDistribution(mu_theta_base, list(pre_flows))

        self.beta = beta
        self.lamb = lamb
        self.mu_theta_dist = mu_theta_dist
        self.z_dist = z_dist

    def sample(self, num):
        """
        For each sample, we sample from the posterior
        and sample obs from the mixture given by the betas etc.
        """

        obs_list = []
        diag_lamb = torch.diag_embed(self.lamb)

        for i in range(num):
            z = self.z_dist.sample()
            mu_theta = self.mu_theta_dist.sample()
            mu = mu_theta[:, :self.ndim]
            theta = mu_theta[:, self.ndim:]

            mu_z = mu[z]
            O_z = angle_to_matrix(theta[z].view(1, -1), self.ndim).view(self.ndim, self.ndim)
            cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)
            obs_dist = MultivariateNormal(mu_z, cov_z)
            obs = obs_dist.sample()
            obs_list.append(obs)

        return torch.stack(obs_list, dim=0)


def show_mix_pre_joint(data, prepred, k=None, n=100):
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for index in range(6):

        lamb = prepred.lamb
        diag_lamb = torch.diag_embed(lamb)

        mu_theta = prepred.mu_theta_dist.sample()
        mu = mu_theta[:, :prepred.ndim]
        theta = mu_theta[:, prepred.ndim:]

        O_z = angle_to_matrix(theta, prepred.ndim)  # [T, ndim, ndim]
        cov_z = O_z @ diag_lamb @ O_z.transpose(-1, -2)
        obs = MultivariateNormal(mu, cov_z).sample([n])

        if k is None:
            obs = obs.view(-1, prepred.ndim)  # [n, T, ndim]
            alpha_obs = 0.1
        else:
            obs = obs[:, k]
            alpha_obs = 1

        i = index // 3
        j = index % 3

        axes[i, j].scatter(data[:, 0], data[:, 1], alpha=0.2)
        axes[i, j].scatter(obs[:, 0], obs[:, 1], alpha=alpha_obs)
        axes[i, j].scatter(mu[:, 0], mu[:, 1], marker="x")
