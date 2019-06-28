from torch import nn, distributions
from tqdm import tqdm

from DatasetManager.the_session.folk_dataset import FolkMeasuresDataset, FolkDatasetNBars
from utils.helpers import *
from utils.trainer import Trainer
from MeasureVAE.measure_vae import MeasureVAE


class VAETrainer(Trainer):
    def __init__(self, dataset,
                 model: MeasureVAE,
                 lr=1e-4):
        super(VAETrainer, self).__init__(dataset, model, lr)

    def loss_and_acc_for_batch(self, batch, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        # extract data
        score = batch
        # perform forward pass of model
        weights, samples, z_dist, prior_dist, z_tilde, z_prior = self.model(
            measure_score_tensor=score,
            train=train
        )
        # compute loss
        recons_loss = self.mean_crossentropy_loss(weights=weights, targets=score)
        # dist_loss = self.compute_mmd_loss(z_tilde, z_prior)
        dist_loss = self.compute_kld_loss(z_dist, prior_dist)
        loss = recons_loss + dist_loss
        # compute accuracy
        accuracy = self.mean_accuracy(weights=weights,
                                      targets=score)
        return loss, accuracy

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, _ = batch
        if isinstance(self.dataset, FolkDatasetNBars):
            batch_size = score_tensor.size(0)
            score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
            score_tensor = score_tensor.view(batch_size * self.dataset.n_bars, -1)
        # convert input to torch Variables
        batch_data = to_cuda_variable_long(score_tensor)
        return batch_data

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        # Nothing to do here
        return

    @staticmethod
    def latent_loss(mu, sigma):
        """

        :param mu: torch Variable,
                    (batch_size, latent_space_dim)
        :param sigma: torch Variable,
                    (batch_size, latent_space_dim)
        :return: scalar, latent KL divergence loss
        """
        mean_sq = mu * mu
        sigma_sq = sigma * sigma
        ll = 0.5 * torch.mean(mean_sq + sigma_sq - torch.log(sigma_sq) - 1)
        return ll


    @staticmethod
    def compute_kernel(x, y, k):
        batch_size_x, dim_x = x.size()
        batch_size_y, dim_y = y.size()
        assert dim_x == dim_y

        xx = x.unsqueeze(1).expand(batch_size_x, batch_size_y, dim_x)
        yy = y.unsqueeze(0).expand(batch_size_x, batch_size_y, dim_y)
        distances = (xx - yy).pow(2).sum(2)
        return k(distances)

    @staticmethod
    def compute_mmd_loss(z_tilde, z_prior, coeff=10):
        """

        :param z_tilde:
        :param z_prior:
        :param coeff:
        :return:
        """
        # gaussian
        def gaussian(d, var=16.):
            return torch.exp(- d / var).sum(1).sum(0)

        # inverse multiquadratics
        def inverse_multiquadratics(d, var=16.):
            """
            :param d: (num_samples x, num_samples y)
            :param var:
            :return:
            """
            return (var / (var + d)).sum(1).sum(0)

        # k = inverse_multiquadratics
        k = gaussian
        batch_size = z_tilde.size(0)
        zp_ker = VAETrainer.compute_kernel(z_prior, z_prior, k)
        zt_ker = VAETrainer.compute_kernel(z_tilde, z_tilde, k)
        zp_zt_ker = VAETrainer.compute_kernel(z_prior, z_tilde, k)

        first_coeff = 1. / (batch_size * (batch_size - 1)) / 2 if batch_size > 1 else 1
        second_coeff = 2 / (batch_size * batch_size)
        mmd = coeff * (first_coeff * zp_ker
                       + first_coeff * zt_ker
                       - second_coeff * zp_zt_ker)
        return mmd

    @staticmethod
    def compute_kld_loss(z_dist, prior_dist, beta=0.001):
        """

        :param z_dist: torch.nn.distributions object
        :param prior_dist: torch.nn.distributions
        :param beta:
        :return: kl divergence loss
        """
        kld = distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = beta * kld.sum(1).mean()
        return kld
