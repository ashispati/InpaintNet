import os
from tqdm import tqdm
from random import randint
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from DatasetManager.the_session.folk_dataset import FolkMeasuresDataset, FolkDatasetNBars
from DatasetManager.helpers import START_SYMBOL, END_SYMBOL

from MeasureVAE.measure_vae import MeasureVAE
from MeasureVAE.vae_trainer import VAETrainer
from utils.helpers import *


class VAETester(object):
    def __init__(self, dataset, model: MeasureVAE):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.filepath = os.path.join('models/',
                                     self.model.__repr__())

        self.decoder = self.model.decoder
        # freeze decoder
        self.train = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.z_dim = self.decoder.z_dim
        self.batch_size = 1
        self.measure_seq_len = 24

    def test_model(self):
        """
        Runs the model on the test set
        :return: tuple: mean_loss, mean_accuracy
        """
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=64,  # TODO: remove this hard coding
            split=(0.01, 0.01)
        )
        print('Num Test Batches: ', len(gen_test))
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print('Test Epoch:')
        print(
            '\tTest Loss: ', mean_loss_test, '\n'
            '\tTest Accuracy: ', mean_accuracy_test * 100
        )

    def test_interp(self):
        """
        Tests the interpolation capabilities of the latent space
        :return: None
        """
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=1,  # TODO: remove this hard coding
            split=(0.01, 0.5)
        )
        gen_it_test = gen_test.__iter__()
        for _ in range(randint(0, len(gen_test))):
            tensor_score1, _ = next(gen_it_test)

        gen_it_val = gen_val.__iter__()
        for _ in range(randint(0, len(gen_val))):
            tensor_score2, _ = next(gen_it_val)

        tensor_score1 = to_cuda_variable(tensor_score1.long())
        tensor_score2 = to_cuda_variable(tensor_score2.long())
        self.test_interpolation(tensor_score1, tensor_score2, 10)

    def decode_mid_point(self, z1, z2, n):
        """
        Decodes the mid-point of two latent vectors
        :param z1: torch tensor, (1, self.z_dim)
        :param z2: torch tensor, (1, self.z_dim)
        :param n: int, number of points for interpolation
        :return: torch tensor, (1, (n+2) * measure_seq_len)
        """
        assert(n >= 1 and isinstance(n, int))
        # compute the score_tensors for z1 and z2
        dummy_score_tensor = to_cuda_variable(torch.zeros(self.batch_size, self.measure_seq_len))
        _, sam1 = self.decoder(z1, dummy_score_tensor, self.train)
        _, sam2 = self.decoder(z2, dummy_score_tensor, self.train)
        # find the interpolation points and run through decoder
        tensor_score = sam1
        for i in range(n):
            z_interp = z1 + (z2 - z1)*(i+1)/(n+1)
            _, sam_interp = self.decoder(z_interp, dummy_score_tensor, self.train)
            tensor_score = torch.cat((tensor_score, sam_interp), 1)
        tensor_score = torch.cat((tensor_score, sam2), 1).view(1, -1)
        #score = self.dataset.tensor_to_score(tensor_score.cpu())
        return tensor_score

    def test_interpolation(self, tensor_score1, tensor_score2, n=1):
        """
        Tests the interpolation in the latent space for two random points in the
        validation and test set
        :param tensor_score1: torch tensor, (1, measure_seq_len)
        :param tensor_score2: torch tensor, (1, measure_seq_len)
        :param n: int, number of points for interpolation
        :return:
        """
        z_dist1 = self.model.encoder(tensor_score1)
        z_dist2 = self.model.encoder(tensor_score2)
        z1 = z_dist1.loc
        z2 = z_dist2.loc
        tensor_score = self.decode_mid_point(z1, z2, n)
        #tensor_score = torch.cat((tensor_score1, tensor_score, tensor_score2), 1)
        score = self.dataset.tensor_to_score(tensor_score.cpu())
        score.show()
        return score

    def loss_and_acc_test(self, data_loader):
        """
        Computes loss and accuracy for test data
        :param data_loader: torch data loader object
        :return: (float, float)
        """
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, (score_tensor, metadata_tensor) in tqdm(enumerate(data_loader)):
            if isinstance(self.dataset, FolkDatasetNBars):
                batch_size = score_tensor.size(0)
                score_tensor1 = score_tensor.view(batch_size, self.dataset.n_bars, -1)
                score_tensor = score_tensor1.view(batch_size * self.dataset.n_bars, -1)
            # convert input to torch Variables
            score_tensor = to_cuda_variable_long(score_tensor)

            # compute forward pass
            weights, samples, _, _, _, _ = self.model(
                measure_score_tensor=score_tensor,
                train=False
            )

            # compute loss
            recons_loss = VAETrainer.mean_crossentropy_loss(
                weights=weights,
                targets=score_tensor
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = VAETrainer.mean_accuracy(
                weights=weights,
                targets=score_tensor
            )
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def loss_and_acc_test_alt(self, data_loader):
        """
        :param data_loader: torch data loader object
        :return: (float, float)
        """
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, (score_tensor, metadata_tensor) in tqdm(enumerate(data_loader)):
            if isinstance(self.dataset, FolkDatasetNBars):
                batch_size = score_tensor.size(0)
                score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
            score_tensor = to_cuda_variable_long(score_tensor)

            # compute forward pass
            weights, samples = self.model.forward_test(
                measure_score_tensor=score_tensor
            )

            # compute loss
            recons_loss = VAETrainer.mean_crossentropy_loss_alt(
                weights=weights,
                targets=score_tensor
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = VAETrainer.mean_accuracy_alt(
                weights=weights,
                targets=score_tensor
            )
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def plot_attribute_dist(self, attribute='num_notes', plt_type='pca'):
        """
        Plots the distribution of a particular attribute in the latent space
        :param attribute: str,
                num_notes, note_range, rhy_entropy, beat_strength
        :param plt_type: str, 'tsne' or 'pca'
        :return:
        """
        (_, _, gen_test) = self.dataset.data_loaders(
            batch_size=64,  # TODO: remove this hard coding
            split=(0.70, 0.20)
        )
        z_all = []
        n_all = []
        num_samples = 5
        for sample_id, (score_tensor, _) in tqdm(enumerate(gen_test)):
            # convert input to torch Variables
            if isinstance(self.dataset, FolkDatasetNBars):
                batch_size = score_tensor.size(0)
                score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
                score_tensor = score_tensor.view(batch_size * self.dataset.n_bars, -1)
            score_tensor = to_cuda_variable_long(score_tensor)
            # compute encoder forward pass
            z_dist = self.model.encoder(score_tensor)
            z_tilde = z_dist.loc
            z_all.append(z_tilde)
            if attribute == 'num_notes':
                attr = self.dataset.get_num_notes_in_measure(score_tensor)
            elif attribute == 'note_range':
                attr = self.dataset.get_note_range_of_measure(score_tensor)
            elif attribute == 'rhy_entropy':
                attr = self.dataset.get_rhythmic_entropy(score_tensor)
            elif attribute == 'beat_strength':
                attr = self.dataset.get_beat_strength(score_tensor)
            else:
                raise ValueError('Invalid attribute type')
            for i in range(attr.size(0)):
                tensor_score = score_tensor[i, :]
                start_idx = self.dataset.note2index_dicts[self.dataset.NOTES][START_SYMBOL]
                end_idx = self.dataset.note2index_dicts[self.dataset.NOTES][END_SYMBOL]
                if tensor_score[0] == start_idx:
                    attr[i] = -0.1
                elif tensor_score[0] == end_idx:
                    attr[i] = -0.2
            n_all.append(attr)
            if sample_id == num_samples:
                break
        z_all = torch.cat(z_all, 0)
        n_all = torch.cat(n_all, 0)
        z_all = to_numpy(z_all)
        n_all = to_numpy(n_all)

        filename = 'plots/' + plt_type + '_' + attribute + '_' + str(num_samples) + '_measure_vae.png'
        if plt_type == 'pca':
            self.plot_pca(z_all, n_all, filename)
        elif plt_type == 'tsne':
            self.plot_tsne(z_all, n_all, filename)
        else:
            raise ValueError('Invalid plot type')

    def plot_transposition_points(self, plt_type='pca'):
        """
        Plots a t-SNE plot for data-points comprising of transposed measures
        :param plt_type: str, 'tsne' or 'pca'
        :return:
        """
        score_gen = self.dataset.iterator_gen().__iter__()
        for _ in range(1): #(randint(0, 100)):
            original_score = next(score_gen)
        possible_transpositions = self.dataset.all_transposition_intervals(original_score)
        z_all = []
        n_all = []
        n = 0
        for trans_int in possible_transpositions:
            score_tensor, _ = self.dataset.transposed_score_and_metadata_tensors(
                original_score,
                trans_int
            )
            score_tensor = self.dataset.split_score_tensor_to_measures(score_tensor)
            score_tensor = to_cuda_variable_long(score_tensor)
            z_dist = self.model.encoder(score_tensor)
            z_tilde = z_dist.loc
            z_all.append(z_tilde)
            t = np.arange(0, z_tilde.size(0))
            n_all.append(torch.from_numpy(t))
            #n_all.append(torch.ones(z_tilde.size(0)) * n)
            n += 1
        print(n)
        z_all = torch.cat(z_all, 0)
        n_all = torch.cat(n_all, 0)
        z_all = to_numpy(z_all)
        n_all = to_numpy(n_all)

        filename = 'plots/' + plt_type + '_transposition_measure_vae.png'
        if plt_type == 'pca':
            self.plot_pca(z_all, n_all, filename)
        elif plt_type == 'tsne':
            self.plot_tsne(z_all, n_all, filename)
        else:
            raise ValueError('Invalid plot type')

    @staticmethod
    def plot_pca(data, target, filename):
        pca = PCA(n_components=2, whiten=False)
        pca.fit(data)
        pca_z = pca.transform(data)
        plt.scatter(
            x=pca_z[:, 0],
            y=pca_z[:, 1],
            c=target,
            cmap='viridis',
            alpha=0.3
        )
        plt.colorbar()
        plt.savefig(filename, format='png', dpi=300)
        plt.show()

    @staticmethod
    def plot_tsne(data, target, filename):
        tsne = TSNE(n_components=2, verbose=1., perplexity=40, n_iter=300)
        tsne_z = tsne.fit_transform(data)
        plt.scatter(
            x=tsne_z[:, 0],
            y=tsne_z[:, 1],
            c=target,
            cmap="viridis",
            alpha=0.3
        )
        plt.colorbar()
        plt.savefig(filename, format='png', dpi=300)
        plt.show()


    @staticmethod
    def get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)

