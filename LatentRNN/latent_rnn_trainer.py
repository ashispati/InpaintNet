from torch import nn

from utils.trainer import Trainer
from LatentRNN.latent_rnn import LatentRNN
from MeasureVAE.vae_tester import *


class LatentRNNTrainer(Trainer):
    def __init__(self, dataset,
                 model: LatentRNN,
                 lr=1e-4,
                 early_stopping=False):
        """
        Initializes the LatentRNNTrainer class
        :param dataset: initialized FolkDatasetNBars object
        :param model: initialized LatentRNN object
        """
        super(LatentRNNTrainer, self).__init__(dataset, model, lr, early_stopping)
        self.min_num_measures_target = 2
        self.max_num_measure_target = 6
        assert(self.max_num_measure_target >= self.min_num_measures_target)
        assert(self.dataset.n_bars > self.min_num_measures_target)
        assert(self.dataset.n_bars > self.max_num_measure_target)
        self.measure_seq_len = self.dataset.subdivision * self.dataset.num_beats_per_bar

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, _ = batch
        batch_data = self.split_score_stochastic(score_tensor)
        return batch_data

    def loss_and_acc_for_batch(self, batch, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        :param batch: tuple of Torch Variable objects
        :param epoch_num: int, used to change training schedule
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        # extract data
        tensor_past, tensor_future, tensor_target = batch
        num_measures_past = tensor_past.size(1)
        num_measures_future = tensor_future.size(1)

        # perform forward pass of model
        weights, pred, _ = self.model(
            past_context=tensor_past,
            future_context=tensor_future,
            target=tensor_target,
            measures_to_generate=self.dataset.n_bars - num_measures_past - num_measures_future,
            train=train
        )
        # compute loss
        loss = self.mean_crossentropy_loss_alt(
            weights=weights,
            targets=tensor_target
        )
        # compute accuracy
        accuracy = self.mean_accuracy_alt(
            weights=weights,
            targets=tensor_target
        )
        return loss, accuracy

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        # Nothing to do here
        return

    def split_score_stochastic(self, score_tensor, extra_outs=False, fix_num_target=None):
        """
        Splits the score tensor into past context, future context and target
        :param score_tensor: torch Tensor,
                (batch_size, 1, seq_len)
        :param extra_outs: bool, output additional items if True
        :param fix_num_target: int, fixes the number of target measures
        :return: tuple of 3 torch Variable objects
                score_past: torch Variable, past context
                            (batch_size, num_measures_past, measure_seq_len)
                score_future: torch Variable, future context
                            (batch_size, num_measures_future, measure_seq_len)
                score_target: torch Variable, measures to be predicted
                            (batch_size, num_of_target_measures * measure_seq_len)
        """
        measures_tensor = LatentRNNTrainer.split_to_measures(
            score_tensor=score_tensor,
            measure_seq_len=self.measure_seq_len
        )
        num_measures = measures_tensor.size(1)
        assert (num_measures == self.dataset.n_bars)
        if fix_num_target is None:
            num_target = int(
                torch.randint(
                    low=self.min_num_measures_target,
                    high=self.max_num_measure_target + 1,
                    # high=num_measures - self.min_num_measures_past - self.min_num_measures_future,
                    size=(1,)
                ).item()
            )
        else:
            num_target = fix_num_target
        num_past = int(
            torch.randint(
                # low=self.min_num_measures_past,
                # high=num_measures - num_target - self.min_num_measures_future,
                low=1,
                high=num_measures - num_target - 1,
                size=(1,)
            ).item()
        )
        num_future = num_measures - num_past - num_target
        # assert (num_future >= self.min_num_measures_future)

        tensor_past, tensor_future, tensor_target = LatentRNNTrainer.split_score(
            score_tensor=score_tensor,
            num_past=num_past,
            num_future=num_future,
            num_target=num_target,
            measure_seq_len=self.measure_seq_len
        )

        if extra_outs:
            return tensor_past, tensor_future, tensor_target, num_past, num_target
        else:
            return tensor_past, tensor_future, tensor_target

    @staticmethod
    def split_score(score_tensor, num_past, num_future, num_target, measure_seq_len):
        """
        Splits the score based on provided number of measures
        :param score_tensor:
        :param num_past: number of measures for past context
        :param num_future: number of measures for future context
        :param num_target: number of measures to predict
        :param measure_seq_len: int, number of ticks in a measure
        :return:
        """
        # split score tensor to measures
        measures_tensor = LatentRNNTrainer.split_to_measures(score_tensor, measure_seq_len)
        num_measures = measures_tensor.size(1)
        # sanity check
        assert(num_measures == num_past + num_future + num_target)
        # split
        tensor_past = to_cuda_variable_long(
            measures_tensor[:, 0:num_past, :]
        )
        tensor_future = to_cuda_variable_long(
            measures_tensor[:, num_measures - num_future:, :]
        )
        tensor_target = to_cuda_variable_long(
            measures_tensor[:, num_past:num_measures - num_future, :]
        )
        return tensor_past, tensor_future, tensor_target

    @staticmethod
    def split_to_measures(score_tensor, measure_seq_len):
        """
        Splits the input tensor to segments containing individual measures
        :param score_tensor: torch Variable,
                (batch_size, 1 , seq_len)
        :param measure_seq_len: int, number of ticks in a measure
        :return: tensor containing individual measure, torch Variable
                (batch_size, num_measures, self.measure_seq_len)
        """
        batch_size, _, seq_len = score_tensor.size()
        if seq_len % measure_seq_len != 0:
            raise ValueError
        measures_tensor = score_tensor.view(batch_size, -1, measure_seq_len)
        return measures_tensor
