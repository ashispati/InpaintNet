import random

from MusicInpainting.utils.auxiliary_loss_trainer import AuxiliaryLossTrainer
from MusicInpainting.AnticipationRNN.anticipation_rnn_gauss_reg_model import ConstraintModelGaussianReg
from MusicInpainting.LatentRNN.latent_rnn_trainer import *
from DatasetManager.the_session.folk_dataset import FolkDatasetNBars


class AnticipationRNNGaussianRegTrainer(Trainer):
    def __init__(self, dataset,
                 model,
                 lr=1e-4,
                 early_stopping=False):
        super(AnticipationRNNGaussianRegTrainer, self).__init__(dataset, model, lr, early_stopping)
        self.min_num_measures_target = 2
        self.max_num_measure_target = 6
        assert (self.max_num_measure_target >= self.min_num_measures_target)
        assert (self.dataset.n_bars > self.min_num_measures_target)
        assert (self.dataset.n_bars > self.max_num_measure_target)
        self.measure_seq_len = self.dataset.subdivision * self.dataset.num_beats_per_bar

    def loss_and_acc_for_batch(self, batch, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        score_tensor, metadata_tensor, constraints_loc, start_tick, end_tick = batch

        weights, _ = self.model(
            score_tensor=score_tensor,
            metadata_tensor=metadata_tensor,
            constraints_loc=constraints_loc,
            start_tick=start_tick,
            end_tick=end_tick,
            train=train
        )

        targets = score_tensor[:, :, (constraints_loc[0, 0, :] == 0).nonzero().squeeze()]
        targets = targets.transpose(0, 1)
        # weights = [weight_per_voice[:, start_tick:end_tick, :] for weight_per_voice in weights]

        loss = self.mean_crossentropy_loss(weights=weights, targets=targets)
        accuracy = self.mean_accuracy(weights=weights,
                                      targets=targets)

        return loss, accuracy

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, metadata_tensor = batch
        # find constraint locations stochastically
        constraint_loc, start_tick, end_tick = self.get_constraints_location(
            score_tensor
        )
        # convert input to torch Variables

        score_tensor = to_cuda_variable_long(score_tensor)
        metadata_tensor = to_cuda_variable_long(metadata_tensor)
        constraint_loc = to_cuda_variable_long(constraint_loc)
        return score_tensor, metadata_tensor, constraint_loc, start_tick, end_tick

    def get_num_target_stochastic(self):
        num_target = int(
            torch.randint(
                low=self.min_num_measures_target,
                high=self.max_num_measure_target + 1,
                # high=num_measures - self.min_num_measures_past - self.min_num_measures_future,
                size=(1,)
            ).item()
        )
        return num_target

    @staticmethod
    def get_num_past_stochastic(num_target, num_measures):
        num_past = int(
            torch.randint(
                # low=self.min_num_measures_past,
                # high=num_measures - num_target - self.min_num_measures_future,
                low=1,
                high=num_measures - num_target - 1,
                size=(1,)
            ).item()
        )
        return num_past

    def get_constraints_location(self, score_tensor, extra_outs=False, fix_num_target=None):
        """

        :param score_tensor: torch Tensor,
                (batch_size, 1, seq_len)
        :param extra_outs: bool, output additional items if True
        :param fix_num_target: int, fixes the number of target measures
        :return: tuple
                constraints_loc:
                start_tick: int,
                end_tick: int,
        """
        # get context boundaries
        measures_tensor = LatentRNNTrainer.split_to_measures(
            score_tensor=score_tensor,
            measure_seq_len=self.measure_seq_len
        )
        num_measures = measures_tensor.size(1)
        assert (num_measures == self.dataset.n_bars)
        if fix_num_target is None:
            num_target = self.get_num_target_stochastic()
        else:
            num_target = fix_num_target
        num_past = self.get_num_past_stochastic(num_target, num_measures)

        # extract tick locations
        start_measure = num_past + 1
        num_measures = num_target
        constraints_location = torch.zeros_like(score_tensor)
        start_tick = start_measure * self.measure_seq_len
        end_tick = start_tick + num_measures * self.measure_seq_len
        if start_tick > 0:
            constraints_location[:, :, :start_tick] = 1
        if end_tick < constraints_location.size(2) - 1:
            constraints_location[:, :, end_tick:] = 1
        return constraints_location, start_tick, end_tick

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        # Nothing to do here
        return

    @staticmethod
    def gaussian_regularization(activations):
        loss_mean = 0
        loss_var = 0
        for hids in activations:
            # hids.shape = [num_layers, batch_size, sequence_length, hidden units]
            num_layers, batch_size, sequence_length, hidden_units = hids.size()
            for hid_layer in hids:
                hid_layer = hid_layer.view(batch_size * sequence_length, hidden_units)
                variances = torch.var(hid_layer, dim=0, keepdim=True)
                means = torch.mean(hid_layer, dim=0, keepdim=True)
                mean_vars = torch.mean(variances)
                loss_mean += torch.sum(means ** 2)
                loss_var += torch.sum((variances - mean_vars) ** 2)
        return loss_mean + loss_var

    @staticmethod
    def mean_accuracy(weights, targets):
        sum = 0
        _, batch_size, seq_len = targets.size()
        for i, weight in enumerate(weights):
            w = weight.contiguous().view(batch_size * seq_len, -1)
            t = targets[i, :, :].contiguous().view(-1)
            max_values, max_indices = w.max(1)
            correct = max_indices == t
            sum += correct.float().mean()
        return sum / len(weights)

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        """
        :param weights: list (batch, num_notes) one for each voice
        since num_notes are different
        :param targets:(voice, batch)
        :return:
        """
        cross_entropy = nn.CrossEntropyLoss(size_average=True)
        _, batch_size, seq_len = targets.size()
        sum = 0
        for i, weight in enumerate(weights):
            w = weight.contiguous().view(batch_size * seq_len, -1)
            t = targets[i, :, :].contiguous().view(-1)
            ce = cross_entropy(w, t)
            sum += ce
        return sum / len(weights)


class AnticipationRNNBaselineTrainer(AnticipationRNNGaussianRegTrainer):
    def __init__(self, dataset,
                 model,
                 lr=1e-4,
                 early_stopping=False):
        super(AnticipationRNNBaselineTrainer, self).__init__(dataset, model, lr, early_stopping)
        self.constraint_prod = 0.5

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, metadata_tensor = batch
        # find constraint locations stochastically
        p = random.random() * 0.5
        constraint_loc = (torch.rand(*score_tensor[0, :, :].size()) < p).unsqueeze(0).repeat(score_tensor.size(0), 1, 1)

        # convert input to torch Variables
        score_tensor = to_cuda_variable_long(score_tensor)
        metadata_tensor = to_cuda_variable_long(metadata_tensor)
        constraint_loc = to_cuda_variable_long(constraint_loc)
        start_tick = None
        end_tick = None
        return score_tensor, metadata_tensor, constraint_loc, start_tick, end_tick


class AnticipationRNNAuxTrainer(AuxiliaryLossTrainer):
    def __init__(
            self, dataset: FolkDatasetNBars,
            model: ConstraintModelGaussianReg,
            lr=1e-3,
            subseq_len=48,
            type='recons'):
        """
        Initializes the class
        :param dataset: initializes FolkDatasetNBars object
        :param model: initialized ConstraintModelGaussianReg object
        :param lr: float, initial learning rate
        :param subseq_len: int, hyper-parameter for auxiliary loss
        :param type: str, specified type of auxiliary loss to be used
        """
        self.full_model = model
        super(AnticipationRNNAuxTrainer, self).__init__(
            dataset, self.full_model.lstm_constraint, lr, subseq_len, type
        )
        # Add the embedding layers to the optimizer also
        note_embed_params = self.full_model.note_embeddings.parameters()
        metadata_embed_params = self.full_model.metadata_embeddings.parameters()
        self.optimizer.add_param_group({"params": note_embed_params})
        self.optimizer.add_param_group({"params": metadata_embed_params})

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, metadata_tensor = batch
        # convert input to torch Variables
        score_tensor = to_cuda_variable_long(score_tensor)
        metadata_tensor = to_cuda_variable_long(metadata_tensor)
        m = self.full_model.embed_metadata(metadata_tensor, score_tensor)
        return m

    def loss_and_acc_for_batch(self, batch, epoch_num, train):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable, should be compatible as input to self.model
                (batch_size, seq_len, num_features)
        :param epoch_num: int, used to chenge training schedule
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        # compute the forward pass
        weights, _ = self.full_model.output_lstm_constraints(batch)

        if self.type == 'recons':
            aux_loss, aux_acc = self.recons_loss_and_acc(weights, batch)
        elif self.type == 'predic':
            aux_loss, aux_acc = self.predic_loss_and_acc(weights, batch)
        else:
            raise ValueError('Invalid type for auxiliary loss computation')
        return aux_loss, aux_acc


class AnticipationRNNTrainerWithAux(Trainer):
    def __init__(self, dataset: FolkDatasetNBars,
                 model: ConstraintModelGaussianReg):
        """
        Initializes the class
        :param dataset: initialized FolkDatasetNBars object
        :param model: initialized ConstraintModelGaussianReg
        """
        super(AnticipationRNNTrainerWithAux, self).__init__(dataset, model)
        self.gaussian_reg_trainer = AnticipationRNNGaussianRegTrainer(
            dataset, model
        )
        self.aux_trainer = AnticipationRNNAuxTrainer(
            dataset, model, type='recons'
        )
        # There are optimizers in the individual objects instantiated above
        self.optimizer = None
        # set scheduling variables
        self.num_pretraining_epochs = self.aux_trainer.num_pretraining_epochs

    def zero_grad(self):
        """
        Zero the grad of the relevant optimizers
        :return:
        """
        self.gaussian_reg_trainer.zero_grad()
        self.aux_trainer.zero_grad()

    def step(self):
        """
        Perform the backward pass and step update for all optimizers
        :return:
        """
        self.gaussian_reg_trainer.step()
        self.aux_trainer.step()

    def loss_and_acc_for_batch(self, batch, epoch_num, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        # extract batch data
        batch_data_reg, batch_data_aux = batch
        # compute auxiliary loss
        aux_loss, _ = self.aux_trainer.loss_and_acc_for_batch(batch_data_aux, epoch_num, train)
        if epoch_num < self.num_pretraining_epochs:
            loss = aux_loss
            accuracy = None
        else:
            # compute gaussian reg loss
            loss, accuracy = self.gaussian_reg_trainer.loss_and_acc_for_batch(batch_data_reg, epoch_num, train)
            loss += aux_loss
        return loss, accuracy

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: torch Variable or tuple of torch Variable objects
        """
        batch_data_reg = self.gaussian_reg_trainer.process_batch_data(batch)
        batch_data_aux = self.aux_trainer.process_batch_data(batch)
        return batch_data_reg, batch_data_aux

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        self.aux_trainer.update_scheduler(epoch_num)
